"""
Module B, step 1: resolve each address to its REAL state and incorporated place.

The mailing city is not the legal city ("Dorchester" is Boston; LA County parcels
include addresses that say "Los Angeles" but sit in another city). The Census
Geocoder answers this with no API key.

    python src/geocode.py              # Census batch + per-point place lookup (needs internet)
    python src/geocode.py --offline    # mailing-city fallback only, for development

Output: build/addresses_geo.json
    {address_id: {state, city, source, matched, uncertain, ...}}
`city` is "City, ST" matching the jurisdiction names in rules.json, or null.
`uncertain` is true when we fell back to the mailing city for a city where the
mailing city is known to be unreliable; the engine then answers "unknown" for
city rules instead of guessing.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CITIES, ROOT, normalize_jurisdiction  # noqa: E402

BATCH_URL = "https://geocoding.geo.census.gov/geocoder/geographies/addressbatch"
COORD_URL = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
PARAMS = {"benchmark": "Public_AR_Current", "vintage": "Current_Current"}

# Mailing-city -> legal city, used ONLY when the geocoder cannot match an address.
FALLBACK = {
    ("MA", n): "Boston, MA" for n in [
        "boston", "dorchester", "roxbury", "east boston", "brighton", "allston", "south boston",
        "jamaica plain", "hyde park", "mattapan", "charlestown", "west roxbury", "roslindale",
        "back bay", "fenway", "south end", "north end", "beacon hill"]
}
FALLBACK.update({("MA", "cambridge"): "Cambridge, MA", ("CA", "san ysidro"): "San Diego, CA"})
# Mailing city is NOT reliable evidence of the legal city for these (county-wide parcel data).
UNRELIABLE_MAILING = {"Los Angeles, CA", "San Diego, CA", "Berkeley, CA"}


def fallback_city(state: str, postal_city: str) -> str | None:
    key = (state, postal_city.strip().lower())
    if key in FALLBACK:
        return FALLBACK[key]
    return normalize_jurisdiction(f"{postal_city}, {state}")


def read_addresses(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def census_batch(rows: list[dict], cache: Path) -> dict[str, dict]:
    import requests
    if cache.exists():
        text = cache.read_text(encoding="utf-8")
    else:
        buf = io.StringIO()
        w = csv.writer(buf)
        for r in rows:
            w.writerow([r["address_id"], r["street_address"], r["postal_city"], r["state"], r["zip"]])
        resp = requests.post(BATCH_URL, data=PARAMS, timeout=300,
                             files={"addressFile": ("addresses.csv", buf.getvalue().encode())})
        resp.raise_for_status()
        text = resp.text
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
    out = {}
    for rec in csv.reader(io.StringIO(text)):
        if len(rec) < 3:
            continue
        aid, matched = rec[0], rec[2] == "Match"
        item = {"matched": matched}
        if matched and len(rec) >= 12 and rec[5]:
            lon, lat = rec[5].split(",")
            item.update(lon=float(lon), lat=float(lat), state_fips=rec[8], county_fips=rec[9])
        out[aid] = item
    return out


def census_place(aid: str, lon: float, lat: float, cache_dir: Path) -> str | None:
    """Incorporated place containing the point, e.g. 'Hoboken city' -> 'Hoboken'."""
    import requests
    cache = cache_dir / f"{aid}.json"
    if cache.exists():
        data = json.loads(cache.read_text())
    else:
        for attempt in range(4):
            try:
                r = requests.get(COORD_URL, params={**PARAMS, "x": lon, "y": lat, "layers": "all",
                                                    "format": "json"}, timeout=60)
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                time.sleep(2 ** attempt)
        else:
            return "__error__"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))
    geos = data.get("result", {}).get("geographies", {})
    for key, vals in geos.items():
        if "Incorporated Place" in key and vals:
            return vals[0].get("BASENAME") or vals[0].get("NAME")
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--addresses", type=Path, default=ROOT / "data" / "sample_addresses.csv")
    p.add_argument("--out", type=Path, default=ROOT / "build" / "addresses_geo.json")
    p.add_argument("--offline", action="store_true")
    a = p.parse_args()

    rows = read_addresses(a.addresses)
    geo: dict[str, dict] = {}

    census: dict[str, dict] = {}
    if not a.offline:
        print(f"Census batch geocoding {len(rows)} addresses...")
        census = census_batch(rows, ROOT / "cache" / "census_batch.csv")
        pts = [(r["address_id"], census[r["address_id"]]) for r in rows
               if census.get(r["address_id"], {}).get("matched") and "lon" in census[r["address_id"]]]
        print(f"  matched {len(pts)}/{len(rows)}; looking up the incorporated place for each point...")
        places: dict[str, str | None] = {}
        with ThreadPoolExecutor(max_workers=6) as ex:
            for (aid, c), place in zip(pts, ex.map(
                    lambda t: census_place(t[0], t[1]["lon"], t[1]["lat"], ROOT / "cache" / "census_place"), pts)):
                places[aid] = place
    else:
        places = {}

    stats = {"census": 0, "fallback": 0, "unincorporated": 0, "unresolved": 0}
    for r in rows:
        aid, st = r["address_id"], r["state"]
        rec = {"state": st, "postal_city": r["postal_city"], "matched": False,
               "city": None, "source": "", "uncertain": False}
        c = census.get(aid, {})
        place = places.get(aid)
        if c.get("matched") and place not in (None, "__error__"):
            rec.update(matched=True, source="census", lon=c.get("lon"), lat=c.get("lat"),
                       county_fips=c.get("county_fips"), place_raw=place,
                       city=normalize_jurisdiction(f"{place}, {st}"))
            stats["census"] += 1
            if rec["city"] is None:
                rec["city_name_unrecognized"] = place  # a real city we hold no rules for
        elif c.get("matched") and place is None:
            rec.update(matched=True, source="census", lon=c.get("lon"), lat=c.get("lat"),
                       county_fips=c.get("county_fips"), place_raw=None)
            stats["unincorporated"] += 1
        else:
            fb = fallback_city(st, r["postal_city"])
            rec.update(source="mailing_city_fallback", city=fb)
            rec["uncertain"] = fb in UNRELIABLE_MAILING
            stats["fallback" if fb else "unresolved"] += 1
        geo[aid] = rec

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(geo, indent=1), encoding="utf-8")
    print(f"\n{len(rows)} addresses -> {a.out}  {stats}")
    by = {}
    for g in geo.values():
        by[g["city"]] = by.get(g["city"], 0) + 1
    for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"  {str(k):22}{v}")
    if a.offline:
        print("\n(offline mode: mailing city only. Rerun without --offline for legal-city boundaries.)")


if __name__ == "__main__":
    main()
