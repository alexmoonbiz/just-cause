"""Shared helpers: paths, jurisdiction names, corpus loading."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("NAV_ROOT") or Path(__file__).resolve().parent.parent)

STATES = {"CA", "NJ", "MA"}
CITIES = {
    "Berkeley, CA", "Los Angeles, CA", "San Francisco, CA", "San Diego, CA",
    "Santa Ana, CA", "Jersey City, NJ", "Hoboken, NJ", "Newark, NJ",
    "Boston, MA", "Cambridge, MA",
}
_STATE_NAMES = {"california": "CA", "new jersey": "NJ", "massachusetts": "MA"}
_CITY_BY_NAME = {c.split(",")[0].lower(): c for c in CITIES}

QUERY_DATE = "2026-10-01"


def normalize_jurisdiction(s: str | None) -> str | None:
    """'City of Boston', 'boston, ma', 'California', 'ca' -> canonical, else None."""
    if not s:
        return None
    t = re.sub(r"^(city|town) of\s+", "", s.strip(), flags=re.I).strip()
    low = t.lower()
    if t.upper() in STATES:
        return t.upper()
    if low in _STATE_NAMES:
        return _STATE_NAMES[low]
    name = low.split(",")[0].strip()
    if name in _CITY_BY_NAME:  # try as-is first: "Jersey City" must not lose its "City"
        return _CITY_BY_NAME[name]
    # Census style: "Hoboken city", "Cambridge city"
    return _CITY_BY_NAME.get(re.sub(r"\s+(city|town)$", "", name))


def level_of(jurisdiction: str) -> str:
    return "state" if jurisdiction in STATES else "city"


def state_of(jurisdiction: str) -> str:
    return jurisdiction if jurisdiction in STATES else jurisdiction.split(",")[-1].strip()


def split_header(raw: str) -> tuple[str, str, str]:
    """Corpus text files start with 'SOURCE: <url>' and 'RETRIEVED: <ts>' lines."""
    url = retrieved = ""
    lines = raw.splitlines()
    i = 0
    while i < min(len(lines), 4):
        if lines[i].startswith("SOURCE:"):
            url = lines[i].split(":", 1)[1].strip()
        elif lines[i].startswith("RETRIEVED:"):
            retrieved = lines[i].split(":", 1)[1].strip()
        elif lines[i].strip() == "":
            pass
        else:
            break
        i += 1
    return url, retrieved, "\n".join(lines[i:]).strip()


def load_docs(corpus_dir: Path) -> list[dict[str, Any]]:
    """Every document we have TEXT for: manifest rows first, then any stray .txt.

    `supplied` is True only for documents the organizers distributed with text.
    Anything else in corpus/text/ (a link-only page we saved ourselves) is supplementary:
    allowed for research, but it does not count toward the citation metric.
    """
    docs: list[dict[str, Any]] = []
    seen: set[str] = set()
    manifest = corpus_dir / "corpus_manifest.csv"
    rows_by_id: dict[str, dict] = {}
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as f:
            rows_by_id = {r["doc_id"]: r for r in csv.DictReader(f)}
        for row in rows_by_id.values():
            tf = (row.get("text_file") or "").strip()
            if not tf:
                continue
            path = corpus_dir / tf
            if not path.exists():
                continue
            url, retrieved, _ = split_header(path.read_text(encoding="utf-8", errors="replace"))
            docs.append({
                "doc_id": row["doc_id"],
                "path": path,
                "url": row.get("url") or url,
                "retrieved": row.get("retrieved_at") or retrieved,
                "jurisdiction_hint": row.get("jurisdictions") or "",
                "supplied": True,  # part of the organizers' distributed corpus
            })
            seen.add(path.name)
    for path in sorted((corpus_dir / "text").glob("*.txt")):
        if path.name in seen:
            continue
        url, retrieved, _ = split_header(path.read_text(encoding="utf-8", errors="replace"))
        m = rows_by_id.get(path.stem, {})  # a text file you added for a link-only manifest row
        docs.append({
            "doc_id": path.stem, "path": path, "url": m.get("url") or url,
            "retrieved": retrieved or m.get("retrieved_at", ""),
            "jurisdiction_hint": m.get("jurisdictions", ""),
            "supplied": False,  # saved by us: research only, not part of the citation-counted corpus
        })
    return docs
