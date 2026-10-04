"""
Module B + C: apply extracted rules to addresses, for any as-of date, and run the
change-tracking tests. Pure Python, deterministic, no LLM, no network.

    python src/engine.py                          # lookups.json (2026-10-01) + changes.json
    python src/engine.py --as-of 2027-07-02       # lookups for another date -> out/lookups_2027-07-02.json
    python src/engine.py --explain A0001

The same Engine class powers the Streamlit app, so the UI's as-of picker recomputes
live from the same code that produced the submitted files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402
from dsl import (APPLIES, BASE_AS_OF, NOT_APPLICABLE, NOT_YET_EFFECTIVE, PENDING, SUPERSEDED,  # noqa: E402
                 UNKNOWN, resolve)

CATEGORY_ORDER = ["rent_increase_limits", "just_cause_eviction", "security_deposits",
                  "application_screening_fees", "screening_restrictions", "algorithmic_rent_setting"]
RANK = {APPLIES: 5, UNKNOWN: 4, NOT_YET_EFFECTIVE: 3, PENDING: 2, SUPERSEDED: 1}


_WORD_NUM = {"five": 5, "four": 4, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def derive_units(state: str, use_code: str, desc: str) -> tuple[int, int | None] | None:
    """Unit-count range implied by the assessor's use code/description, or None.

    Only patterns the data states outright. Examples from the sample:
      "Five or more apartments" -> (5, None)   "APT 7-30 UNITS" -> (7, 30)
      "4-8-UNIT-APT" -> (4, 8)   ">8-UNIT-APT" -> (9, None)   "(5+ units)" -> (5, None)
      "Apartment 5 to 14 Units" -> (5, 14)   NJ property class 4C (apartments) -> (5, None)
    """
    d = (desc or "").lower()
    m = re.search(r"(\d+)\s*(?:to|-)\s*(\d+)[\s-]*units?", d)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r">\s*(\d+)[\s-]*unit", d)
    if m:
        return int(m.group(1)) + 1, None
    m = re.search(r"(\d+)\+\s*units?", d) or re.search(r"(\d+)\s*units?\s*or\s*more", d)
    if m:
        return int(m.group(1)), None
    m = re.search(r"(\w+)\s+or\s+more\s+apartments", d)
    if m and m.group(1) in _WORD_NUM:
        return _WORD_NUM[m.group(1)], None
    m = re.search(r"(\d+)\s*units?\s*or\s*(?:less|fewer)", d)
    if m:
        return 1, int(m.group(1))
    if state == "NJ" and (use_code or "").strip().upper() == "4C":
        return 5, None  # NJ MOD-IV class 4C = apartments of five or more units
    return None


def _int(v: str | None) -> int | None:
    try:
        return int(float(v)) if v not in (None, "") else None
    except ValueError:
        return None


class Engine:
    def __init__(self, root: Path = ROOT):
        self.root = root
        self.rules: list[dict] = json.loads((root / "out" / "rules.json").read_text())["rules"]
        ext_path = root / "build" / "rules_ext.json"
        self.ext: dict[str, dict] = json.loads(ext_path.read_text()) if ext_path.exists() else {}
        self.geo: dict[str, dict] = json.loads((root / "build" / "addresses_geo.json").read_text())
        with (root / "data" / "sample_addresses.csv").open(newline="", encoding="utf-8") as f:
            self.addresses = {r["address_id"]: r for r in csv.DictReader(f)}
        self.by_id = {r["team_rule_id"]: r for r in self.rules}
        self.declared_conflicts: list[tuple[list[dict], list[dict], str]] = []

    # ------------------------------------------------------------ helpers ----
    def facts(self, aid: str) -> dict[str, Any]:
        a = self.addresses[aid]
        units = _int(a.get("units"))
        rng = None if units is not None else derive_units(a["state"], a.get("use_code"), a.get("use_description"))
        return {"year_built": _int(a.get("year_built")), "units": units, "units_range": rng,
                "use_code": (a.get("use_code") or None), "owner_type": None, "occupancy_date": None}

    def in_scope(self, rule: dict, aid: str) -> bool:
        g = self.geo[aid]
        return g["state"] == rule["jurisdiction"] if rule["level"] == "state" else g.get("city") == rule["jurisdiction"]

    def select(self, sel: dict | list) -> list[dict]:
        """Pick rules for a test. Selection is data (config/tests.json), reviewed by a human."""
        if isinstance(sel, list):
            out: list[dict] = []
            for s in sel:
                out += [r for r in self.select(s) if r not in out]
            return out
        if sel.get("team_rule_ids"):
            return [self.by_id[i] for i in sel["team_rule_ids"] if i in self.by_id]
        pat = re.compile(sel["text"], re.I) if sel.get("text") else None
        out = []
        for r in self.rules:
            if sel.get("jurisdiction") and r["jurisdiction"] != sel["jurisdiction"]:
                continue
            if sel.get("level") and r["level"] != sel["level"]:
                continue
            if sel.get("category") and r["category"] != sel["category"]:
                continue
            if sel.get("status") and r["status"] != sel["status"]:
                continue
            if pat and not pat.search(" ".join([r["title"], r["citation"], r["requirement"]])):
                continue
            out.append(r)
        return out

    # ----------------------------------------------------------- evaluation ----
    def evaluate(self, aid: str, as_of: date, rules: list[dict] | None = None,
                 assume_enacted: bool = False) -> list[dict]:
        facts, g, entries = self.facts(aid), self.geo[aid], {}
        for r in (rules if rules is not None else self.rules):
            if not self.in_scope(r, aid):
                continue
            rr = {**r, "status": "in_force", "effective_date": None} if assume_enacted else r
            res, notes = resolve(rr, self.ext.get(r["team_rule_id"], {}), facts, as_of)
            if res == NOT_APPLICABLE:
                continue
            if r["level"] == "city" and g.get("uncertain") and res == APPLIES:
                res = UNKNOWN
                notes = notes + [f"the legal city could not be confirmed (mailing city "
                                 f"'{g.get('postal_city')}' is unreliable and the Census lookup was unavailable)"]
            entries[r["team_rule_id"]] = {"team_rule_id": r["team_rule_id"], "result": res,
                                          "conflict_flag": False, "_notes": notes, "_rule": r}
        self._precedence(entries)
        self._conflicts(entries, as_of)
        for e in entries.values():
            e["explanation"] = self._explain(e, aid, as_of)
        return sorted(entries.values(), key=lambda e: (
            CATEGORY_ORDER.index(e["_rule"]["category"]), e["_rule"]["level"] != "state", e["team_rule_id"]))

    def _precedence(self, entries: dict[str, dict]) -> None:
        for e in entries.values():
            if e["result"] != APPLIES:
                continue
            for oid in e["_rule"].get("overrides") or []:
                o = entries.get(oid)
                if o and o["result"] == APPLIES:
                    e["result"] = SUPERSEDED
                    e["_notes"].append(f"yields to the stricter local rule {oid} ({o['_rule']['title']})")
                    break

    def _conflicts(self, entries: dict[str, dict], as_of: date) -> None:
        for e in entries.values():
            if not e["_rule"].get("conflict_flag"):
                continue
            for o in entries.values():
                if o is not e and o["_rule"]["category"] == e["_rule"]["category"] \
                        and o["_rule"]["level"] != e["_rule"]["level"]:
                    for x in (e, o):
                        x["conflict_flag"] = True
                    note = e["_rule"].get("conflict_note") or "possible conflict between state and local rules"
                    e["_notes"].append(f"conflict for human review: {note}")
        for A, B, note in self.declared_conflicts:
            ids_a = {r["team_rule_id"] for r in A}
            ids_b = {r["team_rule_id"] for r in B}
            pa = [e for i, e in entries.items() if i in ids_a]
            pb = [e for i, e in entries.items() if i in ids_b]
            if pa and pb:
                for x in pa + pb:
                    x["conflict_flag"] = True
                    x["_notes"].append(f"conflict for human review: {note}")

    def _explain(self, e: dict, aid: str, as_of: date) -> str:
        r, a = e["_rule"], self.addresses[aid]
        eff = r.get("effective_date") or "date not stated"
        rng = self.facts(aid)["units_range"]
        units_txt = (f"{a['units']} units" if a.get("units") else
                     (f"{rng[0]}{'+' if rng[1] is None else '-' + str(rng[1])} units per use code" if rng
                      else "unit count not in data"))
        bits = [units_txt,
                f"built {a['year_built']}" if a.get("year_built") else "year built not in data"]
        building = ", ".join(bits)
        src = f"[{r.get('source_doc_id')}, retrieved {(r.get('retrieved_at') or '')[:10] or 'n/a'}]"
        res = e["result"]
        head = f"{r['title']} ({r['citation']})"
        if res == APPLIES:
            body = f"{head} is in force (effective {eff}) and covers this building ({building})."
            if r.get("key_value"):
                body += f" Key term: {r['key_value']}."
        elif res == NOT_YET_EFFECTIVE:
            body = f"{head} is enacted but not effective until {eff}, so it is not in force on {as_of}."
        elif res == PENDING:
            body = f"{head} is a pending bill, not law. It would matter here only if enacted."
        elif res == SUPERSEDED:
            body = f"{head} would cover this building, but " + "; ".join(n for n in e["_notes"] if "yields" in n) + "."
        else:
            why = "; ".join(n for n in e["_notes"] if "conflict for" not in n) or "coverage depends on facts not in the data"
            body = f"{head}: cannot confirm coverage. {why}."
        flags = [n for n in e["_notes"] if n.startswith("conflict for")]
        return " ".join([body] + [f.capitalize() + "." for f in flags] + [src])

    # ---------------------------------------------------------------- output ----
    def lookups(self, as_of: date) -> dict:
        out = {}
        for aid in self.addresses:
            out[aid] = [{"team_rule_id": e["team_rule_id"], "result": e["result"],
                         "explanation": e["explanation"], "conflict_flag": e["conflict_flag"]}
                        for e in self.evaluate(aid, as_of)]
        return {"as_of": as_of.isoformat(), "lookups": out}

    # ------------------------------------------------------------- change tests ----
    @staticmethod
    def _best(entries: list[dict]) -> str:
        return max((e["result"] for e in entries), key=lambda r: RANK.get(r, 0)) if entries else "none"

    def run_tests(self, tests: dict) -> tuple[dict, dict]:
        # declared conflicts first, so lookups and changes agree
        for cfg in tests.values():
            if cfg.get("conflict_with"):
                self.declared_conflicts.append((self.select(cfg["selector"]), self.select(cfg["conflict_with"]),
                                                cfg.get("conflict_note", "possible preemption")))
        out, used = {}, {}
        for tid, cfg in tests.items():
            sel = self.select(cfg["selector"])
            used[tid] = [{"team_rule_id": r["team_rule_id"], "jurisdiction": r["jurisdiction"],
                          "title": r["title"], "status": r["status"], "effective_date": r["effective_date"]}
                         for r in sel]
            affected, conflicts, warn = [], [], []
            kind = cfg["type"]
            for aid in self.addresses:
                if kind == "as_of":
                    b = self._best(self.evaluate(aid, date.fromisoformat(cfg["before"]), sel))
                    a = self._best(self.evaluate(aid, date.fromisoformat(cfg["after"]), sel))
                    hit = b != a
                elif kind == "boundary":
                    hit = bool(self.evaluate(aid, date.fromisoformat(cfg["as_of"]), sel))
                elif kind == "pending":
                    hit = any(e["result"] in (APPLIES, UNKNOWN) for e in
                              self.evaluate(aid, date.fromisoformat(cfg["as_of"]), sel, assume_enacted=True))
                elif kind == "negative":
                    hit = any(e["result"] == APPLIES for e in self.evaluate(aid, date.fromisoformat(cfg["as_of"]), sel))
                else:
                    raise ValueError(f"unknown test type {kind}")
                if hit:
                    affected.append(aid)
                if cfg.get("conflict_with"):
                    when = date.fromisoformat(cfg.get("after") or cfg["as_of"])
                    ents = self.evaluate(aid, when, self.select(cfg["selector"]) + self.select(cfg["conflict_with"]))
                    if any(e["conflict_flag"] for e in ents):
                        conflicts.append(aid)
            gap = cfg.get("known_gap") if not sel else None
            if not sel and not gap:
                warn.append("NO RULES MATCHED this test's selector: extraction missed it, or the selector in config/tests.json needs adjusting")
            if kind == "negative" and affected:
                warn.append(f"VIOLATION: {len(affected)} addresses show a rent cap that this test requires to be empty")
            ids = ", ".join(r["team_rule_id"] for r in sel) or "none"
            out[tid] = {"affected_address_ids": affected, "conflict_flag_address_ids": conflicts,
                        "notes": " ".join(x for x in [cfg.get("describe", ""), f"Rules used: {ids}.", gap or "",
                                                      *warn] if x)}
            for w in warn:
                print(f"  !! {tid}: {w}", file=sys.stderr)
            if gap:
                print(f"  -- {tid}: known gap declared in config/tests.json", file=sys.stderr)
        return out, used


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--as-of", default=BASE_AS_OF.isoformat())
    p.add_argument("--tests", type=Path, default=ROOT / "config" / "tests.json")
    p.add_argument("--explain")
    a = p.parse_args()
    as_of = date.fromisoformat(a.as_of)

    eng = Engine()
    tests = json.loads(a.tests.read_text())
    changes, used = eng.run_tests(tests)  # registers declared conflicts before any lookup

    if a.explain:
        for e in eng.evaluate(a.explain, as_of):
            print(f"{e['team_rule_id']:8} {e['result']:18} conflict={e['conflict_flag']}  {e['explanation']}")
        return

    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    lk = eng.lookups(as_of)
    name = "lookups.json" if as_of == BASE_AS_OF else f"lookups_{as_of}.json"
    (out / name).write_text(json.dumps(lk, indent=1), encoding="utf-8")
    if as_of == BASE_AS_OF:
        (out / "changes.json").write_text(json.dumps(changes, indent=1), encoding="utf-8")
        (out / "test_selection.json").write_text(json.dumps(used, indent=1), encoding="utf-8")

    from collections import Counter
    c = Counter(e["result"] for v in lk["lookups"].values() for e in v)
    print(f"{len(lk['lookups'])} addresses, {sum(c.values())} lookups -> out/{name}   {dict(c)}")
    if as_of == BASE_AS_OF:
        for tid, v in changes.items():
            print(f"  {tid}: {len(v['affected_address_ids']):3} affected, {len(v['conflict_flag_address_ids']):3} conflict-flagged")


if __name__ == "__main__":
    main()
