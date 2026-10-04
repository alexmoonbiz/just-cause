"""Offline tests: DSL logic, verifier, and T1-T5 end to end on the real 500 addresses."""

import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dsl import (APPLIES, NOT_YET_EFFECTIVE, PENDING, UNKNOWN, Tri, evaluate_coverage,  # noqa: E402
                 parse_interval, resolve, temporal_status, tri_and, tri_or)


# ------------------------------------------------------------------ DSL ----
def cond(f, op, v):
    return {"field": f, "op": op, "value": v}


def cov(c, ex=(), facts=None, unstructured=None):
    return evaluate_coverage(c, ex, facts or {}, unstructured)[0]


def test_three_valued_logic():
    assert tri_and([Tri.FALSE, Tri.UNKNOWN]) is Tri.FALSE      # false dominates
    assert tri_and([Tri.TRUE, Tri.UNKNOWN]) is Tri.UNKNOWN
    assert tri_or([Tri.TRUE, Tri.UNKNOWN]) is Tri.TRUE
    assert tri_or([Tri.FALSE, Tri.UNKNOWN]) is Tri.UNKNOWN
    assert tri_and([]) is Tri.TRUE and tri_or([]) is Tri.FALSE


def test_missing_fact_is_unknown_never_false():
    assert cov([cond("units", ">", "4")], facts={"units": None}) is Tri.UNKNOWN
    assert cov([cond("year_built", "<=", "1979")], facts={}) is Tri.UNKNOWN


def test_certificate_cutoff_uses_year_as_interval():
    sf = [cond("occupancy_date", "<=", "1979-06-13")]
    assert cov(sf, facts={"year_built": 1962}) is Tri.TRUE
    assert cov(sf, facts={"year_built": 1985}) is Tri.FALSE
    assert cov(sf, facts={"year_built": 1979}) is Tri.UNKNOWN     # README: cutoff year -> unknown
    assert cov(sf, facts={"year_built": None}) is Tri.UNKNOWN
    la = [cond("occupancy_date", "<=", "1978-10-01")]
    assert cov(la, facts={"year_built": 1978}) is Tri.UNKNOWN
    assert cov(la, facts={"year_built": 1977}) is Tri.TRUE


def test_date_cutoff_written_on_year_built_compares_the_whole_year():
    la = [cond("year_built", "<=", "1978-10-01")]
    assert cov(la, facts={"year_built": 1927}) is Tri.TRUE
    assert cov(la, facts={"year_built": 1990}) is Tri.FALSE
    assert cov(la, facts={"year_built": 1978}) is Tri.UNKNOWN
    new = [cond("year_built", ">=", "2011-04-01")]
    assert cov(new, facts={"year_built": 2015}) is Tri.TRUE
    assert cov(new, facts={"year_built": 1960}) is Tri.FALSE
    assert cov(new, facts={"year_built": 2011}) is Tri.UNKNOWN


def test_exemption_group_is_an_AND_so_big_buildings_escape_owner_occupied_exemption():
    nj_deposit_exemption = [{"all_of": [cond("owner_type", "==", "owner_occupied"), cond("units", "<=", "2")]}]
    # 32 units: group is FALSE even though owner_type is unknown -> rule applies
    assert cov([], nj_deposit_exemption, {"units": 32}) is Tri.TRUE
    # units unknown + owner unknown -> unknown
    assert cov([], nj_deposit_exemption, {"units": None}) is Tri.UNKNOWN
    # 2 units, owner unknown -> unknown (the exception cannot be ruled out)
    assert cov([], nj_deposit_exemption, {"units": 2}) is Tri.UNKNOWN


def test_unstructured_condition_caps_applies_at_unknown():
    assert cov([], [], {"units": 10}, unstructured="new-construction filing") is Tri.UNKNOWN


def test_partial_dates_never_give_false_certainty():
    assert parse_interval("2026") == (date(2026, 1, 1), date(2026, 12, 31))
    assert temporal_status("in_force", "2026-03", date(2026, 3, 15)) == UNKNOWN
    assert temporal_status("in_force", "2026-03", date(2026, 4, 1)) is None


def test_T1_boundary_and_T3_dates():
    assert temporal_status("in_force", "2026-01-01", date(2025, 12, 31)) == NOT_YET_EFFECTIVE
    assert temporal_status("in_force", "2026-01-01", date(2026, 1, 2)) is None
    assert temporal_status("not_yet_effective", "2027-07-01", date(2026, 10, 1)) == NOT_YET_EFFECTIVE
    assert temporal_status("not_yet_effective", "2027-07-01", date(2027, 7, 2)) is None
    assert temporal_status("pending", None, date(2030, 1, 1)) == PENDING   # a bill never becomes law by waiting
    assert temporal_status("failed", None, date(2026, 10, 1)) == "not_applicable"


# ------------------------------------------------------------- verifier ----
@pytest.fixture()
def sandbox(tmp_path):
    for d in ("corpus", "schema", "data", "config"):
        shutil.copytree(ROOT / d, tmp_path / d)
    (tmp_path / "src").symlink_to(ROOT / "src")
    for d in ("build", "out", "runs"):
        (tmp_path / d).mkdir()
    return tmp_path


def run(sandbox, script, *args):
    return subprocess.run([sys.executable, str(ROOT / "src" / script), *args], cwd=sandbox,
                          capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin:/usr/local/bin", "PYTHONPATH": str(ROOT / "src"), "NAV_ROOT": str(sandbox)})


def raw_rule(rid, **kw):
    base = {"team_rule_id": rid, "jurisdiction": "CA", "level": "state", "category": "algorithmic_rent_setting",
            "status": "in_force", "title": "T", "requirement": "R", "key_value": None,
            "coverage_conditions": None, "exemptions": None, "interaction": None, "effective_date": "2026-01-01",
            "citation": "C", "quoted_span": "x" * 30, "confidence": 0.9, "conflict_flag": False,
            "conflict_note": None, "defers_to_local": False, "coverage_dsl": [], "exemption_dsl": [],
            "unstructured_conditions": None, "source_doc_id": "D001", "source_url": "u", "retrieved_at": "2026-10-01"}
    base.update(kw)
    return base


def test_verifier_proves_trims_and_drops(sandbox):
    body = (sandbox / "corpus/text/D001.txt").read_text().split("\n", 3)[3]
    sent = " ".join(body.split()[200:240])                      # a real passage, whitespace-collapsed
    good = raw_rule("r-0001", quoted_span=sent)
    drift = raw_rule("r-0002", quoted_span=sent + " and then the model invented a trailing clause")
    fake = raw_rule("r-0003", quoted_span="The landlord shall provide a free puppy to every tenant at move in.")
    bad_status = raw_rule("r-0004", quoted_span=sent, status="in_force", effective_date="2999-01-01")
    (sandbox / "build/rules_raw.json").write_text(json.dumps({"rules": [good, drift, fake, bad_status]}))
    r = run(sandbox, "verify.py")
    assert r.returncode == 0, r.stderr
    out = {x["team_rule_id"]: x for x in json.loads((sandbox / "out/rules.json").read_text())["rules"]}
    assert set(out) == {"r-0001", "r-0002", "r-0004"}           # fabricated quote dropped
    assert out["r-0001"]["quoted_span"] in body                 # byte-exact source text, newlines and all
    assert out["r-0002"]["quoted_span"] in body                 # drifted quote trimmed to what is provable
    assert out["r-0004"]["status"] == "not_yet_effective"       # contradiction repaired against 2026-10-01
    ext = json.loads((sandbox / "build/rules_ext.json").read_text())
    assert ext["r-0002"]["citation_check"] == "trimmed_to_provable_prefix"


# ---------------------------------------------------- T1-T5 end to end ----
def build_world(sandbox):
    """Synthetic rules shaped like the five tests, run over the REAL 500 addresses."""
    R = lambda rid, **k: raw_rule(rid, **k)  # noqa: E731
    rules = [
        R("r-0001", title="CA AB 325", effective_date="2026-01-01"),
        R("r-0002", jurisdiction="Hoboken, NJ", level="city", title="Hoboken ban", effective_date="2025-06-01"),
        R("r-0003", jurisdiction="Jersey City, NJ", level="city", title="JC ban", effective_date="2025-12-01"),
        R("r-0004", jurisdiction="NJ", title="NJ FAIR Act", status="not_yet_effective", effective_date="2027-07-01",
          conflict_flag=True, conflict_note="may preempt local bans"),
        R("r-0005", jurisdiction="MA", title="MA S.2983", status="pending", effective_date=None),
        R("r-0006", jurisdiction="MA", title="IP 25-21 rent control", category="rent_increase_limits",
          status="failed", effective_date=None),
        R("r-0007", jurisdiction="SF-like", level="state"),  # junk jurisdiction, must be dropped
        R("r-0008", jurisdiction="San Francisco, CA", level="city", category="rent_increase_limits",
          title="SF rent ordinance", coverage_dsl=[cond("occupancy_date", "<=", "1979-06-13")]),
        R("r-0009", jurisdiction="CA", category="rent_increase_limits", title="AB 1482", defers_to_local=True),
        R("r-0010", jurisdiction="NJ", category="security_deposits", title="NJ deposits",
          exemption_dsl=[{"all_of": [cond("owner_type", "==", "owner_occupied"), cond("units", "<=", "2")]}]),
    ]
    # give every rule a real quote from D001 so the verifier keeps them
    body = (sandbox / "corpus/text/D001.txt").read_text().split("\n", 3)[3]
    for i, r in enumerate(rules):
        r["quoted_span"] = " ".join(body.split()[300 + i * 40: 330 + i * 40])
    (sandbox / "build/rules_raw.json").write_text(json.dumps({"rules": rules}))
    assert run(sandbox, "verify.py").returncode == 0
    assert run(sandbox, "geocode.py", "--offline").returncode == 0
    r = run(sandbox, "engine.py")
    assert r.returncode == 0, r.stderr
    return json.loads((sandbox / "out/changes.json").read_text()), json.loads((sandbox / "out/lookups.json").read_text())


def test_T1_to_T5_and_submission_shape(sandbox):
    changes, lk = build_world(sandbox)
    import csv
    addr = list(csv.DictReader((sandbox / "data/sample_addresses.csv").open()))
    ids = lambda pred: {a["address_id"] for a in addr if pred(a)}  # noqa: E731

    assert set(changes["T1"]["affected_address_ids"]) == ids(lambda a: a["state"] == "CA")          # 80+80+50+40
    assert set(changes["T2"]["affected_address_ids"]) == ids(lambda a: a["postal_city"] in ("Hoboken", "Jersey City"))
    assert not ids(lambda a: a["postal_city"] == "Newark") & set(changes["T2"]["affected_address_ids"])
    assert set(changes["T3"]["affected_address_ids"]) == ids(lambda a: a["state"] == "NJ")
    assert set(changes["T3"]["conflict_flag_address_ids"]) == ids(lambda a: a["postal_city"] in ("Hoboken", "Jersey City"))
    assert set(changes["T4"]["affected_address_ids"]) == ids(lambda a: a["state"] == "MA")
    assert changes["T5"]["affected_address_ids"] == []                                             # empty by law

    # submission shape: all 500 addresses, allowed result values, nothing outside the README's list
    assert lk["as_of"] == "2026-10-01" and len(lk["lookups"]) == 500
    allowed = {"applies", "unknown", "superseded", "not_yet_effective", "pending"}
    entries = [e for v in lk["lookups"].values() for e in v]
    assert {e["result"] for e in entries} <= allowed
    assert all(set(e) == {"team_rule_id", "result", "explanation", "conflict_flag"} for e in entries)

    by = lambda aid: {e["team_rule_id"]: e for e in lk["lookups"][aid]}  # noqa: E731
    some_ma = next(a["address_id"] for a in addr if a["state"] == "MA")
    assert by(some_ma)["r-0005"]["result"] == "pending"                                            # T4: reported, never "applies"
    assert not any(e["team_rule_id"] == "r-0006" for e in lk["lookups"][some_ma])                   # T5: failed measure absent

    some_sf = next(a for a in addr if a["postal_city"] == "San Francisco")
    sf = by(some_sf["address_id"])
    if some_sf["year_built"] == "1979":
        assert sf["r-0008"]["result"] == "unknown"
    # the CA statewide cap yields where the stricter SF rule applies (SF rule applies or is unknown for this building)
    assert sf["r-0009"]["result"] in ("superseded", "applies")

    # NJ rows have no unit counts, but class 4C means 5+ units, so the owner-occupied (<=2 units)
    # exemption provably cannot hold: the rule applies. (The sample template makes the same call.)
    nj_4c = next(a["address_id"] for a in addr if a["state"] == "NJ" and a["use_code"] == "4C")
    assert by(nj_4c)["r-0010"]["result"] == "applies"
    assert "5+ units per use code" in by(nj_4c)["r-0010"]["explanation"]

    # junk jurisdiction dropped by verifier
    assert "r-0007" not in json.loads((sandbox / "out/rules.json").read_text()).__str__()


def test_other_as_of_dates(sandbox):
    build_world(sandbox)
    assert run(sandbox, "engine.py", "--as-of", "2027-07-02").returncode == 0
    lk = json.loads((sandbox / "out/lookups_2027-07-02.json").read_text())
    nj = next(v for k, v in lk["lookups"].items() if any(e["team_rule_id"] == "r-0004" for e in v))
    assert {e["team_rule_id"]: e["result"] for e in nj}["r-0004"] == "applies"


def test_unit_ranges_inferred_from_use_description_only_when_stated():
    from engine import derive_units
    assert derive_units("CA", "0500", "Five or more apartments") == (5, None)
    assert derive_units("MA", "A/112", "APT 7-30 UNITS") == (7, 30)
    assert derive_units("MA", "111", "4-8-UNIT-APT") == (4, 8)
    assert derive_units("MA", "112", ">8-UNIT-APT") == (9, None)
    assert derive_units("CA", "7700", "Alameda County use code (5+ units)") == (5, None)
    assert derive_units("CA", "A5", "Apartment 5 to 14 Units") == (5, 14)
    assert derive_units("CA", "A15", "Apartment 15 Units or more") == (15, None)
    assert derive_units("CA", "TIC", "TIC Bldg 4 units or less") == (1, 4)
    assert derive_units("NJ", "4C", "3SB") == (5, None)
    assert derive_units("MA", "A/120", "LUXURY APARTMENT") is None       # says nothing: stay unknown
    assert derive_units("MA", "A/125", "SUBSD HOUSING S- 8") is None


def test_unit_range_decides_thresholds_but_not_straddles():
    f = {"units": None, "units_range": (7, 30)}
    assert cov([cond("units", ">", "4")], facts=f) is Tri.TRUE
    assert cov([cond("units", ">", "20")], facts=f) is Tri.UNKNOWN
    assert cov([cond("units", "<=", "2")], facts=f) is Tri.FALSE
    assert cov([cond("units", ">=", "5")], facts={"units": None, "units_range": (5, None)}) is Tri.TRUE


def test_supplementary_text_is_skipped_and_held_out_by_default(sandbox):
    from common import load_docs
    body = (sandbox / "corpus/text/D001.txt").read_text()
    (sandbox / "corpus/text/D032.txt").write_text("SOURCE: https://ecode360.com/x\nRETRIEVED: 2026-10-03 UTC\n\n" + body.split("\n", 3)[3])
    import os
    os.environ["NAV_ROOT"] = str(sandbox)
    docs = {d["doc_id"]: d for d in load_docs(sandbox / "corpus")}
    assert docs["D001"]["supplied"] is True and docs["D032"]["supplied"] is False
    assert docs["D032"]["jurisdiction_hint"] == "Hoboken, NJ"           # inherits the manifest row

    words = body.split("\n", 3)[3].split()
    mk = lambda rid, doc, off, **k: raw_rule(rid, source_doc_id=doc, quoted_span=" ".join(words[off:off + 30]), **k)  # noqa: E731
    rules = [mk("r-0001", "D001", 100, supplied_source=True),
             mk("r-0002", "D032", 100, supplied_source=False, jurisdiction="Hoboken, NJ", level="city")]
    (sandbox / "build/rules_raw.json").write_text(json.dumps({"rules": rules}))
    assert run(sandbox, "verify.py").returncode == 0
    main = json.loads((sandbox / "out/rules.json").read_text())["rules"]
    held = json.loads((sandbox / "out/rules_supplementary.json").read_text())["rules"]
    assert [r["team_rule_id"] for r in main] == ["r-0001"]
    assert [r["team_rule_id"] for r in held] == ["r-0002"] and held[0]["supplementary_source"] is True

    assert run(sandbox, "verify.py", "--with-supplementary").returncode == 0
    both = json.loads((sandbox / "out/rules.json").read_text())["rules"]
    assert {r["team_rule_id"] for r in both} == {"r-0001", "r-0002"}
    assert next(r for r in both if r["team_rule_id"] == "r-0002")["supplementary_source"] is True
    assert "supplementary_source" not in next(r for r in both if r["team_rule_id"] == "r-0001")
