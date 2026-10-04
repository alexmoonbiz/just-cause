"""Guard the committed outputs the demo serves: if out/ goes stale or a rerun regresses, this fails."""

import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "out"
pytestmark = pytest.mark.skipif(not (OUT / "changes.json").exists(), reason="run `make all` first")


def load(name):
    return json.loads((OUT / name).read_text())


@pytest.fixture(scope="module")
def engine():
    from engine import Engine
    return Engine(ROOT)


def test_every_rule_is_tied_to_a_quoted_source():
    rules = load("rules.json")["rules"]
    assert rules
    for r in rules:
        assert r["quoted_span"].strip(), r["team_rule_id"]
        assert r["source_doc_id"] and r["source_url"] and r["citation"], r["team_rule_id"]


def test_change_test_counts_match_the_final_run():
    ch = load("changes.json")
    counts = {t: len(ch[t]["affected_address_ids"]) for t in ch}
    assert counts == {"T1": 250, "T2": 0, "T3": 140, "T4": 110, "T5": 0}
    assert "VIOLATION" not in ch["T5"]["notes"]


def test_T2_zero_is_a_declared_corpus_gap_not_a_silent_miss(engine):
    tests = json.loads((ROOT / "config" / "tests.json").read_text())
    assert tests["T2"].get("known_gap")
    assert engine.select(tests["T2"]["selector"]) == []
    assert "Corpus gap" in load("changes.json")["T2"]["notes"]


def test_as_of_date_flips_the_california_algorithmic_pricing_ban(engine):
    aid = load("changes.json")["T1"]["affected_address_ids"][0]

    def result(day):
        return {e["team_rule_id"]: e["result"] for e in engine.evaluate(aid, day)}.get("r-0028")

    assert result(date(2025, 12, 31)) == "not_yet_effective"
    assert result(date(2026, 1, 2)) == "applies"


def test_lookups_have_no_uncomparable_year_built_conditions():
    lookups = load("lookups.json")["lookups"]
    assert len(lookups) == 500
    assert "not comparable" not in json.dumps(lookups)
    la_rso = {e["team_rule_id"]: e["result"] for e in lookups["A0001"]}
    assert la_rso["r-0045"] == "applies"
