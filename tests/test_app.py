"""Smoke-test the Streamlit app with Streamlit's own test runner, on synthetic outputs."""

import importlib
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pipeline import ROOT, build_world, sandbox  # noqa: E402,F401  (sandbox is a fixture)


def fresh_app(monkeypatch, root):
    monkeypatch.setenv("NAV_ROOT", str(root))
    for m in ("common", "engine", "dsl"):
        sys.modules.pop(m, None)
    import streamlit as st
    st.cache_resource.clear()
    st.cache_data.clear()
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)


def test_missing_outputs_shows_instructions_and_disclaimer(monkeypatch, sandbox):
    at = fresh_app(monkeypatch, sandbox).run()
    assert not at.exception
    assert any("Not legal advice" in w.value for w in at.warning)
    assert any("make all" in e.value for e in at.error)


def test_app_renders_conflict_and_compare(monkeypatch, sandbox):
    build_world(sandbox)
    at = fresh_app(monkeypatch, sandbox).run()
    assert not at.exception, at.exception
    assert any("Not legal advice" in w.value for w in at.warning)

    # a Hoboken address carries the T3 conflict flag on 2026-10-01
    import csv
    rows = list(csv.DictReader((sandbox / "data/sample_addresses.csv").open()))
    hob = next(r["address_id"] for r in rows if r["postal_city"] == "Hoboken")
    sel = at.selectbox[0]
    sel.set_value(hob).run()
    assert not at.exception, at.exception
    assert any("human review" in e.value for e in at.error)

    # comparing against 2027-07-02 shows the FAIR Act flipping to "applies"
    at.sidebar.checkbox[0].check().run()
    assert not at.exception, at.exception
    frames = [d.value for d in at.dataframe]
    assert any("NJ FAIR Act" in str(f.to_dict()) for f in frames)
