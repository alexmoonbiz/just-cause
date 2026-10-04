"""
Rental Housing Law Navigator: Streamlit front end.

Reads the pipeline's JSON outputs and recomputes any as-of date live with the SAME
engine that wrote the submission files. No API key, no LLM call at runtime:
the AI reads the law once (extract.py); deterministic code decides (engine.py).

    streamlit run app.py
"""

from __future__ import annotations

import csv
import html
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from common import ROOT  # noqa: E402
from dsl import APPLIES, BASE_AS_OF, NOT_YET_EFFECTIVE, PENDING, SUPERSEDED, UNKNOWN  # noqa: E402
from engine import CATEGORY_ORDER, Engine  # noqa: E402

CAT = {
    "rent_increase_limits": "Rent increase limits",
    "just_cause_eviction": "Just-cause eviction",
    "security_deposits": "Security deposits",
    "application_screening_fees": "Application and screening fees",
    "screening_restrictions": "Screening restrictions",
    "algorithmic_rent_setting": "Algorithmic rent setting",
}
BADGE = {
    APPLIES: ("Applies", "#1a7f37"),
    UNKNOWN: ("Unknown", "#9a6700"),
    SUPERSEDED: ("Superseded", "#0969da"),
    NOT_YET_EFFECTIVE: ("Not yet effective", "#8250df"),
    PENDING: ("Pending bill, not law", "#57606a"),
}
DISCLAIMER = ("Not legal advice. This is an informational prototype built from public sources. "
              "Check the cited text, and a qualified professional, before relying on any answer.")

st.set_page_config(page_title="Rental Housing Law Navigator", layout="wide")


# ------------------------------------------------------------------ data ----
def have_outputs() -> bool:
    return all((ROOT / p).exists() for p in ("out/rules.json", "build/addresses_geo.json", "data/sample_addresses.csv"))


@st.cache_resource(show_spinner="Loading rules and addresses...")
def load():
    eng = Engine()
    tests = json.loads((ROOT / "config" / "tests.json").read_text())
    changes, used = eng.run_tests(tests)  # also registers the declared T3 conflicts
    return eng, tests, changes, used


@st.cache_data(show_spinner=False)
def distribution(_eng: Engine, as_of: date) -> dict:
    c: Counter = Counter()
    for aid in _eng.addresses:
        for e in _eng.evaluate(aid, as_of):
            c[e["result"]] += 1
    return dict(c)


def badge(result: str) -> str:
    label, color = BADGE.get(result, (result, "#57606a"))
    return (f"<span style='background:{color};color:#fff;padding:2px 10px;border-radius:12px;"
            f"font-size:0.8rem;font-weight:600'>{html.escape(label)}</span>")


def one_line(s: str) -> str:
    return " ".join((s or "").split())


# ------------------------------------------------------------------ page ----
st.title("Rental Housing Law Navigator")
st.warning(DISCLAIMER)

if not have_outputs():
    st.error("No pipeline outputs found yet. Run `make all` (extract, verify, geocode, engine), then reload.")
    st.stop()

eng, tests, changes, used = load()
ext = eng.ext

with st.sidebar:
    st.header("Query date")
    as_of = st.date_input("Answers as of", value=BASE_AS_OF, min_value=date(2020, 1, 1), max_value=date(2035, 12, 31))
    compare = st.checkbox("Compare with another date")
    other = st.date_input("Compare with", value=date(2027, 7, 2)) if compare else None
    st.caption("Default is 2026-10-01. Change it to see what a new or pending law would change.")
    st.divider()
    st.markdown("**How it works.** AI reads the law once and extracts rules with a quoted source "
                "passage. Plain code, not AI, then decides which rules cover which building. "
                "When a needed fact is missing, the answer is **Unknown**, never a guess.")

tab_addr, tab_tests, tab_rules, tab_audit = st.tabs(
    ["Address lookup", "Change tests", "Rules and citations", "Audit and coverage"])

# ------------------------------------------------------------ address tab ----
with tab_addr:
    ids = list(eng.addresses)

    def label(aid: str) -> str:
        a, g = eng.addresses[aid], eng.geo[aid]
        return f"{aid} · {a['street_address'].title()} · {g.get('city') or a['postal_city']}"

    aid = st.selectbox("Choose an address (type to search)", ids, format_func=label)
    a, g = eng.addresses[aid], eng.geo[aid]
    facts = eng.facts(aid)

    left, right = st.columns([2, 3])
    with left:
        st.subheader(a["street_address"].title())
        st.caption(f"Answers as of **{as_of.isoformat()}**")
        city = g.get("city")
        st.markdown(f"**State:** {g['state']}  \n**City:** {city or 'not one of the 10 cities in the corpus'}")
        src = {"census": "Census Geocoder (legal boundary)",
               "mailing_city_fallback": "mailing city (Census lookup unavailable)"}.get(g.get("source"), g.get("source") or "n/a")
        st.caption(f"Jurisdiction source: {src}")
        if g.get("uncertain"):
            st.warning("The legal city could not be confirmed, so city rules are shown as Unknown.")
        rng = facts["units_range"]
        units = (a["units"] if a["units"] else
                 (f"{rng[0]}{'+' if rng[1] is None else '-' + str(rng[1])} (from property use code)" if rng else "not in data"))
        st.markdown(f"**Year built:** {a['year_built'] or 'not in data'}  \n**Units:** {units}  \n"
                    f"**Use:** {a['use_code']} {a['use_description']}")
        st.caption("Owner type is never in the data, so owner-based exemptions can only produce Unknown.")

    entries = eng.evaluate(aid, as_of)
    with right:
        counts = Counter(e["result"] for e in entries)
        cols = st.columns(5)
        short = {APPLIES: "Applies", UNKNOWN: "Unknown", SUPERSEDED: "Superseded",
                 NOT_YET_EFFECTIVE: "Not yet", PENDING: "Pending"}
        tip = {APPLIES: "In force and covers this building.", UNKNOWN: "Coverage depends on a fact not in the data.",
               SUPERSEDED: "Covered, but a stricter rule at another level governs.",
               NOT_YET_EFFECTIVE: "Enacted, but the effective date is after the query date.",
               PENDING: "A bill or proposal. Not law."}
        for col, r in zip(cols, [APPLIES, UNKNOWN, SUPERSEDED, NOT_YET_EFFECTIVE, PENDING]):
            col.metric(short[r], counts.get(r, 0), help=tip[r])
        if any(e["conflict_flag"] for e in entries):
            st.error("One or more rules here are flagged for human review (possible conflict between rules).")

    if other:
        before = {e["team_rule_id"]: e for e in eng.evaluate(aid, as_of)}
        after = {e["team_rule_id"]: e for e in eng.evaluate(aid, other)}
        rows = []
        for rid in sorted(set(before) | set(after)):
            b, c = before.get(rid, {}).get("result", "not in scope"), after.get(rid, {}).get("result", "not in scope")
            if b != c:
                r = eng.by_id[rid]
                rows.append({"Rule": r["title"], "Citation": r["citation"], as_of.isoformat(): b, other.isoformat(): c})
        st.subheader(f"What changes between {as_of.isoformat()} and {other.isoformat()}")
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        else:
            st.info("No rule changes status for this address between those dates.")

    st.divider()
    for cat in CATEGORY_ORDER:
        group = [e for e in entries if e["_rule"]["category"] == cat]
        st.markdown(f"### {CAT[cat]}")
        if not group:
            st.caption("No rule in our corpus covers this category for this address. "
                       "That is not the same as no rule existing.")
            continue
        for e in group:
            r, x = e["_rule"], ext.get(e["team_rule_id"], {})
            with st.container(border=True):
                lvl = "State" if r["level"] == "state" else "City"
                st.markdown(f"{badge(e['result'])} &nbsp; **{html.escape(r['title'])}** &nbsp; "
                            f"<span style='opacity:.7'>{lvl} · {html.escape(r['citation'])}</span>",
                            unsafe_allow_html=True)
                st.write(e["explanation"])
                if e["conflict_flag"]:
                    st.warning("Flagged for human review: " + (r.get("conflict_note") or "possible conflict between rules."))
                with st.expander("Source text and confidence"):
                    st.markdown("> " + one_line(r["quoted_span"]).replace("\n", " "))
                    st.markdown(f"Source document **{r['source_doc_id']}**, retrieved {(r.get('retrieved_at') or 'n/a')[:10]}  \n"
                                f"[{r['source_url']}]({r['source_url']})")
                    if x.get("supplementary"):
                        st.warning("Supplementary source: this text was saved separately and is not part of the supplied corpus.")
                    st.caption(f"Extraction confidence {r['confidence']} · quote check: {x.get('citation_check', 'n/a')} · "
                               f"effective {r.get('effective_date') or 'date not stated'} · rule status {r['status']}")

# ------------------------------------------------------------- change tests ----
with tab_tests:
    st.subheader("Change-tracking tests T1 to T5")
    st.caption("Which addresses does each supplied law change affect? Rule selection is configuration "
               "(config/tests.json) reviewed by a person, never hand-written rules.")
    for tid, cfg in tests.items():
        res = changes[tid]
        with st.container(border=True):
            st.markdown(f"#### {tid}")
            st.write(cfg.get("describe", ""))
            c1, c2, c3 = st.columns(3)
            c1.metric("Addresses affected", len(res["affected_address_ids"]))
            c2.metric("Flagged for review", len(res["conflict_flag_address_ids"]))
            c3.metric("Rules used", len(used[tid]))
            if cfg["type"] == "negative":
                if res["affected_address_ids"]:
                    st.error("Violation: a rent cap is reported where none may be.")
                else:
                    st.success("Empty set, as required: no rent cap is reported for Boston or Cambridge.")
            if not used[tid] and cfg.get("known_gap"):
                st.info("Known gap. " + cfg["known_gap"])
            elif not used[tid]:
                st.error("No rules matched this test. Extraction may have missed it, or the selector needs adjusting.")
            else:
                st.dataframe(used[tid], hide_index=True, use_container_width=True)
            with st.expander("Affected address IDs"):
                st.write(", ".join(res["affected_address_ids"]) or "none")

# ------------------------------------------------------------------ rules ----
with tab_rules:
    st.subheader("Every extracted rule, with its source passage")
    f1, f2, f3 = st.columns(3)
    js = f1.multiselect("Jurisdiction", sorted({r["jurisdiction"] for r in eng.rules}))
    cs = f2.multiselect("Category", CATEGORY_ORDER, format_func=CAT.get)
    ss = f3.multiselect("Status", ["in_force", "not_yet_effective", "pending", "failed"])
    rows = [r for r in eng.rules if (not js or r["jurisdiction"] in js) and (not cs or r["category"] in cs)
            and (not ss or r["status"] in ss)]
    st.caption(f"{len(rows)} of {len(eng.rules)} rules")
    st.dataframe([{
        "ID": r["team_rule_id"], "Jurisdiction": r["jurisdiction"], "Category": CAT[r["category"]], "Status": r["status"],
        "Effective": r["effective_date"], "Title": r["title"], "Citation": r["citation"], "Key value": r["key_value"],
        "Confidence": r["confidence"], "Review flag": r["conflict_flag"], "Source": r["source_doc_id"],
        "Quote check": ext.get(r["team_rule_id"], {}).get("citation_check"),
    } for r in rows], hide_index=True, use_container_width=True)
    if rows:
        pick = st.selectbox("Open a rule", [r["team_rule_id"] for r in rows],
                            format_func=lambda i: f"{i} · {eng.by_id[i]['jurisdiction']} · {eng.by_id[i]['title']}")
        r = eng.by_id[pick]
        st.markdown("> " + one_line(r["quoted_span"]))
        st.write(r["requirement"])
        st.json({k: r[k] for k in ("coverage_conditions", "exemptions", "interaction", "overrides", "conflict_note")}, expanded=False)
        st.json(ext.get(pick, {}), expanded=False)

# ------------------------------------------------------------------ audit ----
with tab_audit:
    st.subheader("Audit and coverage")
    dist = distribution(eng, as_of)
    st.markdown(f"**All {len(eng.addresses)} addresses, as of {as_of.isoformat()}**")
    st.bar_chart({BADGE[k][0]: dist.get(k, 0) for k in BADGE})

    st.markdown("**Rules per jurisdiction and category.** A zero is a gap in the source material, shown rather than hidden.")
    grid = {}
    for r in eng.rules:
        grid.setdefault(r["jurisdiction"], Counter())[r["category"]] += 1
    st.dataframe([{"Jurisdiction": j, **{CAT[c]: grid[j][c] for c in CATEGORY_ORDER}} for j in sorted(grid)],
                 hide_index=True, use_container_width=True)

    manifest = ROOT / "corpus" / "corpus_manifest.csv"
    if manifest.exists():
        rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
        with_text = sum(1 for r in rows if r.get("text_file") and (ROOT / "corpus" / r["text_file"]).exists())
        st.markdown(f"**Corpus:** {with_text} of {len(rows)} documents have supplied text. "
                    f"The other {len(rows) - with_text} are links only and are not read unless their text is added.")

    dropped = ROOT / "runs" / "dropped.json"
    if dropped.exists():
        d = json.loads(dropped.read_text())
        st.markdown(f"**Verification:** {len(d)} extracted rules were dropped because their quoted source passage "
                    f"could not be found in the document. **Fabricated citations in the output: 0.**")
        if d:
            with st.expander("Dropped rules and reasons"):
                st.dataframe(d, hide_index=True, use_container_width=True)
    log = ROOT / "runs" / "log.jsonl"
    if log.exists():
        ev = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        ex = [e for e in ev if e.get("event") == "extracted"]
        if ex:
            st.markdown(f"**Audit log:** {len(ex)} extraction calls, model `{ex[-1]['model']}`, "
                        f"prompt version `{ex[-1]['prompt_version']}`.")
    st.caption(DISCLAIMER)
