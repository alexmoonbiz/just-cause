# Rental Housing Law Navigator: run guide

Live demo: https://just-cause-housing.replit.app/

    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...             # use a dedicated hackathon project
    make estimate                           # ~$1.90 on gpt-4.1 (estimate)
    make check                              # one real call: key, model name, schema
    make smoke                              # 3 docs. Open out/rules_smoke.json and check one record by eye
    make all                                # extract -> verify -> geocode -> engine (cached, resumable)

Budget: the full corpus is 56 calls. `make check` costs a fraction of a cent and `make smoke`
about 15c, so run both before committing to the ~$1.90 full run. $5 of credit covers the
whole thing with room for one complete redo after a prompt change.

## Model and provider
The extractor picks its provider from the model name and talks to both through the OpenAI
SDK, because Gemini exposes an OpenAI-compatible endpoint. Default is `gpt-4.1`.

| Model | Key | Notes |
|---|---|---|
| `gpt-4.1` (default) | `OPENAI_API_KEY` | ~$1.90 for the corpus |
| `gpt-4.1-mini` | same | ~40c, fine while iterating on the prompt |
| `gemini-2.5-flash` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | free tier, see the warning below |

Override with `NAV_MODEL`, e.g. `NAV_MODEL=gemini-2.5-flash make extract`. The model name
is part of the cache key, so switching models re-runs every call rather than mixing two
models' output in one submission.

### If you use the Gemini free tier
It works and the output quality is good — we extracted and verified 15 rules from a 3-doc
smoke test with zero fabricated citations — but the free tier is metered **per day, per
model**, and the newest models get almost nothing. `gemini-3.5-flash` allows **20 requests
per day**, which cannot finish a 56-call corpus. Check your real limits at
https://ai.dev/rate-limit before relying on a model. Prefer the older Flash models, whose
daily allowances are far larger. Also note Google may use free-tier content to improve
their products; our corpus is public law, so that is fine here.

## Rate limits
Handled by pacing rather than by retrying into a wall:
- requests are spaced `EXTRACT_MIN_INTERVAL` seconds apart (0 on OpenAI, 7 on Gemini) at
  `EXTRACT_CONCURRENCY` (4), since tokens-per-minute is usually the binding limit and
  concurrency is what blows it;
- a 429 parks **every** worker for as long as the server asks, because N workers each
  backing off alone will never get the account under its limit;
- a server hint longer than 15 minutes ("Please retry in 19h5m") means a per-day
  allowance, so the run stops immediately instead of pretending a retry will help.

Any call that never succeeds is listed loudly at the end of extraction and written to
`runs/failed.json`, because an incomplete `rules.json` that looks complete is the worst
outcome available. Rerun the same command to retry only those.

If the key is bad or out of quota, extraction now stops immediately with one clear message
instead of retrying every call six times. Completed calls are cached, so fix and rerun.

## Pipeline
| Stage | File | Needs | Output |
|---|---|---|---|
| A. extract | src/extract.py | OpenAI | build/rules_raw.json |
| A. verify | src/verify.py | nothing | out/rules.json, build/rules_ext.json, runs/dropped.json |
| B. geocode | src/geocode.py | internet (Census, no key) | build/addresses_geo.json |
| B/C. engine | src/engine.py | nothing | out/lookups.json, out/changes.json, out/test_selection.json |

## What counts toward the citation metric
Only the 54 documents the organizers supplied with text. The 33 link-only documents are not part of that corpus.
You may save a link-only page yourself for research if its site terms allow it (`src/add_doc.py`), but rules
from it do not count toward citations. By default the pipeline keeps them out of the submission:

    python src/extract.py --include-supplementary     # extract them as well
    python src/verify.py                              # they land in out/rules_supplementary.json (held out)
    python src/verify.py --with-supplementary         # opt in: they join rules.json, flagged supplementary_source

Decide with the organizers' answer in hand (see the question in the chat). The app labels any such rule.

## Things to check after the first full run
1. `make verify` prints a jurisdiction x category grid. Empty cells are gaps to explain.
2. `out/test_selection.json` lists the rules each of T1-T5 used. Confirm they are the right ones;
   adjust selectors in config/tests.json if not (selection is config, rules are never hand-written).
3. `runs/dropped.json` lists every rule dropped and why.

## Known corpus gaps
33 of 87 documents have no text, so they cannot be cited. The two that matter most:
- T3: covered. The NJ FAIR Act is supplied as D069 (P.L. 2026, c.43), including the municipal-preemption clause.
- T2: the Hoboken and Jersey City algorithmic-pricing ordinances (D032-D035) are link-only. Without them the
  extractor cannot produce those two rules from supplied text. The coverage grid printed by `make verify` shows the gap.

## The app and the live link
    streamlit run app.py                    # after `make all`; reads out/ and build/, recomputes any date live

Deploy (Streamlit Community Cloud, free, needs a public GitHub repo):
1. Run `make all` locally first, so these exist: `out/*.json`, `build/addresses_geo.json`, `build/rules_ext.json`,
   `runs/dropped.json`, `runs/log.jsonl`.
2. Commit them with the code (check that `.gitignore` does not exclude them) and push. Never commit your API key.
3. share.streamlit.io -> New app -> pick the repo -> main file `app.py` -> Deploy.
4. Open the live link on your phone before you record the demo. The app makes no API calls, so it needs no secrets.
5. Safety net: also record a screen capture of the app running locally.

Before deploying, open the app and check three things yourself: an address where a rule is Unknown (and why),
a Hoboken or Jersey City address carrying the review flag, and the Change tests tab showing T5 as an empty set.
