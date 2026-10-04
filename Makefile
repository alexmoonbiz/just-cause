# One command per stage. Every stage reads/writes plain JSON, so any stage can be rerun alone.
# Prefer the project venv so a stale `python3` on PATH cannot run a stage without its deps.
PY ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)

setup:           ## once: create the venv and install dependencies
	python3 -m venv .venv
	.venv/bin/python -m pip install -r requirements.txt

estimate:        ## token + cost estimate, no API calls
	$(PY) src/extract.py --dry-run

check:           ## one real call: proves the key, the model name and the schema
	$(PY) src/extract.py --check

smoke:           ## 3-document trial: check one record by eye before the full run
	$(PY) src/extract.py --only D001,D045,D067 --out build/rules_smoke.json
	$(PY) src/verify.py --raw build/rules_smoke.json --out out/rules_smoke.json --ext build/rules_ext_smoke.json

extract:         ## Module A: corpus -> build/rules_raw.json (cached, resumable)
	$(PY) src/extract.py

verify:          ## drop unprovable quotes, validate schema, print coverage gaps
	$(PY) src/verify.py

geocode:         ## Module B step 1: real state + incorporated place per address
	$(PY) src/geocode.py

engine:          ## Module B/C: out/lookups.json + out/changes.json
	$(PY) src/engine.py

all: extract verify geocode engine   ## the whole pipeline (cached, so reruns are cheap)

test:
	$(PY) -m pytest tests -q

.PHONY: setup estimate check smoke extract verify geocode engine all test
