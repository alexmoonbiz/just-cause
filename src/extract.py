"""
Module A: automated rule extraction (corpus text -> structured rule records).

Run:
    export OPENAI_API_KEY=sk-...                    # ~$1.90 for the full corpus
    python src/extract.py --dry-run                 # token + cost estimate, no API calls
    python src/extract.py --check                   # one real call: key, model, schema
    python src/extract.py --only D001,D045,D067     # smoke test on 3 docs
    python src/extract.py                           # everything (~20 min or less)

Provider follows the model name: anything starting with "gemini" goes to Google's
OpenAI-compatible endpoint with GEMINI_API_KEY, anything else to OpenAI with
OPENAI_API_KEY. One SDK, one code path:
    NAV_MODEL=gemini-2.5-flash python src/extract.py

Properties that matter for the demo and the score:
  * Resumable: every call is cached by sha256(text + prompt + model + schema).
    A crash, a rate limit, or the hour-16 ordinance only ever pays for NEW work.
  * Automated end to end: no rule is hand-written anywhere.
  * Strict structured output: the model cannot return prose or drift from the schema.
  * Audit trail: runs/log.jsonl records model, prompt version, doc, output hash.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import QUERY_DATE, ROOT, load_docs, split_header  # noqa: E402

MODEL = os.environ.get("NAV_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4.1"
PROMPT_VERSION = "v6"
MAX_CHUNK_CHARS = 60_000
CHUNK_OVERLAP_CHARS = 1_500

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
PROVIDER = "gemini" if MODEL.startswith("gemini") else "openai"
# Gemini keys live under either name depending on which Google doc you followed.
KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY") if PROVIDER == "gemini" else ("OPENAI_API_KEY",)

# Gemini counts thinking tokens against the output budget, so it needs more room than
# the ~16k of JSON we actually want back. Free-tier RPM is low, so default to less
# concurrency; both are overridable when the key moves to a paid tier.
if PROVIDER == "gemini":
    MAX_OUTPUT_TOKENS = int(os.environ.get("EXTRACT_MAX_OUTPUT_TOKENS", "32000"))
    MAX_CONCURRENCY = int(os.environ.get("EXTRACT_CONCURRENCY", "2"))
    REASONING_EFFORT = os.environ.get("GEMINI_REASONING_EFFORT", "medium")
    # Free tier is metered per minute, so pace requests instead of discovering the
    # ceiling by being refused. 7s between calls is ~8/min, under a 10 RPM limit.
    MIN_REQUEST_INTERVAL = float(os.environ.get("EXTRACT_MIN_INTERVAL", "7"))
    MAX_ATTEMPTS = int(os.environ.get("EXTRACT_MAX_ATTEMPTS", "12"))
else:
    MAX_OUTPUT_TOKENS = int(os.environ.get("EXTRACT_MAX_OUTPUT_TOKENS", "16000"))
    # A fresh $5 account sits in the lowest paid tier, where tokens-per-minute is the
    # binding limit and concurrency is what blows it. 4 is quick without thrashing.
    MAX_CONCURRENCY = int(os.environ.get("EXTRACT_CONCURRENCY", "4"))
    REASONING_EFFORT = None
    MIN_REQUEST_INTERVAL = float(os.environ.get("EXTRACT_MIN_INTERVAL", "0"))
    MAX_ATTEMPTS = int(os.environ.get("EXTRACT_MAX_ATTEMPTS", "8"))

CACHE_DIR = ROOT / "cache" / "extract"
LOG_PATH = ROOT / "runs" / "log.jsonl"
FAILED_PATH = ROOT / "runs" / "failed.json"
FAILURES: list[str] = []

# Rough USD per 1M tokens (input, output). Estimates only: check the pricing page.
# Gemini entries are the PAID-tier rates; on the free tier the run costs nothing but
# is rate limited, and Google may use the content to improve their products.
PRICES = {
    "gpt-4.1": (2.00, 8.00), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60),
    "gemini-3.5-flash": (1.50, 9.00), "gemini-3.1-flash-lite": (0.30, 2.50),
    "gemini-2.5-flash": (0.30, 2.50), "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
}
FREE_TIER_MODELS = {"gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash",
                    "gemini-2.5-flash-lite", "gemini-2.5-pro"}

# ----------------------------------------------------------------- schema ----

CONDITION = {
    "type": "object", "additionalProperties": False,
    "required": ["field", "op", "value"],
    "properties": {
        "field": {"enum": ["year_built", "units", "use_code", "owner_type", "occupancy_date"]},
        "op": {"enum": ["<", "<=", ">", ">=", "==", "!=", "in", "not_in"]},
        "value": {"type": "string",
                  "description": "Number as digits, date as YYYY-MM-DD, list as comma-separated."},
    },
}

RECORD_PROPS = {
    "jurisdiction": {"type": "string"},
    "level": {"enum": ["state", "city"]},
    "category": {"enum": ["rent_increase_limits", "just_cause_eviction", "security_deposits",
                          "application_screening_fees", "screening_restrictions",
                          "algorithmic_rent_setting"]},
    "status": {"enum": ["in_force", "not_yet_effective", "pending", "failed"]},
    "title": {"type": "string"},
    "requirement": {"type": "string"},
    "key_value": {"type": ["string", "null"]},
    "coverage_conditions": {"type": ["string", "null"]},
    "exemptions": {"type": ["string", "null"]},
    "interaction": {"type": ["string", "null"]},
    "effective_date": {"type": ["string", "null"]},
    "citation": {"type": "string"},
    "quoted_span": {"type": "string"},
    "confidence": {"type": "number"},
    "conflict_flag": {"type": "boolean"},
    "conflict_note": {"type": ["string", "null"]},
    "defers_to_local": {"type": "boolean"},
    "coverage_dsl": {"type": "array", "items": CONDITION},
    "exemption_dsl": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["all_of"],
        "properties": {"all_of": {"type": "array", "items": CONDITION}}}},
    "unstructured_conditions": {"type": ["string", "null"]},
}

EXTRACTION_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["rules"],
    "properties": {"rules": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": list(RECORD_PROPS), "properties": RECORD_PROPS}}},
}

def _type_enums(node):
    """OpenAI strict mode is happiest when every enum also declares its type."""
    if isinstance(node, dict):
        if "enum" in node and "type" not in node:
            node["type"] = "string"
        for v in node.values():
            _type_enums(v)
    elif isinstance(node, list):
        for v in node:
            _type_enums(v)


_type_enums(EXTRACTION_SCHEMA)

# Fields that live in the sidecar (rules_ext.json), not in the submitted rules.json.
EXT_FIELDS = ["defers_to_local", "coverage_dsl", "exemption_dsl", "unstructured_conditions"]

SYSTEM_PROMPT = f"""You turn official rental-housing law into structured rule records.

The query date is {QUERY_DATE}. Everything about "status" is judged as of that date.

SIX CATEGORIES (extract only these):
  rent_increase_limits        caps on how much or how often rent may rise, rent control / stabilization
  just_cause_eviction         limits on terminating a tenancy to listed lawful grounds, relocation pay
  security_deposits           caps, handling, interest, return deadlines for deposits
  application_screening_fees  limits on fees charged to apply or be screened
  screening_restrictions      limits on how tenants may be screened: criminal history, source of income, credit, fair chance
  algorithmic_rent_setting    bans or limits on pricing software / coordinated pricing algorithms

HARD RULES. A violation makes the record worthless:
1. QUOTE VERBATIM. `quoted_span` is copied character for character from the source text
   (20+ characters, one contiguous passage, usually the operative sentence). Do not
   paraphrase, merge two passages, fix typos, or reconstruct from memory. If you cannot
   find an exact supporting passage, do not emit the rule.
2. NEVER INVENT. Silence in the text is not a rule. One record per distinct rule.
3. STATUS, as of {QUERY_DATE}:
     in_force           enacted and effective on or before {QUERY_DATE}
     not_yet_effective  enacted, but the effective date is after {QUERY_DATE}
     pending            a bill or proposal that is not law
     failed             a measure that was struck, defeated, withdrawn or invalidated
   Set `effective_date` to the date the rule takes/took effect, as YYYY, YYYY-MM or
   YYYY-MM-DD, or null if the text gives none. Never guess a date. If the text states it
   relative to enactment or approval (for example "the first day of the twelfth month next
   following the date of enactment"), compute the calendar date from the approval/enactment
   date stated in the document, put the result in `effective_date`, and cite the sentence
   that states the rule (counting: the first month next following is the month after approval).
   The date a bill was "Chaptered", "Approved by the Governor", "Signed" or "Enrolled" is
   NOT its effective date; never put those dates in `effective_date`. If the text states an
   effective or operative date, use it. If the document is a chaptered California statute
   and its text states no effective date and has no urgency clause, California's general
   rule applies (Cal. Const. art. IV, § 8(c)): it takes effect January 1 of the year after
   the year it was chaptered. Apply it, cap `confidence` at 0.8, and end `requirement`
   with "(Effective date inferred from California's general rule; the text states none.)"
4. CITATION: the official cite (e.g. "Cal. Civ. Code § 1947.12", "N.J.S.A. 46:8-21.2",
   "Berkeley Municipal Code 13.63.030"). No URLs here.
5. JURISDICTION: a state code "CA"/"NJ"/"MA" with level "state", or "City, ST"
   (e.g. "Hoboken, NJ") with level "city". Use the document hint but correct it if wrong.
6. `requirement`: one or two plain-language sentences a renter could act on.
   `key_value`: the headline number or formula ("1.5 months' rent", "lesser of CPI+5% or 10%").
7. PENDING BILLS. If the document is a bill that has not become law (a bill page, committee
   report or bill history), emit ONE record with status "pending" and `effective_date` null,
   even when only the bill's title or summary text is available. Quote the title or the
   sentence saying what the bill would do (verbatim, 20+ characters). In `requirement`,
   say what it would prohibit or require if enacted, using only what the quoted text
   supports. Cap `confidence` at 0.6. Choose the category from the subject (for example
   "An Act prohibiting algorithmic rent setting" is algorithmic_rent_setting). Cite it like
   "Mass. S.2983 (194th Gen. Ct.)". Leave `coverage_dsl` empty. Never describe provisions
   the text does not show.

WHO IT COVERS. This matters most: the system evaluates coverage with code, not with you.
The addresses we test are multifamily apartment buildings. Known facts per building are
ONLY: year_built, units, use_code. We never know owner_type or the certificate-of-occupancy date.
 - `coverage_dsl`: AND-ed conditions {{field, op, value}} with fields year_built, units,
   use_code, owner_type, occupancy_date and ops < <= > >= == != in not_in.
   A cutoff on a certificate of occupancy / first occupancy date uses occupancy_date,
   e.g. "buildings first occupied on or before June 13, 1979" ->
   {{"field":"occupancy_date","op":"<=","value":"1979-06-13"}}.
   A minimum size, e.g. "more than 4 units" -> {{"field":"units","op":">","value":"4"}}.
   Statewide or citywide rules that cover all rentals -> empty list.
 - `exemption_dsl`: OR-ed groups; each group is {{"all_of": [conditions]}} that must ALL hold.
   "owner-occupied with 2 or fewer units" -> one group: owner_type == "owner_occupied" AND units <= 2.
   Include only exemptions that could plausibly apply to a multifamily apartment building.
 - Seasonal, hotel, hospital, dormitory, and similar exemptions that cannot describe these
   buildings go in the text field `exemptions` only. Do NOT put them in exemption_dsl.
 - `unstructured_conditions`: use ONLY for a coverage condition that could plausibly decide
   whether a sample building is covered but cannot be expressed with the fields above
   (e.g. a new-construction exemption that depends on a filing, a tenant-specific trigger).
   Plain text. Otherwise null. A non-null value turns every answer for this rule into
   "unknown", which is right when the test truly cannot be run, so do not use it as a hedge.
 - `coverage_conditions` and `exemptions` repeat all of this in plain prose.

INTERACTIONS
 - `defers_to_local`: true ONLY when the text says this (state) rule yields where a stricter
   local rule applies, or does not apply where local rent control governs.
 - `conflict_flag` true, with `conflict_note`, when the text itself says this rule may preempt,
   conflict with, or be preempted by another rule or ordinance (including a clause that bars
   municipalities from enacting conflicting ordinances), or when the document gives two
   different effective dates for the same rule.
 - `interaction`: one sentence on how it relates to other rules, or null.

CONFIDENCE: 0 to 1, your honest estimate that every field is right. Use <0.7 when the text
is ambiguous, secondary (a law-firm summary), or the effective date is inferred.

Skip guidance documents, forms, fee schedules, definitions with no operative rule, and
anything outside the six categories. Return {{"rules": []}} when the text has none."""

USER_TEMPLATE = """Document id: {doc_id}
Jurisdiction hint: {hint}
Source URL: {url}
Retrieved: {retrieved}
Part {part} of {parts}

--- BEGIN SOURCE TEXT ---
{text}
--- END SOURCE TEXT ---"""


# ------------------------------------------------------------- chunking ----

_SECTION = re.compile(
    r"^\s*(?:(?:SEC(?:TION)?|ARTICLE|CHAPTER|Chapter|Article|Section)\b.*|§+\s*[\d.\-]+.*|"
    r"\d+[A-Z]?:\d+[\w\-.:]*\s.*|\d+\.\d+(?:\.\d+)*\s+\S.*)$", re.M)


def chunk_text(text: str) -> list[str]:
    """Split on section headings so no rule is cut mid-sentence; small overlap at seams."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    cuts = [m.start() for m in _SECTION.finditer(text)] or list(range(0, len(text), 20_000))
    chunks, start, last = [], 0, 0
    for c in cuts:
        if c - start >= MAX_CHUNK_CHARS and last > start:
            chunks.append(text[max(0, start - CHUNK_OVERLAP_CHARS):last])
            start = last
        last = c
    chunks.append(text[max(0, start - CHUNK_OVERLAP_CHARS):])
    out = []
    for c in chunks:  # a heading-poor chunk can still be huge: hard-split it
        while len(c) > MAX_CHUNK_CHARS * 1.5:
            out.append(c[:MAX_CHUNK_CHARS])
            c = c[MAX_CHUNK_CHARS - CHUNK_OVERLAP_CHARS:]
        out.append(c)
    return [c for c in out if c.strip()]


# ----------------------------------------------------------- infrastructure ----

def sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def cache_path(job: dict, text: str) -> Path:
    """Cache identity: this text, prompt, model and schema. Change any and the call
    is redone; change none and a crashed or rate-limited run resumes for free."""
    d = job["doc"]
    key = sha(text, PROMPT_VERSION, MODEL, SYSTEM_PROMPT,
              json.dumps(EXTRACTION_SCHEMA, sort_keys=True),
              d["doc_id"], d["jurisdiction_hint"])
    return CACHE_DIR / f"{key[:32]}.json"


def log(event: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": round(time.time(), 1), "provider": PROVIDER, "model": MODEL,
           "prompt_version": PROMPT_VERSION, **event}
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def api_key() -> str | None:
    for name in KEY_ENVS:
        if os.environ.get(name):
            return os.environ[name]
    return None


def make_client():
    """One SDK for both providers: Gemini speaks OpenAI's chat-completions dialect."""
    from openai import AsyncOpenAI
    if PROVIDER == "gemini":
        return AsyncOpenAI(api_key=api_key(), base_url=GEMINI_BASE_URL)
    return AsyncOpenAI(api_key=api_key())


def request_kwargs(user: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        max_completion_tokens=MAX_OUTPUT_TOKENS,
    )
    json_schema: dict[str, Any] = {"name": "rule_extraction", "schema": EXTRACTION_SCHEMA}
    if PROVIDER == "gemini":
        # `strict` is an OpenAI-only key; Gemini constrains decoding from the schema itself.
        # Temperature is left at the model default, which is what Google recommends for
        # Gemini 3, and thinking is capped so the output budget goes to the JSON.
        if REASONING_EFFORT:
            kwargs["reasoning_effort"] = REASONING_EFFORT
    else:
        json_schema["strict"] = True
        if MODEL.startswith(("gpt-4", "gpt-3")):
            kwargs["temperature"] = 0
    kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
    return kwargs


class Pacer:
    """Keeps the whole run under a requests-per-minute ceiling.

    Two jobs. It spaces every request by at least `interval` seconds, and when any
    worker is told 429 it parks ALL of them until the cooldown expires. Without the
    shared cooldown, four workers each backing off independently keep the account
    permanently over its limit and every call eventually fails.
    """

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._lock = asyncio.Lock()
        self._next = 0.0
        self._cooldown_until = 0.0

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                wait = max(self._cooldown_until - now, self._next - now, 0.0)
                if wait <= 0:
                    self._next = now + self.interval
                    return
            await asyncio.sleep(wait)

    def cooldown(self, seconds: float) -> None:
        self._cooldown_until = max(self._cooldown_until, time.monotonic() + seconds)


PACER = Pacer(MIN_REQUEST_INTERVAL)


# Past this, the server is describing a per-day allowance rather than a per-minute one.
DAILY_QUOTA_WAIT = 900.0


def retry_after(e: Exception, default: float) -> float:
    """Honour the server's own backoff hint when it sends one.

    Gemini puts it in prose ("Please retry in 19h5m38s") as well as in retryDelay,
    so both are parsed; the prose form is the one that reveals a daily lockout.
    """
    resp = getattr(e, "response", None)
    hdr = getattr(resp, "headers", None)
    if hdr:
        for name in ("retry-after", "x-ratelimit-reset-requests"):
            raw = hdr.get(name)
            if raw:
                try:
                    return float(str(raw).rstrip("s"))
                except ValueError:
                    pass
    body = str(e)
    m = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", body)
    if m and any(m.groups()):
        h, mi, s = (float(g or 0) for g in m.groups())
        return h * 3600 + mi * 60 + s
    m = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s", body)
    return float(m.group(1)) if m else default


class FatalAPIError(RuntimeError):
    """An error no amount of retrying will fix: bad key, no quota, unknown model."""


def classify(e: Exception) -> Exception:
    """Turn a dead-end API error into FatalAPIError so the run stops instead of
    burning six backoffs per call on a key that can never work."""
    status = getattr(e, "status_code", None)
    body = str(e)
    # Gemini reports a bad key as 400 "Please pass a valid API key", not 401, so the
    # message has to be matched as well as the status.
    if status in (401, 403, 404) or any(
        s in body for s in ("insufficient_quota", "no credits remaining",
                            "valid API key", "API key not valid", "API_KEY_INVALID",
                            "API key expired", "is not found", "PERMISSION_DENIED",
                            "Unsupported parameter", "Invalid JSON payload")
    ):
        return FatalAPIError(body[:400])
    return e


def est_tokens(chars: int) -> int:
    return chars // 4


def build_jobs(docs: list[dict], only: set[str] | None, include_supp: bool = False) -> list[dict]:
    jobs = []
    for d in docs:
        if only and d["doc_id"] not in only:
            continue
        if not d.get("supplied", True) and not include_supp:
            print(f"  skipping {d['doc_id']}: supplementary (not in the supplied corpus). "
                  f"Use --include-supplementary to extract it anyway.")
            continue
        _, _, body = split_header(d["path"].read_text(encoding="utf-8", errors="replace"))
        chunks = chunk_text(body)
        for i, c in enumerate(chunks):
            jobs.append({"doc": d, "text": c, "part": i + 1, "parts": len(chunks)})
    return jobs


async def call_model(client, job: dict, text: str, depth: int = 0) -> list[dict]:
    from openai import APIError, RateLimitError  # imported lazily so --dry-run needs no key
    d = job["doc"]
    cache = cache_path(job, text)
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["rules"]

    user = USER_TEMPLATE.format(doc_id=d["doc_id"], hint=d["jurisdiction_hint"] or "unknown",
                                url=d["url"], retrieved=d["retrieved"],
                                part=job["part"], parts=job["parts"], text=text)
    kwargs = request_kwargs(user)

    for attempt in range(MAX_ATTEMPTS):
        try:
            await PACER.acquire()
            resp = await client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            if choice.finish_reason == "length" and depth < 2 and len(text) > 8_000:
                # Output hit the cap: halve the input at a paragraph break and retry each half.
                mid = text.rfind("\n\n", 0, len(text) // 2) or len(text) // 2
                a = await call_model(client, job, text[:mid], depth + 1)
                b = await call_model(client, job, text[max(0, mid - CHUNK_OVERLAP_CHARS):], depth + 1)
                return a + b
            rules = json.loads(choice.message.content)["rules"]
            break
        except (RateLimitError, APIError, json.JSONDecodeError, KeyError, TypeError) as e:
            fatal = classify(e)
            if isinstance(fatal, FatalAPIError):
                raise fatal
            if isinstance(e, RateLimitError):
                # Park every worker, not just this one, or the herd never gets under
                # the limit. Rate limits are a throughput problem, not a failure, so
                # these attempts are cheap and plentiful.
                wait = retry_after(e, min(15 * 2 ** attempt, 120))
                if wait > DAILY_QUOTA_WAIT:
                    # "Please retry in 19h5m": that is a per-day allowance, not a
                    # per-minute one. No retry schedule survives it.
                    raise FatalAPIError(
                        f"daily quota exhausted, server says retry in {wait / 3600:.1f}h.\n"
                        f"  {str(e)[:300]}")
                PACER.cooldown(wait)
            else:
                wait = min(2 ** attempt, 45)
            print(f"  retry {attempt + 1}/{MAX_ATTEMPTS} {d['doc_id']} in {wait:.0f}s "
                  f"({type(e).__name__})", file=sys.stderr)
            log({"event": "retry", "doc": d["doc_id"], "error": str(e)[:500]})
            await asyncio.sleep(wait)
    else:
        print(f"  FAILED {d['doc_id']} part {job['part']}", file=sys.stderr)
        log({"event": "failed", "doc": d["doc_id"], "part": job["part"]})
        FAILURES.append(f"{d['doc_id']} part {job['part']}/{job['parts']}")
        return []

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"rules": rules}), encoding="utf-8")
    log({"event": "extracted", "doc": d["doc_id"], "part": job["part"], "n_rules": len(rules),
         "in_sha": sha(text)[:16], "out_sha": sha(json.dumps(rules))[:16]})
    return rules


async def run(args) -> None:
    docs = load_docs(args.corpus)
    only = set(args.only.split(",")) if args.only else None
    jobs = build_jobs(docs, only, args.include_supplementary)
    cached = sum(1 for j in jobs if cache_path(j, j["text"]).exists())
    print(f"{len(docs)} documents with text -> {len(jobs)} extraction calls, "
          f"model {MODEL} via {PROVIDER}")
    print(f"{cached} already cached, {len(jobs) - cached} to fetch at concurrency "
          f"{MAX_CONCURRENCY}, min {MIN_REQUEST_INTERVAL:.0f}s between calls")

    client = make_client()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    done = 0

    async def one(job):
        nonlocal done
        async with sem:
            rules = await call_model(client, job, job["text"])
        done += 1
        print(f"  [{done}/{len(jobs)}] {job['doc']['doc_id']} part {job['part']}/{job['parts']}: {len(rules)} rules")
        return rules

    t0 = time.time()
    try:
        results = await asyncio.gather(*(one(j) for j in jobs))
    except FatalAPIError as e:
        log({"event": "fatal", "error": str(e)[:400]})
        sys.exit(f"\nStopping: the API rejected the request in a way retrying cannot fix.\n"
                 f"  {e}\n"
                 f"Finished calls are cached, so fix the key or quota and rerun to resume.")

    merged, seen, n = [], set(), 0
    for job, rules in zip(jobs, results):
        d = job["doc"]
        for r in rules:
            fp = (r["jurisdiction"].lower(), r["category"], re.sub(r"\W+", "", r["citation"].lower()),
                  re.sub(r"\W+", "", r["quoted_span"].lower())[:50])
            if fp in seen:
                continue
            seen.add(fp)
            n += 1
            r = dict(r)
            r["team_rule_id"] = f"r-{n:04d}"
            r["source_doc_id"] = d["doc_id"]
            r["source_url"] = d["url"]
            r["retrieved_at"] = d["retrieved"]
            r["supplied_source"] = bool(d.get("supplied", True))
            merged.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rules": merged}, indent=1), encoding="utf-8")
    print(f"\n{len(merged)} unique rules -> {args.out}  ({time.time() - t0:.0f}s)")
    log({"event": "extract_complete", "n_rules": len(merged), "n_failed": len(FAILURES)})

    FAILED_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAILED_PATH.write_text(json.dumps(FAILURES, indent=1), encoding="utf-8")
    if FAILURES:
        # An incomplete extraction that looks complete is the worst outcome here, so
        # say so plainly rather than leaving it in the scrollback.
        print(f"\n{'!' * 60}")
        print(f"INCOMPLETE: {len(FAILURES)} of {len(jobs)} calls never succeeded, so the "
              f"rules they\nwould have produced are missing from {args.out.name}:")
        for f in FAILURES:
            print(f"  {f}")
        print(f"\nEvery successful call is cached. Rerun the same command to retry only\n"
              f"these, once the quota that refused them has recovered.")
        print(f"{'!' * 60}")
    else:
        print("Next: python src/verify.py")


CHECK_TEXT = ("A landlord shall not demand or receive a security deposit exceeding one "
              "month's rent for an unfurnished residential unit. (Cal. Civ. Code 1950.5)")


async def check() -> None:
    """One real call, with the exact request the pipeline sends: proves the key works,
    the model name exists, and the schema survives this provider's structured output."""
    client = make_client()
    user = USER_TEMPLATE.format(doc_id="D000", hint="CA", url="https://example.test",
                                retrieved="2026-10-01", part=1, parts=1, text=CHECK_TEXT)
    print(f"calling {MODEL} via {PROVIDER} ...")
    try:
        resp = await client.chat.completions.create(**request_kwargs(user))
    except Exception as e:  # noqa: BLE001 - the whole point is to show the raw failure
        fatal = classify(e)
        kind = "not retryable" if isinstance(fatal, FatalAPIError) else "retryable"
        sys.exit(f"FAILED ({kind}): {type(e).__name__}\n  {str(e)[:600]}")
    choice = resp.choices[0]
    print(f"finish_reason: {choice.finish_reason}")
    if getattr(resp, "usage", None):
        print(f"tokens: {resp.usage.prompt_tokens} in, {resp.usage.completion_tokens} out")
    try:
        rules = json.loads(choice.message.content)["rules"]
    except (TypeError, json.JSONDecodeError, KeyError) as e:
        sys.exit(f"FAILED: response was not schema-shaped JSON ({type(e).__name__})\n"
                 f"  {str(choice.message.content)[:600]}")
    print(f"schema OK: {len(rules)} rule(s) parsed")
    if rules:
        r = rules[0]
        print(f"  {r['jurisdiction']} / {r['category']} / {r['status']}")
        print(f"  key_value: {r['key_value']}")
        print(f"  quoted_span: {r['quoted_span'][:120]}")
        print(f"  verbatim in source: {r['quoted_span'] in CHECK_TEXT}")
    print("\nOK. Next: make smoke")


def dry_run(args) -> None:
    docs = load_docs(args.corpus)
    jobs = build_jobs(docs, set(args.only.split(",")) if args.only else None, args.include_supplementary)
    chars = sum(len(j["text"]) for j in jobs)
    tin = est_tokens(chars) + len(jobs) * est_tokens(len(SYSTEM_PROMPT) + 400)
    tout = len(jobs) * 3_000
    pin, pout = PRICES.get(MODEL, (2.00, 8.00))
    print(f"documents with text : {len(docs)}")
    print(f"extraction calls    : {len(jobs)}")
    print(f"provider / model    : {PROVIDER} / {MODEL}")
    print(f"input tokens (est.) : {tin:,}")
    print(f"output tokens (est.): {tout:,}")
    print(f"cost (est.)         : ${tin / 1e6 * pin + tout / 1e6 * pout:.2f} on {MODEL}  (verify prices before relying on this)")
    if MODEL in FREE_TIER_MODELS:
        print(f"                      $0.00 if the key is on Google's free tier, which is rate")
        print(f"                      limited and lets Google train on the content")
    big = sorted(jobs, key=lambda j: -len(j["text"]))[:3]
    for j in big:
        print(f"  largest chunk: {j['doc']['doc_id']} part {j['part']}/{j['parts']} {len(j['text']):,} chars")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=ROOT / "corpus")
    p.add_argument("--out", type=Path, default=ROOT / "build" / "rules_raw.json")
    p.add_argument("--only", help="comma-separated doc ids, e.g. D001,D045")
    p.add_argument("--include-supplementary", action="store_true",
                   help="also extract text files that are not part of the supplied corpus (research only)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--check", action="store_true",
                   help="one real API call on a one-sentence document: key, model, schema")
    args = p.parse_args()
    if args.dry_run:
        return dry_run(args)
    if not api_key():
        sys.exit(f"Model {MODEL} needs {' or '.join(KEY_ENVS)} (or use --dry-run).")
    if args.check:
        return asyncio.run(check())
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
