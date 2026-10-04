"""
Three-valued coverage logic. No LLM calls here: pure functions, fully tested.

Core idea: when a rule's coverage depends on a fact we do not have, the answer
is UNKNOWN, never a guess. UNKNOWN is a valid, partially credited result.

Two things make this module match the real data rather than a textbook:

  * Year built is not a certificate-of-occupancy date. A rule that cuts off at
    1979-06-13 cannot be decided for a building "built 1979". We therefore treat
    a year as the interval [Jan 1, Dec 31] and answer TRUE/FALSE only when the
    whole interval sits on one side of the cutoff. Otherwise UNKNOWN.
  * Effective dates are sometimes partial ("2026", "2026-03"). They get the same
    interval treatment, so a partial date never produces a false certainty.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Iterable, Mapping, Optional

BASE_AS_OF = date(2026, 10, 1)  # the pack's default query date


class Tri(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    def __bool__(self) -> bool:  # force explicit comparison, never `if tri:`
        raise TypeError("Tri is three-valued; compare against Tri.TRUE explicitly")


def tri_and(values: Iterable[Tri]) -> Tri:
    """FALSE dominates; else any UNKNOWN -> UNKNOWN. Empty -> TRUE."""
    unknown = False
    for v in values:
        if v is Tri.FALSE:
            return Tri.FALSE
        if v is Tri.UNKNOWN:
            unknown = True
    return Tri.UNKNOWN if unknown else Tri.TRUE


def tri_or(values: Iterable[Tri]) -> Tri:
    """TRUE dominates; else any UNKNOWN -> UNKNOWN. Empty -> FALSE."""
    unknown = False
    for v in values:
        if v is Tri.TRUE:
            return Tri.TRUE
        if v is Tri.UNKNOWN:
            unknown = True
    return Tri.UNKNOWN if unknown else Tri.FALSE


def tri_not(v: Tri) -> Tri:
    return {Tri.TRUE: Tri.FALSE, Tri.FALSE: Tri.TRUE, Tri.UNKNOWN: Tri.UNKNOWN}[v]


# ---------------------------------------------------------------- dates ----

_PARTIAL = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")


def parse_interval(s: Any) -> Optional[tuple[date, date]]:
    """'2026' -> (2026-01-01, 2026-12-31); '2026-03' -> month; full date -> day."""
    if s is None:
        return None
    if isinstance(s, date):
        return (s, s)
    m = _PARTIAL.match(str(s).strip())
    if not m:
        return None
    y, mo, d = int(m.group(1)), m.group(2), m.group(3)
    try:
        if d:
            x = date(y, int(mo), int(d))
            return (x, x)
        if mo:
            last = calendar.monthrange(y, int(mo))[1]
            return (date(y, int(mo), 1), date(y, int(mo), last))
        return (date(y, 1, 1), date(y, 12, 31))
    except ValueError:
        return None


def compare_interval(op: str, interval: tuple[date, date], cutoff: date) -> Tri:
    """Does every day in `interval` satisfy `day <op> cutoff`? TRUE / FALSE / UNKNOWN."""
    lo, hi = interval
    if op == "<=":
        return Tri.TRUE if hi <= cutoff else Tri.FALSE if lo > cutoff else Tri.UNKNOWN
    if op == "<":
        return Tri.TRUE if hi < cutoff else Tri.FALSE if lo >= cutoff else Tri.UNKNOWN
    if op == ">=":
        return Tri.TRUE if lo >= cutoff else Tri.FALSE if hi < cutoff else Tri.UNKNOWN
    if op == ">":
        return Tri.TRUE if lo > cutoff else Tri.FALSE if hi <= cutoff else Tri.UNKNOWN
    if op == "==":
        if lo == hi == cutoff:
            return Tri.TRUE
        return Tri.FALSE if (cutoff < lo or cutoff > hi) else Tri.UNKNOWN
    if op == "!=":
        return tri_not(compare_interval("==", interval, cutoff))
    raise ValueError(f"op {op!r} not valid for dates")


def compare_range(op: str, lo: float, hi: float, x: float) -> Tri:
    """Does every number in [lo, hi] satisfy `n <op> x`? hi may be float('inf')."""
    if op == "<=":
        return Tri.TRUE if hi <= x else Tri.FALSE if lo > x else Tri.UNKNOWN
    if op == "<":
        return Tri.TRUE if hi < x else Tri.FALSE if lo >= x else Tri.UNKNOWN
    if op == ">=":
        return Tri.TRUE if lo >= x else Tri.FALSE if hi < x else Tri.UNKNOWN
    if op == ">":
        return Tri.TRUE if lo > x else Tri.FALSE if hi <= x else Tri.UNKNOWN
    if op == "==":
        if lo == hi == x:
            return Tri.TRUE
        return Tri.FALSE if (x < lo or x > hi) else Tri.UNKNOWN
    if op == "!=":
        return tri_not(compare_range("==", lo, hi, x))
    raise ValueError(op)


# ------------------------------------------------------------ conditions ----

FIELDS = {"year_built", "units", "use_code", "owner_type", "occupancy_date"}
NUMERIC_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _as_list(v: Any) -> list[str]:
    if isinstance(v, (list, tuple, set)):
        return [str(x).strip().lower() for x in v]
    return [x.strip().lower() for x in str(v).split(",") if x.strip()]


@dataclass(frozen=True)
class Condition:
    field: str
    op: str
    value: Any

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Condition":
        if d["field"] not in FIELDS:
            raise ValueError(f"unsupported field {d['field']!r}")
        if d["op"] not in {*NUMERIC_OPS, "in", "not_in"}:
            raise ValueError(f"unsupported op {d['op']!r}")
        return Condition(d["field"], d["op"], d["value"])

    def evaluate(self, facts: Mapping[str, Any]) -> tuple[Tri, str]:
        """Return (result, human-readable reason). Missing data -> UNKNOWN."""
        f, op, want = self.field, self.op, self.value

        if f == "occupancy_date":
            iv = parse_interval(facts.get("occupancy_date"))
            basis = "certificate of occupancy date"
            if iv is None:
                yb = facts.get("year_built")
                if yb in (None, ""):
                    return Tri.UNKNOWN, "no year built or certificate date in the data"
                try:
                    iv = parse_interval(str(int(float(yb))))
                except (TypeError, ValueError):
                    return Tri.UNKNOWN, "year built is not a usable number"
                basis = f"year built {int(float(yb))} (certificate date not in data)"
            cutoff = parse_interval(want)
            if cutoff is None or cutoff[0] != cutoff[1]:
                return Tri.UNKNOWN, f"cutoff {want!r} is not a full date"
            res = compare_interval(op, iv, cutoff[0])
            why = f"{basis} vs cutoff {cutoff[0].isoformat()}"
            return res, (why if res is not Tri.UNKNOWN else why + " is too close to call")

        actual = facts.get(f)
        if f == "units" and actual in (None, "") and facts.get("units_range"):
            lo, hi = facts["units_range"]
            try:
                res = compare_range(op, float(lo), float(hi) if hi is not None else float("inf"), float(want))
            except (TypeError, ValueError, KeyError):
                return Tri.UNKNOWN, "units not comparable"
            span = f"{lo}+" if hi is None else f"{lo}-{hi}"
            return res, f"units {span} (inferred from the property use code; exact count not in data)"
        if actual is None or actual == "":
            return Tri.UNKNOWN, f"{f} not in the data"

        if op in ("in", "not_in"):
            hit = str(actual).strip().lower() in _as_list(want)
            res = Tri.TRUE if hit == (op == "in") else Tri.FALSE
            return res, f"{f}={actual}"

        if f == "year_built" and re.match(r"^\d{4}-\d{2}", str(want).strip()):
            # Extraction sometimes writes the statute's cutoff date on year_built
            # ("built before October 1, 1978"). The data holds only a year, so compare
            # the whole calendar year: 1927 is clearly before, 1978 is too close to call.
            cutoff = parse_interval(want)
            try:
                built = parse_interval(str(int(float(actual))))
            except (TypeError, ValueError):
                built = None
            if cutoff is None or cutoff[0] != cutoff[1] or built is None:
                return Tri.UNKNOWN, f"{f} not comparable"
            res = compare_interval(op, built, cutoff[0])
            why = f"year built {int(float(actual))} vs cutoff {cutoff[0].isoformat()}"
            return res, (why if res is not Tri.UNKNOWN else why + " is too close to call")

        if f in ("year_built", "units"):
            try:
                a, b = float(actual), float(want)
            except (TypeError, ValueError):
                return Tri.UNKNOWN, f"{f} not comparable"
            return (Tri.TRUE if NUMERIC_OPS[op](a, b) else Tri.FALSE), f"{f}={actual}"

        a, b = str(actual).strip().lower(), str(want).strip().lower()
        return (Tri.TRUE if NUMERIC_OPS[op](a, b) else Tri.FALSE), f"{f}={actual}"


def _group(group: Mapping[str, Any] | list) -> list[Mapping[str, Any]]:
    return group["all_of"] if isinstance(group, Mapping) else list(group)


def evaluate_coverage(
    coverage: Iterable[Mapping[str, Any]],
    exemption_groups: Iterable[Any],
    facts: Mapping[str, Any],
    unstructured: str | None = None,
) -> tuple[Tri, list[str]]:
    """Covered iff every coverage condition holds AND no exemption group holds.

    `exemption_groups` is a list of groups; each group is an AND of conditions
    (`{"all_of": [...]}`), and the groups are OR'd. Example, the NJ deposit rule's
    "owner-occupied with 2 or fewer units" is one group of two conditions: a
    building with 5+ units makes the group FALSE even though owner_type is unknown.

    Anything the extractor could not express structurally (`unstructured`) caps a
    would-be TRUE at UNKNOWN: we never claim a rule applies on a test we did not run.
    """
    notes: list[str] = []

    cov = []
    for c in coverage:
        r, why = Condition.from_dict(c).evaluate(facts)
        cov.append(r)
        if r is not Tri.TRUE:
            notes.append(f"coverage {c['field']} {c['op']} {c['value']}: {r.value} ({why})")
    covered = tri_and(cov)
    if covered is Tri.FALSE:
        return Tri.FALSE, notes

    group_results = []
    for g in exemption_groups:
        rs = []
        for c in _group(g):
            r, why = Condition.from_dict(c).evaluate(facts)
            rs.append(r)
            if r is Tri.UNKNOWN:
                notes.append(f"exemption {c['field']} {c['op']} {c['value']}: unknown ({why})")
        group_results.append(tri_and(rs))
    exempt = tri_or(group_results)
    if exempt is Tri.TRUE:
        return Tri.FALSE, notes + ["an exemption applies"]

    result = tri_and([covered, tri_not(exempt)])
    if unstructured and result is Tri.TRUE:
        notes.append(f"unevaluated condition: {unstructured}")
        return Tri.UNKNOWN, notes
    return result, notes


# -------------------------------------------------------------- lifecycle ----

APPLIES = "applies"
UNKNOWN = "unknown"
SUPERSEDED = "superseded"
NOT_YET_EFFECTIVE = "not_yet_effective"
PENDING = "pending"
NOT_APPLICABLE = "not_applicable"  # never written to lookups.json


def effective_state(eff: str | None, as_of: date) -> Tri:
    """TRUE = in force on as_of, FALSE = not yet, UNKNOWN = partial date straddles it."""
    iv = parse_interval(eff)
    if iv is None:
        return Tri.UNKNOWN
    lo, hi = iv
    if as_of >= hi:
        return Tri.TRUE
    if as_of < lo:
        return Tri.FALSE
    return Tri.UNKNOWN


def temporal_status(status: str, effective_date: str | None, as_of: date) -> str | None:
    """Terminal lifecycle result, or None meaning 'live on as_of: evaluate coverage'.

    `status` is the schema enum, defined as of BASE_AS_OF (2026-10-01). For other
    as-of dates we re-derive from effective_date; without a date we can only vouch
    for dates on the same side of BASE_AS_OF as the stated status.
    """
    if status == "pending":
        return PENDING
    if status == "failed":
        return NOT_APPLICABLE  # T5: a struck measure is not law and never reported as a cap
    if status not in ("in_force", "not_yet_effective"):
        return UNKNOWN

    if parse_interval(effective_date) is not None:
        e = effective_state(effective_date, as_of)
        if e is Tri.TRUE:
            return None
        return NOT_YET_EFFECTIVE if e is Tri.FALSE else UNKNOWN

    # No usable date.
    if status == "in_force":
        return None if as_of >= BASE_AS_OF else UNKNOWN
    return NOT_YET_EFFECTIVE if as_of <= BASE_AS_OF else UNKNOWN


def resolve(
    rule: Mapping[str, Any],
    ext: Mapping[str, Any],
    facts: Mapping[str, Any],
    as_of: date,
) -> tuple[str, list[str]]:
    """One rule x one address (already known to be in the rule's jurisdiction)."""
    terminal = temporal_status(rule["status"], rule.get("effective_date"), as_of)
    if terminal is not None:
        return terminal, []
    tri, notes = evaluate_coverage(
        ext.get("coverage_dsl") or [],
        ext.get("exemption_dsl") or [],
        facts,
        ext.get("unstructured_conditions"),
    )
    if tri is Tri.TRUE:
        return APPLIES, notes
    if tri is Tri.UNKNOWN:
        return UNKNOWN, notes
    return NOT_APPLICABLE, notes
