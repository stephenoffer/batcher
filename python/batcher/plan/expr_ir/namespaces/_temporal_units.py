"""The truncation-unit vocabulary shared by `.dt.truncate`/`floor`/`ceil`/`round`.

Split out of `temporal.py` only for size (`just lint-structure`); it is one concept and
belongs to the `.dt` namespace. Keeping it here rather than inline also makes the one
place that decides what a unit *means* easy to find, which matters because two spellings
of the same vocabulary met in this namespace and neither understood the other.
"""

from __future__ import annotations

import re

from batcher._internal.errors import PlanError

__all__ = ["TRUNC_UNITS", "trunc_unit"]


#: The truncation units `bc_expr`'s `date_trunc` accepts, in coarse-to-fine order. This
#: list mirrors the Rust match in `bc-expr/src/eval/temporal/date.rs` exactly; the two must
#: change together, because this side is what decides whether the plan is built at all.
TRUNC_UNITS: tuple[str, ...] = (
    "millennium",
    "century",
    "decade",
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "hour",
    "minute",
    "second",
    "millisecond",
    "microsecond",
)

#: Spellings that mean one of `TRUNC_UNITS`. Two vocabularies met here and neither
#: understood the other: `truncate` took only the long names, while `offset_by`/`ceil`/
#: `round` take the duration spellings (`1mo`, `1d`), so `truncate("1mo")` failed on a
#: string the neighbouring method accepts and `offset_by("1month")` failed on the string
#: `truncate` requires. The duration spellings are also what Polars' `dt.truncate` takes,
#: so accepting both is the compatible reading rather than a new dialect.
#:
#: `mo` is months and `m` is minutes, matching `_OFFSET_UNITS` above -- the one ambiguity
#: in the vocabulary, resolved the same way in both places.
_ALIASES: dict[str, str] = {
    "millennia": "millennium",
    "millenium": "millennium",  # the misspelling DuckDB itself accepts
    "centuries": "century",
    "decades": "decade",
    "y": "year",
    "yr": "year",
    "yrs": "year",
    "years": "year",
    "q": "quarter",
    "qtr": "quarter",
    "quarters": "quarter",
    "mo": "month",
    "mon": "month",
    "mons": "month",
    "months": "month",
    "w": "week",
    "wk": "week",
    "wks": "week",
    "weeks": "week",
    "d": "day",
    "dy": "day",
    "days": "day",
    "h": "hour",
    "hr": "hour",
    "hrs": "hour",
    "hours": "hour",
    "m": "minute",
    "min": "minute",
    "mins": "minute",
    "minutes": "minute",
    "s": "second",
    "sec": "second",
    "secs": "second",
    "seconds": "second",
    "ms": "millisecond",
    "milli": "millisecond",
    "millis": "millisecond",
    "milliseconds": "millisecond",
    "us": "microsecond",
    "micro": "microsecond",
    "micros": "microsecond",
    "microseconds": "microsecond",
}

_COUNT_RE = re.compile(r"^(\d+)\s*([a-z]+)$")


def trunc_unit(unit: str, func: str) -> str:
    """Normalize a truncation unit to the one spelling `date_trunc` knows, or raise.

    Validation happens *here*, at the API edge, rather than in the engine. It used not to
    happen at all on this path: the string was handed straight to `bc-expr`, so a typo
    built a perfectly good plan and only failed once the query ran -- after the scan --
    with a bare `RuntimeError` from Rust rather than a typed `PlanError`. Every
    neighbouring method (`ceil`, `round`, `offset_by`) already rejected a bad unit at
    build time; this one was the outlier.

    Args:
        unit: The caller's spelling, e.g. ``"month"``, ``"1mo"``, or ``"mo"``.
        func: The method name to name in the error, e.g. ``"truncate"``.

    Returns:
        The canonical unit name from `TRUNC_UNITS`.

    Raises:
        PlanError: If `unit` is not a string, names no known unit, or carries a
            multiplier other than 1.
    """
    if not isinstance(unit, str):
        raise PlanError(f".dt.{func}() unit must be a string, got {type(unit).__name__}")
    key = unit.strip().lower()
    count_match = _COUNT_RE.match(key)
    if count_match is not None:
        count, key = int(count_match.group(1)), count_match.group(2)
        if count != 1:
            # "5d" is a *bucket width*, which flooring to a calendar boundary cannot
            # express. Silently dropping the multiplier would floor to 1 day and return
            # plausible, wrong timestamps -- so it is refused rather than approximated.
            raise PlanError(
                f".dt.{func}({unit!r}) is not supported: a truncation unit floors to a "
                f"calendar boundary, so it takes no multiplier other than 1. Use "
                f"{'1' + count_match.group(2)!r} for the boundary itself."
            )
    canonical = _ALIASES.get(key, key)
    if canonical not in TRUNC_UNITS:
        raise PlanError(
            f".dt.{func}({unit!r}) is not a known unit; use one of "
            f"{list(TRUNC_UNITS)} (the duration spellings '1mo'/'mo', '1d'/'d', ... are "
            f"accepted too, where 'mo' is months and 'm' is minutes)"
        )
    return canonical
