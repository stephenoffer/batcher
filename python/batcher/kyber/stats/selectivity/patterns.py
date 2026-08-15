"""What a text pattern says about how many rows it matches.

Text predicates are the filters of unstructured data, and a single blunt prior for all of
them throws away what the pattern plainly states. `LIKE 'AIR%'` and `LIKE '%foo%'` differ
by an order of magnitude; `regexp_matches(s, '^[0-9]+$')` classifies a whole value while
`regexp_matches(s, 'error')` scans for a substring anywhere in it.

Both wildcard vocabularies reduce to the same three shapes, so they are answered here
together rather than by two rules that could drift:

* **exact** — the pattern has no wildcards or metacharacters at all and is anchored at both
  ends, so the predicate is equality and gets the equality estimate (a measured skew
  frequency, else `1/ndv`), often 10-100x more selective than a substring;
* **anchored** — the pattern is pinned to the start or the end of the value, so it matches
  far fewer rows than a floating substring (`prefix_selectivity`);
* **substring** — anything else, which keeps `substring_selectivity` and is what every
  pattern used to get.

Split out of `leaves.py` because it is pattern parsing rather than predicate dispatch, and
because that module was within thirty lines of the size limit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.kyber.stats.distribution import residual_eq_frequency, residual_mass
from batcher.kyber.stats.selectivity.scalars import _mcv_lookup
from batcher.plan.expr_ir import Col

if TYPE_CHECKING:
    from collections.abc import Callable

    from batcher.config import CardinalityConfig
    from batcher.plan.expr_ir import StrFunc

__all__ = [
    "anchored_selectivity",
    "like_selectivity",
    "measured_match_fraction",
    "pattern_has_wildcard",
    "regex_selectivity",
    "wildcard_prior",
]


def anchored_selectivity(cfg: CardinalityConfig) -> float:
    """Selectivity of a match anchored to one or both ends of the value.

    One named rule rather than three reads of `cfg.prefix_selectivity`, because `LIKE 'x%'`,
    `starts_with`, `ends_with`, and an anchored regex are all the *same* shape and must not
    drift apart.

    **A known inconsistency this deliberately does not "fix".** An anchored match is a
    strict subset of the floating one — every value matching `'foo%'` also matches
    `'%foo%'` — so `P(anchored) <= P(substring)` holds by construction. The shipped defaults
    assert the reverse (`prefix_selectivity` 0.10 against `substring_selectivity` 0.05).
    Clamping to restore the containment is a one-line change and was tried; it makes the
    *absolute* error worse on the query that exercises it. TPC-H Q14's
    `p_type LIKE 'PROMO%'` really keeps about 20% of `part`, so 0.10 is a 2x under-estimate
    and the clamped 0.05 is a 4x one. Which of the two priors is mis-tuned (and in which
    direction) is a question for `benchmarks/run.py`, not for a containment argument, and
    retuning a cold-start constant on reasoning alone is how a benchmark regression ships.
    Recorded here so the inconsistency is visible rather than silently inherited.

    Args:
        cfg: The cardinality config carrying the anchored prior.

    Returns:
        The anchored-match selectivity.
    """
    return cfg.prefix_selectivity


# SQL `LIKE` wildcards: `%` matches any run, `_` any single character.
_LIKE_WILDCARDS = ("%", "_")

# Every character that can make a regex mean something other than the literal text.
# Deliberately over-inclusive: a pattern wrongly *believed* to contain a metacharacter falls
# back to the substring prior, which is the estimate it had before this module existed. The
# reverse mistake would claim equality for a pattern that matches a whole class of values.
_REGEX_META = frozenset(r".^$*+?{}[]\|()")


def _equality_selectivity(
    expr: StrFunc,
    literal: str,
    ndv: dict[str, float],
    cfg: CardinalityConfig,
    mcv: dict[str, dict[str, float]],
    non_null: float = 1.0,
) -> float:
    """The equality estimate for `col = literal` — a measured skew frequency, else `1/ndv`.

    Returned as a share of every row. A measured frequency already is one; the `1/ndv` and
    cold-start branches describe the non-null values only, so `non_null` puts them in the
    same space (see `residual_eq_frequency`).
    """
    estimate = non_null * cfg.eq_selectivity
    if isinstance(expr.input, Col):
        name = expr.input.name
        col_mcv = (mcv or {}).get(name)
        freq = _mcv_lookup(col_mcv, literal)
        if freq is not None:
            return freq
        d = ndv.get(name)
        if d and d > 0:
            estimate = residual_eq_frequency(d, col_mcv, cfg.eq_selectivity, non_null=non_null)
    return estimate


def like_selectivity(
    expr: StrFunc,
    ndv: dict[str, float],
    cfg: CardinalityConfig,
    mcv: dict[str, dict[str, float]],
    non_null: float = 1.0,
) -> float:
    """`col LIKE pattern` selectivity, read from where the wildcards fall.

    A single blunt `substring_selectivity` for every `LIKE` conflates three shapes that
    differ by an order of magnitude:

    * **no wildcards** (`col LIKE 'DELIVER IN PERSON'`) is exact *equality* — a common shape
      (TPC-H Q19's `l_shipinstruct`, an enum column matched to a constant);
    * an **anchored prefix** (`'AIR%'`) or **suffix** (`'%ing'`) matches far fewer rows than
      an unanchored substring;
    * a genuine **substring** (`'%foo%'`, or any pattern with an interior `%`/`_`).

    An unparseable or non-literal pattern falls back to `substring_selectivity`, exactly as
    every `LIKE` did before.

    Args:
        expr: The `like`/`ilike` predicate.
        ndv: Distinct counts by column name.
        cfg: The cardinality config carrying the three priors.
        mcv: Measured most-common-value frequencies by column name.

    Returns:
        The estimated fraction of rows the predicate keeps.
    """
    pat = expr.pattern
    if not isinstance(pat, str):
        return non_null * cfg.substring_selectivity
    prior = wildcard_prior(expr, cfg)
    if prior is None:  # no wildcard at all: the predicate is equality
        return _equality_selectivity(expr, pat, ndv, cfg, mcv, non_null)
    return non_null * prior


def wildcard_prior(expr: StrFunc, cfg: CardinalityConfig) -> float | None:
    """The shape prior for a `LIKE` pattern that carries a wildcard, else None.

    The single reading of where the wildcards fall — an anchored run at exactly one end is
    `prefix_selectivity`, anything else is `substring_selectivity`. `like_selectivity` uses it
    to pick a prior and `leaves` uses it to decide whether a pattern is refinable against the
    measured values, so the two cannot disagree about which shape a pattern has.

    None means "not a wildcard pattern": the predicate is plain equality (or the pattern is
    not a plan-time string), which the equality estimator answers instead.

    Args:
        expr: The `like`/`ilike` predicate.
        cfg: The cardinality config carrying the two priors.

    Returns:
        The conditional shape prior, or None when the pattern carries no wildcard.
    """
    pat = expr.pattern
    if not isinstance(pat, str) or not pattern_has_wildcard(pat):
        return None
    if "_" not in pat:
        body = pat.strip("%")
        # An anchored match: the single wildcard run sits at exactly one end.
        if body and "%" not in body and (pat.startswith("%") ^ pat.endswith("%")):
            return anchored_selectivity(cfg)
    return cfg.substring_selectivity


def regex_selectivity(
    expr: StrFunc,
    ndv: dict[str, float],
    cfg: CardinalityConfig,
    mcv: dict[str, dict[str, float]],
    non_null: float = 1.0,
) -> float:
    """`regexp_matches(col, pattern)` selectivity, read from the pattern's anchors.

    Every regex used to get `substring_selectivity`, which is right for `'error'` and wrong
    for the anchored patterns the public text API actually generates. `is_alpha`,
    `is_numeric`, `is_alnum`, `is_space`, `is_url`, and `is_email` all lower to an anchored
    `regexp_matches` (`'^[A-Za-z]+$'` and friends, in
    `plan/expr_ir/namespaces/strings.py`), and none of them scans for a substring — they
    classify the *whole* value. Estimating a whole-value classification as an
    anywhere-in-the-text search over-counts the survivors of the most common text filter in
    the engine, on exactly the unstructured data where cardinality is hardest to recover
    from later.

    The three shapes mirror `like_selectivity` exactly, which is the point of stating them
    once: `'^foo$'` with no metacharacters is equality, anything anchored at one or both
    ends is `prefix_selectivity`, and everything else keeps the substring prior.

    Args:
        expr: The `regexp_matches` predicate.
        ndv: Distinct counts by column name.
        cfg: The cardinality config carrying the three priors.
        mcv: Measured most-common-value frequencies by column name.

    Returns:
        The estimated fraction of rows the predicate keeps.
    """
    pat = expr.pattern
    if not isinstance(pat, str) or not pat:
        return non_null * cfg.substring_selectivity
    starts = pat.startswith("^")
    # `\$` is an escaped literal dollar, not an end anchor — a price search, not a
    # whole-value match. Tested *before* the anchor check, so a pattern whose only apparent
    # anchor is an escaped one falls all the way back to the substring prior rather than
    # being read as anchored.
    ends = pat.endswith("$") and not (len(pat) >= 2 and pat[-2] == "\\")
    if not (starts or ends):
        return non_null * cfg.substring_selectivity
    body = pat[1:] if starts else pat
    body = body[:-1] if ends else body
    if not body:
        return non_null * cfg.substring_selectivity  # a bare anchor matches everything
    literal = not any(c in _REGEX_META for c in body)
    if starts and ends and literal:
        return _equality_selectivity(expr, body, ndv, cfg, mcv, non_null)
    return non_null * anchored_selectivity(cfg)


# --- deciding a text predicate against the values that were measured ---------

# The text predicates whose match can be decided against a stored value with plain string
# operations. `regexp_matches` is deliberately absent: the engine matches with Rust's `regex`
# crate, which has no backtracking and is linear in the subject, while this would have to use
# Python's `re`, which backtracks. A pattern that is perfectly well-behaved in the engine
# (`(a+)+$` is linear there) can take exponential time here — and it would take it *in the
# planner*, turning a cost estimate into a hang. A regex keeps its pattern-shape prior.
_VALUE_MATCHERS: dict[str, Callable[[str, str], bool]] = {
    "contains": lambda value, needle: needle in value,
    "starts_with": lambda value, prefix: value.startswith(prefix),
    "ends_with": lambda value, suffix: value.endswith(suffix),
    "like": lambda value, pattern: _like_matches(value, pattern),
    "ilike": lambda value, pattern: _like_matches(value.lower(), pattern.lower()),
}


def pattern_has_wildcard(pattern: object) -> bool:
    """Whether a `LIKE` pattern contains a wildcard (so it is not plain equality)."""
    return isinstance(pattern, str) and any(w in pattern for w in _LIKE_WILDCARDS)


def _like_matches(value: str, pattern: str) -> bool:
    """SQL `LIKE`: `%` matches any run of characters, `_` exactly one.

    A two-pointer scan with a single restart point rather than a translation to a regex.
    Both are O(n·m) in the worst case, but this one has no recursion and no backtracking
    stack, so a pattern like `%a%a%a%a%b` cannot turn a plan-time estimate into a hang the
    way the regex translation of the same pattern can.
    """
    v = p = 0
    star = -1
    resume = 0
    while v < len(value):
        if p < len(pattern) and pattern[p] in ("_", value[v]):
            v += 1
            p += 1
        elif p < len(pattern) and pattern[p] == "%":
            star = p
            resume = v
            p += 1
        elif star >= 0:
            p = star + 1
            resume += 1
            v = resume
        else:
            return False
    while p < len(pattern) and pattern[p] == "%":
        p += 1
    return p == len(pattern)


def measured_match_fraction(
    expr: StrFunc,
    mcv: dict[str, dict[str, float]] | None,
    prior: float,
    non_null: float = 1.0,
) -> float | None:
    """A text predicate's selectivity decided against the column's *measured* values.

    A prior for `contains`/`starts_with`/`LIKE` is a statement about text in general, and the
    columns these predicates actually filter are the ones a prior describes worst: `status`,
    `country`, `category`, `event_type` hold a handful of values, all of them in the
    most-common-value table. The predicate can simply be *evaluated* on each of them.

    So the estimate splits into a part that is known and a part that is guessed::

        Σ f(v) over the listed values the pattern matches  +  residual_mass · prior

    The first term is exact — those frequencies were measured — and the second applies the
    prior only to the mass the table does not cover. On a column whose table enumerates
    nearly all of the mass the answer is nearly exact; on a high-cardinality free-text column
    the table covers little and this degrades smoothly to the prior it started from.

    That gap was the single worst estimate in the selectivity model. Measured on a five-value
    string column where four values contain `"a"`: `str.contains("a")` was estimated at
    `substring_selectivity` — 5% of the rows — against 80% actual, a 16x under-estimate, and
    an under-estimate is the dangerous direction because it sizes hash tables and picks build
    sides.

    Args:
        expr: The string predicate.
        mcv: Measured most-common-value frequencies by column name.
        prior: The shape prior to apply to the uncovered mass.
        non_null: The share of rows the column holds a value on, which bounds the table's
            residual (see `residual_mass`).

    Returns:
        The refined selectivity, or None when there is no table, no plain-string matcher for
        this function, or no readable pattern — in each case the caller keeps its prior.
    """
    if not isinstance(expr.input, Col) or not isinstance(expr.pattern, str):
        return None
    table = (mcv or {}).get(expr.input.name)
    if not table:
        return None
    matcher = _VALUE_MATCHERS.get(expr.fn)
    if matcher is None:
        return None
    if expr.fn in ("like", "ilike") and "\\" in expr.pattern:
        return None  # an escape sequence this matcher does not implement
    matched = 0.0
    for value, freq in table.items():
        try:
            if matcher(value, expr.pattern):
                matched += max(0.0, float(freq))
        except (TypeError, ValueError, AttributeError):  # pragma: no cover - odd MCV key
            return None
    return min(1.0, matched + residual_mass(table, non_null) * prior)
