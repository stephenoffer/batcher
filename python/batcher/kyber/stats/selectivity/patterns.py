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

from batcher.kyber.stats.distribution import residual_eq_frequency
from batcher.kyber.stats.selectivity.scalars import _mcv_lookup
from batcher.plan.expr_ir import Col

if TYPE_CHECKING:
    from batcher.config import CardinalityConfig
    from batcher.plan.expr_ir import StrFunc

__all__ = ["anchored_selectivity", "like_selectivity", "regex_selectivity"]


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
) -> float:
    """The equality estimate for `col = literal` — a measured skew frequency, else `1/ndv`."""
    estimate = cfg.eq_selectivity
    if isinstance(expr.input, Col):
        name = expr.input.name
        col_mcv = (mcv or {}).get(name)
        freq = _mcv_lookup(col_mcv, literal)
        if freq is not None:
            return freq
        d = ndv.get(name)
        if d and d > 0:
            estimate = residual_eq_frequency(d, col_mcv, cfg.eq_selectivity)
    return estimate


def like_selectivity(
    expr: StrFunc,
    ndv: dict[str, float],
    cfg: CardinalityConfig,
    mcv: dict[str, dict[str, float]],
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
        return cfg.substring_selectivity
    if not any(w in pat for w in _LIKE_WILDCARDS):
        return _equality_selectivity(expr, pat, ndv, cfg, mcv)
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
        return cfg.substring_selectivity
    starts = pat.startswith("^")
    # `\$` is an escaped literal dollar, not an end anchor — a price search, not a
    # whole-value match. Tested *before* the anchor check, so a pattern whose only apparent
    # anchor is an escaped one falls all the way back to the substring prior rather than
    # being read as anchored.
    ends = pat.endswith("$") and not (len(pat) >= 2 and pat[-2] == "\\")
    if not (starts or ends):
        return cfg.substring_selectivity
    body = pat[1:] if starts else pat
    body = body[:-1] if ends else body
    if not body:
        return cfg.substring_selectivity  # a bare anchor matches everything
    literal = not any(c in _REGEX_META for c in body)
    if starts and ends and literal:
        return _equality_selectivity(expr, body, ndv, cfg, mcv)
    return anchored_selectivity(cfg)
