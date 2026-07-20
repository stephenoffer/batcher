"""Join elimination — removing a join outright, and the proofs that make it legal.

Dropping a join is the most dangerous rewrite an optimizer can make: a join does *two*
things to its probe side and both must be proven harmless before it can go — it can
**filter** (a probe row with no match disappears) and it can **duplicate** (a probe row
with `n` matches comes out `n` times).

Uniqueness of the build key kills duplication: with a proven-unique key every probe row
matches at most one build row, so nothing fans out. Nothing in this engine kills the
*filtering* for an **inner** join. That needs *referential integrity* — a declared foreign
key guaranteeing every probe key is present on the build side — and Batcher has no FK
constraints (no catalog declares one, and `RelStats` cannot express one). So:

    **A plain INNER join is NOT eliminable here, however unique and unused its build
    side is.** `t JOIN dim ON t.k = dim.k` still drops every `t` row whose `k` is
    missing from `dim` (and every row whose `k` is NULL). Any rule that removed it
    would silently invent rows. This module deliberately does not implement one — see
    `inner_join_to_semi_when_right_unique`, which is the strongest *sound* thing to do
    with that shape: keep the filtering, drop the payload.

What **is** sound, and is what this module implements:

* an **outer** join to a unique key filters nothing (its preserved side survives by
  definition) — so if no column of the null-supplying side is read, it is dead weight
  (`eliminate_left_join`, already in `rules.joins`; `eliminate_left_join_under_distinct`
  here removes the uniqueness precondition when a `DISTINCT` sits above);
* a **self**-join to a proven-unique, proven-non-null key matches every row to itself,
  exactly once (`self_join_elimination`, `self_semi_join_to_filter`,
  `self_anti_join_to_null_keys`);
* a **cartesian** (constant-key) join filters nothing by construction — everything
  matches everything — so a one-row or merely non-empty other side settles it
  (`eliminate_cross_join_of_single_row`, `semi_join_of_nonempty_cartesian`,
  `anti_join_of_nonempty_cartesian_to_empty`);
* **provably disjoint** key ranges settle the join the other way: nothing matches, so an
  inner/semi join is empty and a left/right/anti join is its preserved side
  (`join_disjoint_keys_to_empty`, `no_match_join_to_preserved_side`).

Two evidence rules bind the whole file. **Uniqueness must be proven, never estimated** —
a structural `GROUP BY`/`DISTINCT` on the key, or an `EXACT` distinct count reaching an
`EXACT` row count (`rules.joins._right_unique_on_keys`, reused rather than re-derived); a
sketched/learned NDV is a guess and can never drop a join, and every range, row count and
null count read here must likewise carry `Provenance.EXACT`. And no rewrite may change the
output **schema**: the join's `output` aliases are re-projected, in order, off whichever
input survives.
"""

from __future__ import annotations

# `_relation_key` is re-exported (redundant alias = an explicit re-export): it is the one
# structural relation-identity this engine has, and `setops_extra` reads it from here.
from batcher.kyber.rules.extra.join_elim.evidence import _relation_key as _relation_key
from batcher.kyber.rules.extra.join_elim.rules import (
    anti_join_of_nonempty_cartesian_to_empty,
    eliminate_cross_join_of_single_row,
    eliminate_left_join_under_distinct,
    inner_join_to_semi_when_right_unique,
    join_disjoint_keys_to_empty,
    no_match_join_to_preserved_side,
    self_anti_join_to_null_keys,
    self_join_elimination,
    self_semi_join_to_filter,
    semi_join_of_nonempty_cartesian,
)

__all__ = [
    "anti_join_of_nonempty_cartesian_to_empty",
    "eliminate_cross_join_of_single_row",
    "eliminate_left_join_under_distinct",
    "inner_join_to_semi_when_right_unique",
    "join_disjoint_keys_to_empty",
    "no_match_join_to_preserved_side",
    "self_anti_join_to_null_keys",
    "self_join_elimination",
    "self_semi_join_to_filter",
    "semi_join_of_nonempty_cartesian",
]
