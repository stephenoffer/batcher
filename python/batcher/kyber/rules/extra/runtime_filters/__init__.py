"""Runtime filters and scan-level data skipping — the sideways-information-passing family.

A join already *knows* something about the rows its other side can possibly match. SIP is the
discipline of turning that into a **superset filter** on the opposite input: a predicate that may
only remove rows which provably cannot match, and is therefore free to sink all the way to the
source, where a zone map or a bloom skips whole row groups. `rules.joins.runtime_join_filter`
opens the family (a key's `[min, max]` range); this package carries it the rest of the way.

    evidence  — the proofs, once: null-key, membership (`IN`/`=`), bloom absence, zone bounds.
    sip       — the filters a join implies about its other side.
    skipping  — deciding a conjunct / disjunct / `IN` member from metadata, and the joins those
                proofs settle outright (empty).

`skipping`'s rules must *register* before `sip`'s — within a phase, registration order is run
order, and the emptiness proofs have to read the source's own statistics before an inserted runtime
filter erases them (a `Filter` sets `null_count` to unknown and downgrades provenance away from
EXACT, which is precisely the evidence `empty_join_from_all_null_key` needs). `sip` imports
`skipping` at its top to guarantee that, so the import order *here* is free to stay alphabetical.

Every rule registers in **PUSHDOWN**, not ENFORCE. That is a correctness requirement: a runtime
filter changes the join's subtree, hence its `plan_signature`, and the learning loop keys the
join-strategy arm and the cardinality correction on that signature at SELECTION time while
`annotate_ops` stamps the *final* plan's. Insert a filter after SELECTION and the two keys differ,
so what Core measures is filed under a name Kyber never looks up again — the Core-measures /
Kyber-decides loop silently stops closing. `evidence.SIP` carries the full argument.

Two invariants hold across all of it. `FILTERABLE_SIDES` (owned by `rules.joins`) is the law for
which side each join type may reduce — `inner`/`semi` → both, `anti`/`left` → right only, `right`
→ left only, `full` → **nothing** — and every filter-inserting rule routes through it. And a
bloom proves **absence only**: `contains() -> False` is definitive, `True` is a maybe, and every
probe is domain-guarded, because a cross-domain probe reports a definitive absence for a value
that *is* present and would delete rows.

A module lives here rather than as one `extra/runtime_filters.py` because the family outgrew the
500-line module limit; `.claude/rules/maintainability.md` says package-ize, not shim. The import
path is unchanged. Re-export/registration only — no logic in this file.
"""

from __future__ import annotations

from batcher.kyber.rules.extra.runtime_filters.sip import (
    dedup_source_predicates,
    prune_asof_right_by_on_bound,
    prune_join_side_in_list_by_other_side_bloom,
    prune_join_side_in_list_by_other_side_range,
    push_in_list_across_join_keys,
    push_is_not_null_from_asof_on_key,
    push_is_not_null_from_join_key,
)
from batcher.kyber.rules.extra.runtime_filters.skipping import (
    drop_filter_conjunct_implied_by_zonemap,
    drop_filter_disjunct_refuted_by_zonemap,
    empty_join_from_all_null_key,
    empty_join_from_bloom_absent_key,
    empty_join_from_disjoint_key_values,
    prune_in_list_by_bloom,
    prune_in_list_by_zonemap,
)

__all__ = [
    "dedup_source_predicates",
    "drop_filter_conjunct_implied_by_zonemap",
    "drop_filter_disjunct_refuted_by_zonemap",
    "empty_join_from_all_null_key",
    "empty_join_from_bloom_absent_key",
    "empty_join_from_disjoint_key_values",
    "prune_asof_right_by_on_bound",
    "prune_in_list_by_bloom",
    "prune_in_list_by_zonemap",
    "prune_join_side_in_list_by_other_side_bloom",
    "prune_join_side_in_list_by_other_side_range",
    "push_in_list_across_join_keys",
    "push_is_not_null_from_asof_on_key",
    "push_is_not_null_from_join_key",
]
