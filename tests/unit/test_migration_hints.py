"""A failed lookup names the Batcher spelling the registry knows, for every row it knows.

The traceback is what a migrating user reads first, so `absent_error` consults the migration
registry when the curated tables beside each hook say nothing. These tests hold that every rule
and every row renders (a malformed row would otherwise surface as a crash inside an
`AttributeError` handler, the worst place to find it), that the rendered text is what a real
lookup raises, and that a plain typo still gets its did-you-mean rather than registry noise.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.migration import Status, load_registry, load_renames
from batcher._internal.migration.hints import SURFACE_RECEIVERS, migration_hint


def test_every_rename_rule_renders_a_hint() -> None:
    rules = [r for table in load_renames().values() for r in table.values()]
    assert len(rules) > 100
    for rule in rules:
        hint = migration_hint(rule.receiver, rule.removed)
        assert hint and "removed" in hint and "batcher.migrate" in hint, rule


def test_every_registry_row_on_a_mapped_receiver_renders() -> None:
    rows = [r for r in load_registry().rows.values() if (r.engine, r.surface) in SURFACE_RECEIVERS]
    assert len(rows) > 1500
    for row in rows:
        receiver = SURFACE_RECEIVERS[(row.engine, row.surface)]
        assert row.name in (migration_hint(receiver, row.name) or ""), row


@pytest.mark.parametrize(
    ("lookup", "expected"),
    [
        (lambda: bt.from_pydict({"x": [1]}).zipWithIndex, "with_row_index"),
        (lambda: bt.col("x").eqNullSafe, "`Expr.eq_missing`"),
        (lambda: bt.hll_sketch_agg, "no Batcher equivalent yet"),
    ],
)
def test_a_real_lookup_raises_the_registry_text(lookup, expected: str) -> None:
    with pytest.raises(AttributeError, match=expected):
        lookup()


def test_a_typo_still_gets_did_you_mean() -> None:
    with pytest.raises(AttributeError, match="Did you mean 'zfill'"):
        bt.col("x").str.zfill_nope  # noqa: B018


def test_a_mismatch_hint_carries_the_difference() -> None:
    row = next(
        r
        for r in load_registry().rows.values()
        if r.status is Status.MISMATCH and (r.engine, r.surface) in SURFACE_RECEIVERS
    )
    hint = migration_hint(SURFACE_RECEIVERS[(row.engine, row.surface)], row.name)
    assert row.note is not None and row.note in hint
