"""A row callback can say which columns it reads, so the scan stops reading the rest.

`map_batches` has taken `input_columns` since it existed; `map` and `flat_map` did not, and
they are the two operators the declaration is worth the most to. A batch callback pays for an
undeclared column once, when it is decoded. A row callback pays twice: once to decode it, and
again to build it into a Python object for every row. A `map` reading two columns of a
41-column table was boxing all 41, per row, with no way to say otherwise.

These tests pin that the declaration reaches the plan and that the optimizer acts on it. The
result-equality half — that declaring columns does not change the answer — is in
`tests/differential/test_diff_row_callback_pruning.py`.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.rules.projections import required_columns_per_source

pytestmark = pytest.mark.unit


def _wide():
    return bt.from_pydict(
        {
            "id": [1, 2, 3],
            "score": [0.1, 0.2, 0.3],
            "img": [b"aaaaaaaa", b"bbbbbbbb", b"cccccccc"],
        }
    )


def _scanned(ds) -> set[str]:
    """The columns the executor will read for `ds` — the same analysis, on the optimized plan.

    `PhysicalPlan.source_projections` is exactly ``required_columns_per_source`` of the
    optimized logical plan, but building the physical plan lowers the tree to IR and a
    `map_batches` deliberately does not lower. So the projection is read from the logical
    rewrite, which is where it comes from either way.
    """
    optimized = Optimizer().logical_rewrite(ds._plan)
    return set(required_columns_per_source(optimized).get(0, []))


def test_map_carries_the_declaration_to_the_plan():
    plan = _wide().ml.map(lambda row: {"id": row["id"]}, input_columns=["id"])._plan
    assert plan.input_columns == ("id",)


def test_flat_map_carries_the_declaration_to_the_plan():
    plan = _wide().ml.flat_map(lambda row: [row], input_columns=["id"])._plan
    assert plan.input_columns == ("id",)


def test_the_dataset_level_sugar_carries_it_too():
    """`ds.map` is the Ray Data spelling most people reach for, so it must not lose the knob."""
    assert _wide().map(lambda row: row, input_columns=["id"])._plan.input_columns == ("id",)
    assert _wide().flat_map(lambda row: [row], input_columns=["id"])._plan.input_columns == ("id",)


def test_a_declared_row_map_stops_the_scan_reading_the_wide_column():
    ds = _wide().ml.map(lambda row: {"id": row["id"]}, input_columns=["id"], output_columns=["id"])
    assert _scanned(ds.select("id")) == {"id"}


def test_a_declared_flat_map_prunes_too():
    ds = _wide().ml.flat_map(lambda row: [row], input_columns=["id"], output_columns=["id"])
    assert _scanned(ds.select("id")) == {"id"}


def test_a_declared_row_filter_prunes_too():
    ds = _wide().ml.filter(lambda row: row["id"] > 1, input_columns=["id"])
    assert _scanned(ds.select("id")) == {"id"}


def test_an_undeclared_row_map_must_keep_every_column_alive():
    """The safe default: with nothing declared the callback may read anything."""
    ds = _wide().ml.map(lambda row: {"id": row["id"]}, output_columns=["id"])
    assert _scanned(ds.select("id")) == {"id", "score", "img"}


def test_declaring_the_columns_does_not_change_the_rows():
    declared = _wide().ml.map(
        lambda row: {"id": row["id"] * 2}, input_columns=["id"], output_columns=["id"]
    )
    undeclared = _wide().ml.map(lambda row: {"id": row["id"] * 2}, output_columns=["id"])
    assert declared.to_pydict() == undeclared.to_pydict() == {"id": [2, 4, 6]}


# --- dirty data on the row surface ------------------------------------------------------


def _parse(row):
    return {"n": int(row["s"])}


def test_a_row_map_can_tolerate_a_malformed_record():
    """`max_errored_rows` was a `map_batches` option only, and `map`/`flat_map` lower to
    `map_batches` — so the knob existed and no row callback could reach it."""
    ds = bt.from_pydict({"s": ["1", "2", "oops", "4"]})
    assert ds.ml.map(_parse, output_columns=["n"], max_errored_rows=10).to_pydict() == {
        "n": [1, 2, 4]
    }


def test_a_row_flat_map_can_too():
    ds = bt.from_pydict({"s": ["1", "oops"]})
    out = ds.ml.flat_map(
        lambda row: [{"n": int(row["s"])}], output_columns=["n"], max_errored_rows=10
    )
    assert out.to_pydict() == {"n": [1]}


def test_the_dataset_level_sugar_carries_the_budget():
    ds = bt.from_pydict({"s": ["1", "oops"]})
    assert ds.map(_parse, output_columns=["n"], max_errored_rows=10).to_pydict() == {"n": [1]}
