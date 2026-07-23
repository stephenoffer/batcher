"""The post-run learner must sketch only the columns something will actually read.

`learn_column_stats` builds an HLL + KLL + Misra-Gries sketch over **every value** of every
column it is given. That is an O(rows x columns) pass, and it used to run over *all* of a
source's columns after *every* query — whether or not any of those statistics could ever be
consulted.

On a plain ``read_parquet(dir).collect()`` — 20M rows, 16 columns, no join, no group-by, no
filter, and therefore not one statistic the estimator is able to use — it cost **22.9 seconds
on top of a 0.73-second read**. The query paid thirty times its own cost to learn things
nothing would ever ask for, and it did so on the single most common operation a data engine
performs.

The fix is the rule the rest of the metadata layer already follows (`ndv_columns`,
`column_bounds_needed`): *computing a column the optimizer does not read only wastes work.*
These tests pin both halves of it — that nothing is sketched when nothing can consume it, and
that the columns which **do** steer a plan are still learned, because a learner that learns
nothing is not a fix, it is a different bug.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import kyber
from batcher.api.terminal._metadata import learn_column_stats, learnable_columns
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend
from batcher.plan.expr_ir import col, count


def _table() -> pa.Table:
    return pa.table(
        {
            "k": [1, 2, 3, 4] * 25,
            "v": [10, 20, 30, 40] * 25,
            "payload": ["x" * 32] * 100,
        }
    )


def _learned_columns(hub: MetadataHub) -> set[str]:
    """Every column the hub holds a measured statistic for, across all sources."""
    learned = kyber.load_learned_stats(hub)
    out: set[str] = set()
    for key in (kyber.NDV_KEY, kyber.AVG_BYTES_KEY):
        blob = learned.get(key) or {}
        for qualified in blob:
            # Entries are stored source-qualified (`<source key>\x1f<column>`), so that one
            # table's `id` can never answer for another's.
            out.add(str(qualified).rsplit("\x1f", 1)[-1])
    return out


def _learn_for(ds, hub: MetadataHub) -> set[str]:
    """Run the learner exactly as the conductor does, and report what it measured."""
    from batcher.io.source import read_source

    resolved = [read_source(s, None, None) for s in ds._sources]
    learn_column_stats(hub, resolved, ds._sources, ds._plan)
    return _learned_columns(hub)


# --------------------------------------------------------------------------------------
# What the plan can consume
# --------------------------------------------------------------------------------------


def test_a_bare_scan_consumes_no_column_statistic() -> None:
    """No join, no group-by, no filter ⇒ nothing to learn. This is `read().collect()`."""
    assert learnable_columns(bt.from_arrow(_table())._plan) == set()


def test_a_group_by_consumes_its_keys() -> None:
    plan = bt.from_arrow(_table()).group_by("k").agg(n=count())._plan
    assert "k" in learnable_columns(plan)
    assert "payload" not in learnable_columns(plan)


def test_a_filter_consumes_its_predicate_columns() -> None:
    plan = bt.from_arrow(_table()).filter(col("v") > 15)._plan
    assert "v" in learnable_columns(plan)
    assert "payload" not in learnable_columns(plan)


def test_a_join_consumes_both_sides_keys() -> None:
    left = bt.from_arrow(_table())
    right = bt.from_arrow(pa.table({"k": [1, 2], "w": [7, 8]}))
    plan = left.join(right, on="k")._plan
    assert "k" in learnable_columns(plan)
    assert "payload" not in learnable_columns(plan)


# --------------------------------------------------------------------------------------
# What the learner therefore sketches
# --------------------------------------------------------------------------------------


def test_a_plain_scan_sketches_nothing(tmp_path) -> None:
    """The regression: a plain read must not pay an O(rows x columns) sketch."""
    path = str(tmp_path / "t.parquet")
    bt.from_arrow(_table()).write.parquet(path)

    hub = MetadataHub(InProcessBackend())
    measured = _learn_for(bt.read.parquet(path), hub)

    assert measured == set(), (
        f"a bare scan sketched {sorted(measured)} — columns nothing in the plan can read"
    )


def test_a_group_by_still_learns_its_key(tmp_path) -> None:
    """The other half: the loop must keep working where it pays for itself."""
    path = str(tmp_path / "t.parquet")
    bt.from_arrow(_table()).write.parquet(path)

    hub = MetadataHub(InProcessBackend())
    measured = _learn_for(bt.read.parquet(path).group_by("k").agg(n=count()), hub)

    assert "k" in measured, "the group key was not learned; the estimator stays blind"
    assert "payload" not in measured, "an unread column was sketched anyway"


def test_a_filter_still_learns_its_predicate_column(tmp_path) -> None:
    path = str(tmp_path / "t.parquet")
    bt.from_arrow(_table()).write.parquet(path)

    hub = MetadataHub(InProcessBackend())
    measured = _learn_for(bt.read.parquet(path).filter(col("v") > 15), hub)

    assert "v" in measured
    assert "payload" not in measured


def test_the_cell_cap_bounds_the_sketch(tmp_path) -> None:
    """One enormous column must not turn a cheap query into an expensive one.

    The pre-optimize pass has always honored `ndv_sketch_max_cells`; this one had no ceiling
    at all, which is what let a single scan run away with 22.9 seconds.
    """
    import dataclasses

    from batcher.config import active_config, config_context

    path = str(tmp_path / "t.parquet")
    bt.from_arrow(_table()).write.parquet(path)

    base = active_config()
    capped = dataclasses.replace(
        base,
        optimizer=dataclasses.replace(base.optimizer, ndv_sketch_max_cells=1),
    )

    hub = MetadataHub(InProcessBackend())
    with config_context(capped):  # a cap below any real input
        measured = _learn_for(bt.read.parquet(path).group_by("k").agg(n=count()), hub)

    assert measured == set(), "the sketch ignored the cell cap"


@pytest.mark.parametrize("distributed", [False])
def test_the_result_is_unchanged_by_what_was_learned(tmp_path, distributed) -> None:
    """Learning is an optimization. Whatever it does or skips, the answer is the same."""
    path = str(tmp_path / "t.parquet")
    table = _table()
    bt.from_arrow(table).write.parquet(path)

    got = bt.read.parquet(path).collect(distributed=distributed)
    assert got.num_rows == table.num_rows
    assert got.column_names == table.column_names
    assert got.to_pydict()["k"] == table.to_pydict()["k"]

    grouped = (
        bt.read.parquet(path)
        .group_by("k")
        .agg(n=count())
        .collect(distributed=distributed)
        .to_pydict()
    )
    assert sorted(grouped["n"]) == [25, 25, 25, 25]
