"""A learned top-N bound must not outlive the column type it was measured on.

`kyber.learned_tuning.topn_bound` remembers the k-th best value a top-N returned and seeds
the next run of the same shape with it as a predicate. Its whole safety argument is that an
*inaccurate* bound cannot return a wrong answer -- the seeded plan removes only rows strictly
beyond the bound, so any `k` survivors are the true top-k -- and that the one failure mode,
too few survivors, is visible in the row count and answered by re-running as written. So "the
cost of a stale bound is one wasted cheap scan".

The bound is keyed by `plan_signature`, whose scan token carries `Scan.source_key` and
therefore identifies the *relation*. It does not carry the column **types**: the schema is
deliberately not in the IR, because the engine reads types off the Arrow batches it is
handed. A source keeps its key when its schema changes, so rewriting a path with a new type
for the same column read the previous run's bound back and seeded a comparison the engine
cannot evaluate:

    RuntimeError: Invalid argument error: Invalid comparison operation: Utf8 >= Int64

That is a raised query rather than a wasted scan, and the `api` verification cannot see it:
it counts the survivors of a plan that never ran. Overwriting a path with a new schema is
ordinary in a medallion pipeline, so this needs nothing exotic to reach -- which is why it is
worth an end-to-end test and not only the unit ones.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt

pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.integration

_N = 100


def _write(path, values, arrow_type):
    pq.write_table(pa.table({"x": values}, schema=pa.schema([("x", arrow_type)])), path)


def _top5(path):
    return bt.read.parquet(str(path)).sort("x", descending=True).limit(5).to_pydict()["x"]


def test_a_path_rewritten_with_a_new_column_type_still_answers(tmp_path):
    """int -> string over the same path. This raised before the bound was type-checked."""
    path = tmp_path / "t.parquet"

    _write(path, list(range(_N)), pa.int64())
    assert _top5(path) == [99, 98, 97, 96, 95]

    _write(path, [f"v{i:03d}" for i in range(_N)], pa.string())
    assert _top5(path) == ["v099", "v098", "v097", "v096", "v095"]


def test_the_reverse_type_change_answers_too(tmp_path):
    """string -> int, which fails the same way in the other direction."""
    path = tmp_path / "t.parquet"

    _write(path, [f"v{i:03d}" for i in range(_N)], pa.string())
    assert _top5(path) == ["v099", "v098", "v097", "v096", "v095"]

    _write(path, list(range(_N)), pa.int64())
    assert _top5(path) == [99, 98, 97, 96, 95]


def test_a_widening_rewrite_keeps_answering(tmp_path):
    """int -> float is a comparison the engine performs, so the bound stays usable.

    The guard consults the engine's own type lattice rather than requiring type equality,
    and this is the case that would regress if it required equality instead.
    """
    path = tmp_path / "t.parquet"

    _write(path, list(range(_N)), pa.int64())
    assert _top5(path) == [99, 98, 97, 96, 95]

    _write(path, [float(i) / 2 for i in range(_N)], pa.float64())
    assert _top5(path) == [49.5, 49.0, 48.5, 48.0, 47.5]


def test_repeated_runs_of_one_shape_still_agree(tmp_path):
    """The seeding must keep working across runs -- the guard must not disable it.

    Three runs of the same query over the same unchanged table: the first learns the bound,
    the later two are seeded by it, and all three must return the same rows.
    """
    path = tmp_path / "t.parquet"
    _write(path, list(range(_N)), pa.int64())
    assert _top5(path) == _top5(path) == _top5(path) == [99, 98, 97, 96, 95]
