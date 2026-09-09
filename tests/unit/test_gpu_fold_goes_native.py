"""A fan-out's driver-side fold reaches the engine's executor at every size, not just huge ones.

`merge_shards` used to hand the fold to the engine only above 1,048,576 combined partial rows,
on the reasoning that a plan build and an FFI crossing are not worth paying for a small fold.
Measured, the reverse holds: the *pandas* path carries the larger fixed cost (constructing a
`DfBackend`, converting Arrow to pandas and back, ~3.3 ms flat) against the engine's ~0.25 ms,
so the engine was faster at every size measured, down to sixty rows.

The row count also read the wrong quantity. A fold's cost tracks **bytes**, and a string key
carries twenty times the bytes of a numeric one at the same row count — so ClickBench q12's
783,737-row fold sat *under* the threshold while being 178 MiB, and spent 2.6 s of a 3.0 s query
in pandas. Removing the gate took that query to 0.436 s.

Two things are asserted here and they are different claims. The first is **routing**: a small
fold now attempts the engine at all, which is what regressed. The second is **agreement**: the
two paths return the same rows — the fold is best-effort and falls back to pandas on any
failure, so a silent divergence between them would be invisible to the fallback contract and
show up only as a wrong answer.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.dist.gpu import aggregate as agg
from batcher.dist.gpu.aggregate import merge_shards

pytestmark = pytest.mark.unit


def _partials(rows: int, groups: int, shards: int = 4, width: int = 0) -> list[pa.Table]:
    """`shards` tables of `rows // shards` rows keyed on `groups` distinct values."""
    per = max(1, rows // shards)
    out = []
    for s in range(shards):
        keys = [str((i + s) % groups).ljust(width, "x") for i in range(per)]
        out.append(pa.table({"k": pa.array(keys), "c": pa.array([1] * per, pa.int64())}))
    return out


_SUM_K = [
    {
        "op": "aggregate",
        "group_keys": [{"expr": {"e": "col", "name": "k"}, "alias": "k"}],
        "aggregates": [{"func": "sum", "alias": "c", "input": {"e": "col", "name": "c"}}],
    }
]


def _pandas_fold(partials: list[pa.Table], ops: list[dict]) -> pa.Table:
    """The path `merge_shards` falls back to, called directly, as the comparison oracle."""
    import pandas as pd

    from batcher.core.gpu_plan import DfBackend
    from batcher.core.gpu_plan.execute import run_chain

    be = DfBackend(pd)
    return be.to_arrow(run_chain(pa.concat_tables(partials), ops, be))


@pytest.mark.parametrize("rows", [8, 400, 20_000])
def test_a_small_fold_is_offered_to_the_engine(monkeypatch, rows):
    """Every folded chain reaches `_native_fold`, however few rows it carries.

    Sizes span three orders of magnitude below the retired 1,048,576-row gate, because the claim
    is that the gate had no lower end rather than that it was set slightly too high.
    """
    offered: list[int] = []
    real = agg._native_fold

    def spy(partials, ops, nbytes):
        offered.append(sum(p.num_rows for p in partials))
        return real(partials, ops, nbytes)

    monkeypatch.setattr(agg, "_native_fold", spy)
    partials = _partials(rows, groups=max(2, rows // 4))
    merge_shards(partials, _SUM_K)

    assert offered == [sum(p.num_rows for p in partials)]


def test_a_row_local_chain_is_still_a_plain_concatenation(monkeypatch):
    """An empty `ops` is the concatenation itself and must not build a plan for it.

    This is the positive control for the test above: without it, an assertion that the engine is
    *offered* the fold would pass just as well against an implementation that offered it
    unconditionally, including where there is nothing to fold.
    """
    offered: list[int] = []
    monkeypatch.setattr(agg, "_native_fold", lambda *a: offered.append(1))
    partials = _partials(40, groups=10)

    out = merge_shards(partials, [])

    assert offered == []
    assert out.num_rows == sum(p.num_rows for p in partials)


@pytest.mark.parametrize(
    ("rows", "groups", "width"),
    [(8, 3, 0), (400, 40, 0), (20_000, 5_000, 0), (20_000, 5_000, 64)],
)
def test_the_engine_fold_and_the_pandas_fold_agree(rows, groups, width):
    """Both paths return the same groups and the same sums.

    The wide-key case is the shape that motivated the change: same row count, twenty times the
    bytes. It is included so agreement is checked where the two paths' costs diverge most, not
    only where they are both cheap.
    """
    partials = _partials(rows, groups, width=width)
    native = agg._native_fold(partials, _SUM_K, sum(p.nbytes for p in partials))
    assert native is not None, "the engine declined a fold it should have taken"

    expected = _pandas_fold(partials, _SUM_K)
    assert native.column_names == expected.column_names
    assert native.schema.types == expected.schema.types
    assert sorted(zip(*native.to_pydict().values(), strict=True)) == sorted(
        zip(*expected.to_pydict().values(), strict=True)
    )


def test_the_fold_falls_back_when_the_engine_declines(monkeypatch):
    """A `None` from the engine leaves the pandas answer, unchanged and complete.

    The fast path is best-effort by construction; this pins that the "effort" half cannot lose
    rows when it fails, which is the only way the widened gate could have made anything worse.
    """
    monkeypatch.setattr(agg, "_native_fold", lambda *a: None)
    partials = _partials(400, groups=40)

    out = merge_shards(partials, _SUM_K)

    assert out.num_rows == 40
    assert sum(out.to_pydict()["c"]) == sum(p.num_rows for p in partials)
