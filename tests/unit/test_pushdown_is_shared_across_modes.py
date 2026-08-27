"""Every execution mode narrows the source to the same columns.

Column pushdown is decided once, by Kyber, and each mode then asks for it through its own
function: `collect()` reads `PhysicalPlan.source_projections`, `iter_batches()` calls
`stream.pipeline._pushdown`, and `collect(spill=True)` calls `dist.spill.scratch.map_projection`.
Three callers, one decision -- so the decision can reach one of them and not the others, and
the only symptom is a mode that decodes columns it discards. The rows are right either way,
which is why no differential suite can see it.

Both of the specific gaps this guards against have already shipped here. The bounded-state
drivers in `core.streaming` read the source *whole* until `_pushdown` was added: a
`group_by("user").sum("cents")` over a forty-column event decoded thirty-eight columns per
micro-batch and threw them away. `map_projection` records the same defect on the spill side --
`_iter_spill_morsels` took a `projection` parameter that no call site ever passed -- and that
one costs the most, because the source read is the dominant IO of a spilling aggregate and an
unwanted column is decoded, chunked, hash-partitioned, compressed, written to disk, and read
back.

**Deliberately compares the pure functions rather than spying on the readers.** An earlier
version of this file wrapped `read_source`, `iter_source` and `_iter_spill_morsels` to record
what each mode actually asked for. That is closer to the truth and it was unusable: those
names are bound in many modules, patching a module twice made `monkeypatch` restore a wrapper
instead of the original, and the tap then stayed unwrapped for the *next* test -- producing a
shape that passed alone, failed after its neighbour, and reported an empty read. An instrument
coming loose is indistinguishable from the defect being hunted, so the instrument was removed
instead. These three functions are pure and total, and comparing them needs no patching at all.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def wide(tmp_path_factory):
    """Five columns, of which each query below needs at most two."""
    table = pa.table(
        {
            "k": ["a", "b", "a", "c"] * 40,
            "v": list(range(160)),
            "unused1": ["z"] * 160,
            "unused2": list(range(160)),
            "unused3": [1.5] * 160,
        }
    )
    directory = tmp_path_factory.mktemp("wide")
    for part in range(4):
        pq.write_table(table, directory / f"p{part}.parquet")
    return str(directory)


def _build(name: str, path: str):
    ds = bt.read.parquet(path)
    if name == "agg_two_cols":
        return ds.group_by("k").agg(s=bt.col("v").sum())
    if name == "sort_one_col":
        return ds.select("v").sort("v")
    if name == "filter_project":
        return ds.filter(bt.col("v") > 10).select("k")
    if name == "window_two_cols":
        # `select` first: a bare `with_columns` keeps every existing column, so the
        # source legitimately needs all five and the shape would assert nothing.
        return ds.select("k", "v").with_columns(
            r=bt.row_number().over(partition_by="k", order_by="v")
        )
    raise AssertionError(name)


#: shape -> the columns source 0 must produce. Written down rather than derived, so a change
#: to pushdown has to be acknowledged here instead of silently redefining the reference.
_EXPECTED = {
    "agg_two_cols": ["k", "v"],
    "sort_one_col": ["v"],
    "filter_project": ["k", "v"],
    "window_two_cols": ["k", "v"],
}


def _collect_projection(ds) -> list[str] | None:
    from batcher import kyber

    physical, _logical, _decisions = kyber.optimize_full(ds._plan, None, ds._sources, None)
    projection = physical.source_projections.get(0)
    return sorted(projection) if projection else None


def _stream_projection(ds) -> list[str] | None:
    from batcher.api.terminal.stream.pipeline import _pushdown

    projection = _pushdown(ds._plan)
    return sorted(projection) if projection else None


def _spill_projection(ds) -> list[str] | None:
    from batcher.dist.spill.scratch import map_projection

    projection = map_projection(ds._plan, 0)
    return sorted(projection) if projection else None


_MODES = {
    "collect": _collect_projection,
    "stream": _stream_projection,
    "spill": _spill_projection,
}


@pytest.mark.parametrize("shape", sorted(_EXPECTED))
@pytest.mark.parametrize("mode", sorted(_MODES))
def test_a_mode_narrows_the_source_the_same_way(shape, mode, wide):
    got = _MODES[mode](_build(shape, wide))
    assert got == _EXPECTED[shape], (
        f"{mode} narrows source 0 to {got}, but {shape} needs {_EXPECTED[shape]}; the "
        "pushdown Kyber decided reached some modes and not this one, so it decodes columns "
        "it discards"
    )


def test_the_reference_is_actually_narrower_than_the_table(wide):
    """Guard against a vacuous suite.

    Every assertion above compares against `_EXPECTED`. If pushdown stopped narrowing at all
    and `_EXPECTED` were updated to match, the file would still pass while asserting that
    every mode reads everything. Pin the property that makes the comparison worth making:
    the reference is a strict subset of the table's columns.
    """
    table_columns = set(bt.read.parquet(wide).schema.names)
    assert len(table_columns) == 5
    for shape, expected in _EXPECTED.items():
        assert set(expected) < table_columns, f"{shape} does not narrow the source at all"
