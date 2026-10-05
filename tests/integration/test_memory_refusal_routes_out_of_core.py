"""A query the chunked route ran out of memory on goes out of core, not resident.

When the chunked executor gives way with `MemoryBudgetExceededError` -- its build sides over
budget, or the machine's headroom guard tripping -- and its partitioned fallback does not
apply, it returns `None`. The conductor used to answer every `None` the same way: if
admission's estimates said the plan fitted, it read every source into memory and ran there,
which is the route certain to need *more* memory than the streaming one that just ran short.
At SF1000 that is an OOM kill. `refused_for_memory` lets the conductor tell the two `None`s
apart.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import col
from batcher._internal.errors import MemoryBudgetExceededError
from batcher.api.orchestration import chunked, run

pytestmark = pytest.mark.integration


@pytest.fixture
def parquet_ds(tmp_path):
    path = tmp_path / "t.parquet"
    pq.write_table(pa.table({"k": [i % 7 for i in range(20_000)], "v": list(range(20_000))}), path)
    return bt.read.parquet(str(path))


def _want(ds) -> list[tuple]:
    return sorted(ds.group_by("k").agg(s=col("v").sum()).collect().to_pylist(), key=str)


def _count_spills(monkeypatch) -> list[int]:
    from batcher.dist import spill as spill_module

    calls: list[int] = []
    real = spill_module.spill_collect

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(spill_module, "spill_collect", counting)
    return calls


def test_a_memory_refusal_routes_out_of_core_even_when_the_estimates_fit(parquet_ds, monkeypatch):
    want = _want(parquet_ds)
    calls = _count_spills(monkeypatch)

    def refused(*_a, **_k):
        chunked._MEMORY_REFUSED.set(True)
        return None

    monkeypatch.setattr(run, "run_chunked", refused)
    got = parquet_ds.group_by("k").agg(s=col("v").sum()).collect()
    assert calls, "the refusal fell through to the resident route"
    assert sorted(got.to_pylist(), key=str) == want


def test_a_shape_refusal_still_runs_resident(parquet_ds, monkeypatch):
    """Positive control: a `None` that is *not* a memory refusal keeps the resident route."""
    want = _want(parquet_ds)
    calls = _count_spills(monkeypatch)

    def declined(*_a, **_k):
        chunked._MEMORY_REFUSED.set(False)
        return None

    monkeypatch.setattr(run, "run_chunked", declined)
    got = parquet_ds.group_by("k").agg(s=col("v").sum()).collect()
    assert not calls
    assert sorted(got.to_pylist(), key=str) == want


def test_the_chunked_route_reports_a_memory_refusal(parquet_ds, monkeypatch):
    """The flag is set by the chunked route itself when the engine gives way for memory.

    Driven through `execute_chunked` directly: a fixture this small is not routed through the
    chunked path by `collect`. The positive control is the same call with the engine allowed
    to run, which returns the result and leaves the flag clear.
    """
    from batcher import core, kyber
    from batcher.api.orchestration.sizing import projected_input_bytes

    ds = parquet_ds.group_by("k").agg(s=col("v").sum())
    sources = ds._sources
    opt, _logical, _ = kyber.optimize_full(ds._plan, sources=sources)

    def run_route():
        chunked._MEMORY_REFUSED.set(False)
        return chunked.execute_chunked(
            sources,
            opt,
            lambda i: projected_input_bytes(sources, opt.source_projections, [i]),
            python_chunks=True,
        )

    assert run_route() is not None and not chunked.refused_for_memory()

    def out_of_memory(*_a, **_k):
        raise MemoryBudgetExceededError(
            "operator state (2 bytes) exceeds the memory budget (1 bytes)"
        )

    monkeypatch.setattr(core, "execute_local_parquet", out_of_memory)
    monkeypatch.setattr(core, "execute_local_chunked", out_of_memory)
    assert run_route() is None
    assert chunked.refused_for_memory()
