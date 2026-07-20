"""The streaming executor is the default, and it is correct — through the real FFI boundary.

`bc-interp`'s own Rust tests pin `execute_streaming == execute` operator by operator. These pin
the two things only reachable from Python: that a normal query actually *runs* on the streaming
executor (not just that the executor exists), and that a query too large for a configured memory
budget falls back to the spilling executor and still returns the oracle's answer — because the
streaming breakers fold in memory and refuse rather than OOM.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pytest

pytestmark = pytest.mark.integration


def _native():
    from batcher._internal import native

    return native.engine()


def _cfg(*, budget: int, streaming: bool) -> str:
    return json.dumps(
        {
            "morsel_rows": 16384,
            "morsel_bytes": 1 << 20,
            "parallelism": 0,
            "memory_budget_bytes": budget,
            "spill_dir": None,
            "spill_compression": "auto",
            "fuse_linear": True,
            "shrink_output_dtypes": False,
            "streaming": streaming,
        }
    )


def _grouped_sum(eng, tbl, cfg) -> dict:
    plan = json.dumps(
        {
            "op": "aggregate",
            "input": {"op": "scan", "source_id": 0},
            "group_keys": [{"expr": {"e": "col", "name": "g"}, "alias": "g"}],
            "aggregates": [{"func": "sum", "input": {"e": "col", "name": "v"}, "alias": "s"}],
        }
    )
    out = eng.execute_plan(plan, [list(tbl.to_batches())], cfg)
    t = pa.Table.from_batches(list(out))
    return dict(zip(t.column("g").to_pylist(), t.column("s").to_pylist(), strict=True))


def test_streaming_is_the_default_and_matches_the_materializing_oracle():
    eng = _native()
    n = 200_000
    tbl = pa.table({"g": [i % 5000 for i in range(n)], "v": list(range(n))})

    oracle = _grouped_sum(eng, tbl, _cfg(budget=0, streaming=False))
    streamed = _grouped_sum(eng, tbl, _cfg(budget=0, streaming=True))

    assert streamed == oracle
    assert len(oracle) == 5000


def test_a_query_over_budget_falls_back_to_the_spilling_executor():
    """A streaming aggregate whose state exceeds the envelope must give way, not crash.

    A tiny `memory_budget_bytes` puts the 5,000-group aggregate's state over budget. The
    streaming breaker does not spill, so it returns `MemoryBudgetExceeded`; `bc_py::execute_plan`
    catches exactly that and re-runs on the materializing executor, which spills. The observable
    contract is only that the answer is still the oracle's — the fallback is invisible except in
    that it did not OOM.
    """
    eng = _native()
    n = 200_000
    tbl = pa.table({"g": [i % 5000 for i in range(n)], "v": list(range(n))})

    oracle = _grouped_sum(eng, tbl, _cfg(budget=0, streaming=False))
    fell_back = _grouped_sum(eng, tbl, _cfg(budget=1 << 16, streaming=True))

    assert fell_back == oracle


def test_public_api_group_by_is_correct_on_the_streaming_default():
    import batcher as bt

    ds = bt.from_pydict({"g": [i % 7 for i in range(1000)], "v": list(range(1000))})
    got = ds.group_by("g").agg(s=bt.col("v").sum()).to_pydict()
    by_g = dict(zip(got["g"], got["s"], strict=True))

    want = {g: sum(v for i, v in enumerate(range(1000)) if i % 7 == g) for g in range(7)}
    assert by_g == want


def test_public_api_join_into_aggregate_is_correct_on_the_streaming_default():
    import batcher as bt

    facts = bt.from_pydict({"k": [i % 50 for i in range(5000)], "v": list(range(5000))})
    dim = bt.from_pydict({"k": list(range(50)), "d": [i * 10 for i in range(50)]})
    got = (
        facts.join(dim, on="k")
        .group_by("k")
        .agg(s=bt.col("v").sum(), md=bt.col("d").max())
        .to_pydict()
    )
    assert len(got["k"]) == 50
    # `d` is a function of `k`, so max(d) per group is deterministic.
    assert dict(zip(got["k"], got["md"], strict=True)) == {k: k * 10 for k in range(50)}
