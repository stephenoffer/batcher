"""A self-join rule must decline a side it cannot identify, not raise lowering it.

`join_elim.evidence._relation_key` compares two join sides by their engine IR, and a
`map_batches` node has none: its `to_ir` raises. Optimized with sources bound — which the
distributed gate does to ask whether the aligned executor claims a plan — a semi or anti
join over a UDF operand raised `NotImplementedError` out of the optimizer instead of
leaving the join alone.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import kyber

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("how", ["anti", "semi", "inner"])
def test_a_join_over_a_udf_operand_optimizes_with_sources_bound(how):
    left = bt.from_pydict({"k": [1, 2, 3, None], "v": [1.0, 2.0, 3.0, 4.0]})
    ds = left.join(left.map_batches(lambda b: b), on="k", how=how)
    kyber.optimize_logical(ds._plan, sources=ds._sources)
    want = {"anti": 1, "semi": 3, "inner": 3}[how]
    assert ds.collect().num_rows == want
