"""A source the plan does not scan is not read, and the one it does scan still is.

The bound source list belongs to the *query*; the plan handed to `resolve_sources` is often
a *sub*-plan of it. The adaptive staging loop executes one breaker's subtree at a time
against the full list, so a stage scanning two tables was handed all six and read every one.

That is not merely wasted I/O, and this is the half worth pinning: projection pushdown
records a column list only for scans the plan *contains*, and an absent entry means "read
every column". So the sources a stage does not use are exactly the ones it reads at full
width. Measured on TPC-H q9 at sf10, the first stage resolved 8.9 GiB where its own plan
needed 2.9 GiB, and the query was killed by the OOM killer at 20.8 GiB resident. With the
unscanned sources skipped it peaks at 14.5 GiB and returns its 175 rows.

Both directions are asserted here, because "reads nothing" is trivially satisfiable by a
`resolve_sources` that reads nothing at all.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.orchestration import stages
from batcher.core import ExecutionContext
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend
from batcher.plan.physical import PhysicalPlan

pytestmark = pytest.mark.unit


def _two_sources() -> list:
    """Two independent in-memory relations, bound at indices 0 and 1."""
    left = bt.from_pydict({"a": [1, 2, 3]})
    right = bt.from_pydict({"b": [4, 5, 6]})
    return [left._sources[0], right._sources[0]]


def _plan_scanning(*source_ids: int) -> PhysicalPlan:
    """A plan whose IR reads exactly `source_ids` — nested, so the walk has to descend."""
    scans = [{"op": "scan", "source_id": sid} for sid in source_ids]
    ir = scans[0] if len(scans) == 1 else {"op": "union", "inputs": scans}
    return PhysicalPlan(ir={"op": "limit", "input": ir, "n": 1}, output_schema=None)


def _reads(monkeypatch, opt: PhysicalPlan) -> tuple[list[int], object]:
    """Resolve `opt`'s sources, returning the row counts of the reads that happened."""
    seen: list[int] = []
    original = stages.read_source

    def _counting(src, projection=None, predicate=None, limit=None, ordering=None):
        batches = original(src, projection, predicate, limit, ordering)
        seen.append(sum(b.num_rows for b in batches))
        return batches

    monkeypatch.setattr(stages, "read_source", _counting)
    ctx = ExecutionContext(columns=[], hub=MetadataHub(InProcessBackend()))
    resolved = stages.resolve_sources(_two_sources(), opt, ctx)
    return seen, resolved


def test_the_source_a_plan_does_not_scan_is_never_read(monkeypatch):
    """One scan, two bound sources: exactly one read, and the other resolves to no batches."""
    seen, resolved = _reads(monkeypatch, _plan_scanning(0))

    assert seen == [3], "only the scanned source should have been read"
    assert [len(b) for b in resolved.batches] == [1, 0]
    assert sum(b.num_rows for b in resolved.batches[0]) == 3
    # Nothing was read for source 1, so nothing was proven about its size. Claiming a
    # complete scan there would record a zero-row table in the cardinality model.
    assert resolved.complete[1] is False


def test_the_sources_a_plan_does_scan_are_all_read(monkeypatch):
    """The positive control: with both scanned, both are read, at their real row counts."""
    seen, resolved = _reads(monkeypatch, _plan_scanning(0, 1))

    assert sorted(seen) == [3, 3], "both scanned sources should have been read"
    assert [sum(b.num_rows for b in rel) for rel in resolved.batches] == [3, 3]


def test_the_scan_walk_descends_into_every_ir_shape():
    """`scanned_source_ids` finds a scan wherever the IR nests it.

    Under-reporting is the failure that costs a wrong answer — a caller skipping a source
    the plan does read — so the walk is structural rather than keyed on operator names it
    knows. A scan buried under an unfamiliar key must still be found.
    """
    ir = {
        "op": "hash_join",
        "left": {"op": "filter", "input": {"op": "scan", "source_id": 7}},
        "right": {"op": "some_future_op", "branches": [{"op": "scan", "source_id": 2}]},
    }
    assert PhysicalPlan(ir=ir, output_schema=None).scanned_source_ids() == frozenset({7, 2})
    assert PhysicalPlan(ir={"op": "limit"}, output_schema=None).scanned_source_ids() == frozenset()


def test_a_staged_query_returns_the_same_rows_as_an_unstaged_one():
    """End to end: a join whose stages read different sources still answers correctly.

    The skip is keyed off the plan the executor is handed, so a plan that scans everything
    must be unaffected. This is the shape the skip could break — several sources, a breaker
    between them — run through the public API against a hand-computed answer.
    """
    left = bt.from_pydict({"k": [1, 2, 3, 4], "v": [10, 20, 30, 40]})
    right = bt.from_pydict({"k": [2, 3, 5], "w": [200, 300, 500]})
    got = (
        left.join(right, on="k", how="inner")
        .group_by("k")
        .agg(total=bt.col("v").sum() + bt.col("w").sum())
        .collect()
    )
    assert got.sort_by("k").to_pydict() == pa.table({"k": [2, 3], "total": [220, 330]}).to_pydict()
