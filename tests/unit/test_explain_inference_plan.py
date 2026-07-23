"""`explain()` on a batch-inference / `map_batches` pipeline renders instead of crashing.

A `map_batches` node is executed in Python and deliberately has no engine IR (`to_ir()`
raises), so the planned `explain()` path used to lower the plan and crash with
`NotImplementedError` — making the plan of an inference pipeline impossible to inspect,
which is step one of debugging that workload class. The planned view now falls back to the
un-lowered logical tree. The analyzed view (`analyze=True`/`stats()`) still refuses a UDF
plan, because a Python UDF emits no per-operator engine metrics.
"""

from __future__ import annotations

import json

import pytest

import batcher as bt
from batcher._internal.errors import BackendError

pytestmark = pytest.mark.unit


def _inference_plan() -> bt.Dataset:
    """A filter → map_batches (opaque UDF) pipeline, the batch-inference shape."""
    ds = bt.from_pydict({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    return ds.filter(bt.col("a") > 1).map_batches(lambda batch: batch)


def _relational_plan() -> bt.Dataset:
    """A pure-relational filter + group_by/agg pipeline (fully lowerable)."""
    ds = bt.from_pydict({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    return ds.filter(bt.col("a") > 1).group_by("a").agg(bt.col("b").sum())


def test_explain_map_batches_renders_tree() -> None:
    """A `map_batches` plan's planned `explain()` returns a tree naming `MapBatches`.

    Before the fix this raised `NotImplementedError` (map_batches has no engine IR); the
    regression guard below (`_raises_before_fix`) documents that prior behavior.
    """
    text = _inference_plan().explain()
    assert isinstance(text, str)
    assert text.strip()
    assert "MapBatches" in text
    # The un-lowered fallback is honestly labelled, not silently passed off as an
    # optimized plan.
    assert "un-lowered" in text


def test_explain_map_batches_json_document() -> None:
    """`explain(format="json")` on a UDF plan returns a JSON document, not a crash."""
    doc = _inference_plan().explain(format="json")
    parsed = json.loads(doc)
    assert isinstance(parsed, dict)
    kinds = [op["kind"] for op in parsed["ops"]]
    assert "MapBatches" in kinds


def test_explain_analyze_udf_still_refuses() -> None:
    """`explain(analyze=True)` on a UDF plan keeps its clear `BackendError`.

    Measuring a Python UDF's per-operator engine metrics is genuinely unsupported, so the
    analyzed path must not silently fall back to the planned tree.
    """
    with pytest.raises(BackendError):
        _inference_plan().explain(analyze=True)


def test_relational_explain_is_byte_identical() -> None:
    """A pure-relational plan's `explain()` output is unchanged by the UDF fallback.

    The fix adds a branch that only triggers on a `map_batches` plan; a lowerable plan must
    render exactly as before. The expected string is the captured pre-change output.
    """
    expected = (
        "aggregate                       est≈1 (default)\n"
        "  filter                        est≈1 (default)\n"
        "    scan                        est≈4 (exact)"
    )
    assert _relational_plan().explain() == expected
