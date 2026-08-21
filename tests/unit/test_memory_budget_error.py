"""The memory envelope's refusal is a *type*, not a message.

Most stateful operators spill when they would exceed the envelope. A few cannot, because
they need one global order over the whole relation: a window with no `PARTITION BY`, an
ASOF join with no `by` keys, and the right side of a range join. Those raise rather than
risking the process.

That refusal is the one execution failure with an obvious programmatic answer -- raise the
envelope, or re-plan so the non-spillable operator is not on the path -- and it only ever
reaches a caller who asked for a ceiling to begin with. It used to arrive as a bare
`RuntimeError`, so the only way to recognize it was to match on the message text, which is
not a contract anybody should have to depend on.

What is pinned here is that plumbing: the engine's typed error crosses the FFI as a named
exception the control plane re-exports. The operator semantics behind it belong to the
engine's own tests, which can pin the budget exactly.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from batcher._internal.errors import MemoryBudgetExceededError, ResourceError

pytestmark = pytest.mark.unit

#: A window with no `PARTITION BY` over one scan -- the operator that provably cannot spill,
#: written as IR rather than built through the optimizer, so the plan under test does not
#: move with whatever the learning loop happens to have seen.
_RANKING_PLAN = json.dumps(
    {
        "op": "window",
        "input": {"op": "scan", "source_id": 0},
        "partition_keys": [],
        "order_keys": [
            {"expr": {"e": "col", "name": "v"}, "descending": False, "nulls_first": False}
        ],
        "functions": [{"func": "rank", "alias": "r", "offset": 1}],
        "rank_limit": None,
    }
)

#: Enough rows that the ranking state is megabytes, so a 100 KB envelope is refused and a
#: 512 MB one is not.
_ROWS = 300_000


def test_it_is_a_named_exception_a_caller_can_catch():
    """Importable by name from the one error module, whether or not the extension is built.

    The native build re-exports it from `_native`, where it inherits `RuntimeError`; the
    pure-Python shim used without the extension puts it under `ResourceError`, which is where
    it belongs in the hierarchy. Either way a caller names it rather than reading a message.
    """
    assert issubclass(MemoryBudgetExceededError, Exception)
    assert MemoryBudgetExceededError.__name__ == "MemoryBudgetExceededError"
    assert issubclass(ResourceError, Exception)


def _run_in_a_fresh_process(budget: int) -> str:
    """Execute the ranking plan under `budget` bytes in a new interpreter, and report.

    A subprocess, and that is the point rather than an inconvenience. The data plane's
    memory pool is **process-wide** and its limit only ever grows (`bc_py::process`), so
    admission in this process is decided by the largest envelope any earlier query used --
    which in a test suite is whatever ran before. In-process, this assertion passes alone and
    fails after any neighbour that ran a query under a wide budget, which is a test that
    reports the suite's history rather than the engine's behaviour.

    Args:
        budget: The `max_memory_bytes` the query runs under.

    Returns:
        Either ``"raised:<exception type name>"`` or ``"rows:<count>"``.
    """
    script = textwrap.dedent(
        """
        import json, sys
        import pyarrow as pa
        from batcher._internal.native import engine
        from batcher._internal.errors import MemoryBudgetExceededError
        from batcher.config import Config, MemoryConfig

        plan, rows, budget = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
        cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=budget))
        sources = [pa.table({"v": pa.array(range(rows), pa.int64())}).to_batches()]
        try:
            out = engine().execute_plan(plan, sources, cfg.engine_config_json())
            print(f"rows:{sum(b.num_rows for b in out)}")
        except MemoryBudgetExceededError as exc:
            print(f"raised:{type(exc).__name__}:{exc}")
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script, _RANKING_PLAN, str(_ROWS), str(budget)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, f"child failed:\n{done.stderr}"
    return done.stdout.strip()


def test_an_operator_that_cannot_spill_raises_it_across_the_ffi():
    """Over the envelope, the engine's typed error arrives in Python with its type intact."""
    pytest.importorskip("batcher._native", reason="native engine not built")
    result = _run_in_a_fresh_process(100_000)
    assert result.startswith("raised:MemoryBudgetExceededError"), result
    # Both figures and the reason, which is what makes the error actionable rather than a
    # notification that something went wrong.
    assert "memory budget" in result
    assert "cannot spill" in result
    assert "PARTITION BY" in result


def test_the_same_plan_inside_a_large_enough_envelope_succeeds():
    """The refusal is about the envelope, not the shape: raise it and the plan runs.

    This is what makes the error worth catching. A caller that widens the envelope and
    retries gets an answer, so `except MemoryBudgetExceededError` is a recovery path rather
    than a nicer way to report a dead end.
    """
    pytest.importorskip("batcher._native", reason="native engine not built")
    assert _run_in_a_fresh_process(512_000_000) == f"rows:{_ROWS}"
