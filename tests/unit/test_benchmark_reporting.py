"""What the benchmark harness records and reports beside the best-of-N headline.

A best-of-N minimum is an optimistic tail, a geomean hides where the time goes, and a gap
in the table can mean four different things. These tests pin the parts of the harness that
keep each of those visible: every repetition and the first call are recorded, the timing
order rotates per case, an untimed case is counted rather than dropped, and a gap says why.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pytest

_BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))

from harness import (  # noqa: E402
    RESULT_PREFIX,
    CompareResult,
    EngineResult,
    compare,
    emit_result,
    summarize,
    timing_order,
)
from harness.report import _parse_result, cell_status  # noqa: E402

pytestmark = pytest.mark.unit


def _const(values: list[float]):
    return lambda: pa.table({"x": values})


def test_compare_keeps_every_repetition_and_the_first_call() -> None:
    fns = {"duckdb": _const([1.0, 2.0]), "batcher": _const([2.0, 1.0])}
    result = compare("q1", fns, ["batcher", "duckdb"], runs=4)
    assert result.status == "OK"
    for er in result.engines.values():
        assert len(er.samples_ms) == 4
        assert er.ms == min(er.samples_ms)
        assert er.first_ms is not None
        assert er.cpu_ms is not None
        assert er.median_ms is not None
        assert er.p95_ms == max(er.samples_ms)


def test_a_wrong_infinite_answer_fails_the_row() -> None:
    fns = {"duckdb": _const([1.0]), "batcher": _const([float("inf")])}
    result = compare("q1", fns, ["batcher", "duckdb"], runs=1)
    assert result.status == "FAILED"
    assert result.engines["batcher"].correct is False


def test_timing_order_rotates_across_cases_and_is_stable() -> None:
    lineup = ["batcher", "duckdb", "polars"]
    firsts = {timing_order(f"q{i}", lineup)[0] for i in range(40)}
    assert firsts == set(lineup)
    assert timing_order("q7", lineup) == timing_order("q7", lineup)
    assert sorted(timing_order("q7", lineup)) == sorted(lineup)


def test_an_untimed_case_is_counted_not_dropped() -> None:
    ok = CompareResult(name="q1")
    ok.engines = {
        "batcher": EngineResult(ms=10.0, correct=True),
        "duckdb": EngineResult(ms=5.0, correct=True),
    }
    fast = CompareResult(name="q2")
    fast.engines = {
        "batcher": EngineResult(ms=2.0, correct=True),
        "duckdb": EngineResult(ms=4.0, correct=True),
    }
    unsupported = CompareResult(name="q3")
    unsupported.engines = {
        "batcher": EngineResult(ms=3.0, correct=True),
        "duckdb": EngineResult(error="n/a"),
    }
    (s,) = summarize([ok, fast, unsupported], ["batcher", "duckdb"])
    assert s.included == 2
    assert s.excluded == {"duckdb n/a": 1}
    assert s.provenance() == "2 of 3 cases (1 duckdb n/a)"
    # The geomean says 1.0x; the totals say Batcher took 12 ms against 9 ms.
    assert s.value == pytest.approx(1.0)
    assert s.batcher_total_ms == 12.0
    assert s.engine_total_ms == 9.0
    assert s.worst == ("q1", 2.0)
    assert "ratio of totals 1.33x" in s.weight()


def test_a_gap_says_why() -> None:
    assert cell_status(EngineResult(error="n/a")) == "n/a"
    assert cell_status(EngineResult(error="MemoryError: allocation failed")) == "OOM"
    assert cell_status(EngineResult(error="RuntimeError: boom")) == "ERR"
    assert cell_status(EngineResult()) == "-"


def test_the_distribution_survives_the_isolated_child(capsys: pytest.CaptureFixture) -> None:
    original = CompareResult(name="q1")
    original.engines["batcher"] = EngineResult(
        ms=1.0, correct=True, first_ms=9.0, samples_ms=[1.0, 2.0, 3.0], cpu_ms=4.0
    )
    emit_result(original)
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith(RESULT_PREFIX))
    back = _parse_result(line).engines["batcher"]
    assert (back.first_ms, back.samples_ms, back.cpu_ms) == (9.0, [1.0, 2.0, 3.0], 4.0)
