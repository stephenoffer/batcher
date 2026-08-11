"""A benchmark case that kills its process costs one row, not the rest of the run.

`compare()` catches an exception per engine, so a query that *raises* is already one
`ERROR` row in a table that still reports everything else. Nothing catches a signal, and
a suite where one query in four is OOM-killed therefore reports nothing rather than the
three quarters that work — which is the state the Join Order Benchmark is in.

These tests pin the two halves of `benchmarks/harness.py` that make a fatal case
survivable: the wire that carries a child's result back intact, and the fallback that
turns a child which died without printing one into a `KILLED` row naming the signal.

The subprocess is stubbed, so this needs no dataset, no engine, and no second process.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))

from harness import (  # noqa: E402
    RESULT_PREFIX,
    CompareResult,
    EngineResult,
    emit_result,
    run_isolated,
)

pytestmark = pytest.mark.unit


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["stub"], returncode=returncode, stdout=stdout, stderr=""
    )


def test_a_result_survives_the_trip_through_the_child(monkeypatch, capsys):
    """Every field the table renders comes back out of the wire unchanged."""
    original = CompareResult(name="job-q1a", status="OK", note="")
    original.engines["batcher"] = EngineResult(ms=12.5, error=None, correct=True)
    original.engines["duckdb"] = EngineResult(ms=25.0, error=None, correct=True)

    emit_result(original)
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith(RESULT_PREFIX))

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(line + "\n"))
    (round_tripped,) = run_isolated(["job-q1a"])

    assert round_tripped.name == "job-q1a"
    assert round_tripped.status == "OK"
    assert round_tripped.engines["batcher"].ms == 12.5
    assert round_tripped.engines["duckdb"].ms == 25.0
    assert round_tripped.engines["batcher"].correct is True


def test_a_failed_case_keeps_its_status_and_note(monkeypatch, capsys):
    """A FAILED row is the benchmark working; isolation must not launder it into OK."""
    original = CompareResult(name="q98", status="FAILED", note="3527 rows vs 2521")
    original.engines["batcher"] = EngineResult(ms=None, error="mismatch", correct=False)

    emit_result(original)
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith(RESULT_PREFIX))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(line + "\n"))

    (round_tripped,) = run_isolated(["q98"])

    assert round_tripped.status == "FAILED"
    assert round_tripped.note == "3527 rows vs 2521"
    assert round_tripped.engines["batcher"].correct is False
    assert round_tripped.engines["batcher"].error == "mismatch"


def test_a_killed_child_becomes_one_row_naming_the_signal(monkeypatch):
    """SIGKILL is the JOB failure mode: it must not end the run."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed("", returncode=-9))

    (result,) = run_isolated(["job-q7c"])

    assert result.name == "job-q7c"
    assert result.status == "KILLED"
    assert "SIGKILL" in result.note


def test_a_child_that_exits_without_a_result_is_not_silently_dropped(monkeypatch):
    """A non-zero exit with no result line is still a row, with its code recorded."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed("boom\n", returncode=1))

    (result,) = run_isolated(["job-q10a"])

    assert result.status == "KILLED"
    assert "exited 1" in result.note


def test_one_dead_case_does_not_cost_the_ones_after_it(monkeypatch):
    """The whole point: a fatal case is one row, and the rest of the suite still reports."""
    calls: list[str] = []

    def fake_run(argv, **_kwargs):
        case = argv[argv.index("--isolate-case") + 1]
        calls.append(case)
        if case == "dies":
            return _completed("", returncode=-9)
        payload = {
            "name": case,
            "status": "OK",
            "note": "",
            "engines": {"batcher": {"ms": 1.0, "error": None, "correct": True}},
        }
        return _completed(RESULT_PREFIX + json.dumps(payload) + "\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run.py", "--benchmark", "job", "--isolate"])

    results = run_isolated(["before", "dies", "after"])

    assert calls == ["before", "dies", "after"]
    assert [r.status for r in results] == ["OK", "KILLED", "OK"]
    assert [r.name for r in results] == ["before", "dies", "after"]


def test_the_child_command_line_is_the_parents_minus_isolate(monkeypatch):
    """Every flag reaches the child, and --isolate does not, or the child recurses."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--benchmark", "job", "--engines", "batcher,duckdb", "--isolate"],
    )
    seen: dict[str, list[str]] = {}

    def fake_run(argv, **_kwargs):
        seen["argv"] = argv
        return _completed("", returncode=-9)

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_isolated(["job-q1a"])

    argv = seen["argv"]
    assert "--isolate" not in argv
    assert argv[-2:] == ["--isolate-case", "job-q1a"]
    assert "--benchmark" in argv and "job" in argv
    assert "batcher,duckdb" in argv
