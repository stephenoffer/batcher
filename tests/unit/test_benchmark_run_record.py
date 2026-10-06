"""`benchmarks/run.py --json-out` keeps every input an aggregate is computed from.

The printed table and geomean are renderings: once the terminal scrolls, the per-sample
timings and the rows a geomean leaves out are gone, so a ratio quoted in
BENCHMARK_RESULTS.md cannot be recomputed. These tests pin that the run record carries the
machine fingerprint, the arguments, and every case of every repeat -- a ``FAILED`` row and
a ``KILLED`` row included -- and that a case is serialized by the same function the
``--isolate`` wire uses, so the file and the wire cannot drift apart.

No dataset or engine runs: the dataset loop and the environment guards are stubbed.
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
    case_payload,
    emit_result,
    run_isolated,
)

pytestmark = pytest.mark.unit

_FINGERPRINT = {
    "host": "box",
    "cpu_model": "cpu",
    "cpu_count_available": 8,
    "cpu_count_logical": 8,
    "memory_bytes": 1 << 34,
    "load_per_core_at_start": 0.1,
    "engine": "0.0.0",
    "engine_profile": "release",
    "git_sha": "abc123",
}


def _results(monkeypatch: pytest.MonkeyPatch) -> list[CompareResult]:
    ok = CompareResult(name="q1", status="OK")
    ok.engines["batcher"] = EngineResult(ms=10.0, correct=True, samples_ms=[10.0, 11.0, 12.5])
    ok.engines["duckdb"] = EngineResult(ms=20.0, correct=True, samples_ms=[20.0, 21.0])
    failed = CompareResult(name="q2", status="FAILED", note="1 row differs")
    failed.engines["batcher"] = EngineResult(ms=5.0, correct=False, samples_ms=[5.0])
    failed.engines["duckdb"] = EngineResult(ms=6.0, correct=True, samples_ms=[6.0])
    # A KILLED row as the parent really builds one: from a child that died on SIGKILL.
    with monkeypatch.context() as m:
        m.setattr(
            subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                args=["stub"], returncode=-9, stdout="", stderr=""
            ),
        )
        (killed,) = run_isolated(["q3"])
    return [ok, failed, killed]


def test_the_wire_line_is_the_shared_case_payload(monkeypatch, capsys):
    (result, *_rest) = _results(monkeypatch)
    emit_result(result)
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith(RESULT_PREFIX))
    assert json.loads(line[len(RESULT_PREFIX) :]) == case_payload(result)


def test_json_out_records_every_case_of_every_repeat(monkeypatch, tmp_path, capsys):
    import run

    calls: list[str] = []
    rows = _results(monkeypatch)

    def fake_run_dataset(benchmark, args, engines):
        calls.append(benchmark)
        return rows

    out = tmp_path / "run.json"
    monkeypatch.setattr(
        sys, "argv", ["run.py", "--benchmark", "tpch", "--repeat", "2", "--json-out", str(out)]
    )
    monkeypatch.setattr(run, "_run_dataset", fake_run_dataset)
    monkeypatch.setattr(run, "machine_fingerprint", lambda: dict(_FINGERPRINT))
    monkeypatch.setattr(run, "require_release_build", lambda **_k: None)
    monkeypatch.setattr(run, "require_quiet_box", lambda **_k: None)
    monkeypatch.setattr(run.engines_mod, "default_names", lambda _tier: ["batcher", "duckdb"])
    monkeypatch.setattr(run.engines_mod, "resolve", lambda names: [_Engine(n) for n in names])

    assert run.main() == 1  # a FAILED and a KILLED row fail the run, and are still recorded
    assert calls == ["tpch", "tpch"]

    record = json.loads(out.read_text())
    assert record["fingerprint"] == _FINGERPRINT
    assert record["args"]["repeat"] == 2
    assert record["args"]["benchmark"] == "tpch"
    assert len(record["runs"]) == 2
    for run_cases in record["runs"]:
        assert [c["status"] for c in run_cases] == ["OK", "FAILED", "KILLED"]
        assert run_cases[0]["engines"]["batcher"]["samples_ms"] == [10.0, 11.0, 12.5]
        assert run_cases[1]["note"] == "1 row differs"
        assert "SIGKILL" in run_cases[2]["note"]
    # The per-repeat geomean is recorded beside the cases it was computed from.
    assert len(record["summaries"]) == 2
    (summary,) = record["summaries"][0]
    assert summary["engine"] == "duckdb"
    assert summary["included"] == 1


def test_an_isolate_child_does_not_write_the_record(monkeypatch, tmp_path):
    import run

    out = tmp_path / "run.json"
    argv = ["run.py", "--json-out", str(out), "--isolate-case", "q1"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(run, "_run_dataset", lambda *_a: _results(monkeypatch)[:1])
    monkeypatch.setattr(run, "machine_fingerprint", lambda: dict(_FINGERPRINT))
    monkeypatch.setattr(run, "require_release_build", lambda **_k: None)
    monkeypatch.setattr(run.engines_mod, "default_names", lambda _tier: ["batcher", "duckdb"])
    monkeypatch.setattr(run.engines_mod, "resolve", lambda names: [_Engine(n) for n in names])

    assert run.main() == 0
    assert not out.exists()


class _Engine:
    def __init__(self, name: str) -> None:
        self.name = name

    def prepare(self) -> None:
        return None
