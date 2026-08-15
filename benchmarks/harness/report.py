"""Rendering a run's results, and running each case in its own process.

``print_table`` renders the aligned per-engine table. ``emit_result`` / ``_parse_result``
are the one-line wire format an isolated child uses to hand a result back, and
``run_isolated`` is what keeps a case that *kills its process* from taking the whole suite's
report with it.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys

from .compare import CompareResult, EngineResult


# Reporting
# --------------------------------------------------------------------------- #
def _fmt_ms(er: EngineResult) -> str:
    if er.error == "n/a":
        return "n/a"
    if er.error:
        return "ERR"
    if er.ms is None:
        return "-"
    return f"{er.ms:.1f}"


def print_table(results: list[CompareResult], engines: list[str]) -> None:
    """Print an aligned table: query | per-engine ms | batcher/<engine> ratios | status.

    Columns are driven by ``engines`` (the resolved lineup), so the table adapts to
    whatever single-node or multi-node engines were selected. A ``b/<engine>`` ratio
    is shown for every comparator when Batcher is in the lineup.
    """
    has_batcher = "batcher" in engines
    comparators = [e for e in engines if e != "batcher"]
    headers = ["query"] + [f"{e}_ms" for e in engines]
    if has_batcher:
        headers += [f"b/{e}" for e in comparators]
    headers += ["status"]

    rows = []
    for r in results:
        cells = [r.name] + [_fmt_ms(r.engines.get(e, EngineResult())) for e in engines]
        if has_batcher:
            b = r.engines.get("batcher", EngineResult())
            for e in comparators:
                ce = r.engines.get(e, EngineResult())
                cells.append(f"{b.ms / ce.ms:.2f}x" if b.ms and ce.ms else "-")
        cells.append(r.status)
        rows.append(cells)

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            if i == 0:
                out.append(cell.ljust(widths[i]))
            else:
                out.append(cell.rjust(widths[i]))
        return "  ".join(out)

    line = "-" * (sum(widths) + 2 * (len(widths) - 1))
    print(fmt_row(headers))
    print(line)
    for row in rows:
        print(fmt_row(row))

    # Footnotes for any failed / partial rows.
    notes = [r for r in results if r.note]
    if notes:
        print()
        for r in notes:
            print(f"[{r.status}] {r.name}: {r.note}")


# --------------------------------------------------------------------------- #
# Per-case process isolation, so a query that kills the process costs one row
# --------------------------------------------------------------------------- #
#
# ``compare()`` catches an exception per engine, so a query that *raises* is already one
# ``ERROR`` row in a table that still reports every other query. Nothing catches a signal.
# A query the OOM killer takes, or one that aborts inside a native kernel, ends the whole
# runner — and with it every result after it, including the ones already computed.
#
# That is not a hypothetical failure mode here. On the Join Order Benchmark a per-query
# survey found 24 of the first 85 queries dying by ``SIGKILL`` rather than raising, spread
# through the suite rather than clustered, so no ``--skip`` list makes a full run reachable
# and the suite reports nothing instead of the three quarters that work.
#
# `run_isolated` runs each case in its own subprocess. The child does exactly what the
# in-process loop would do for that one case and prints its ``CompareResult`` as JSON; the
# parent reads it back. A child that dies without printing one becomes a ``KILLED`` row
# carrying the signal that killed it, which is the same shape ``ERROR`` already has and
# reports the same fact the survey had to reconstruct by hand.
#
# Two properties are deliberate:
#
# **Isolation is per case, not per engine.** The comparison is the unit of meaning — a
# timing without the oracle's answer beside it is not a result — so a child runs the whole
# lineup for one query.
#
# **There is no timeout.** A wall clock cannot distinguish a hang from a query that is
# merely slow, and this suite has both: TPC-DS q72 legitimately takes ~30 s single-node and
# scale-factor runs take minutes. Marking a slow-but-correct query as failed would be a
# worse error than the one this module fixes, so a hang still needs ``--skip`` or a human.

#: Marks the one stdout line a child uses to hand its result back. A prefix rather than
#: "parse the last line" because the engines print freely and a native library may write
#: to the same stream after the result is known.
RESULT_PREFIX = "__BENCH_RESULT__ "


def emit_result(result: CompareResult) -> None:
    """Print `result` on the wire the parent reads. Called in the child."""
    payload = {
        "name": result.name,
        "status": result.status,
        "note": result.note,
        "engines": {
            name: {"ms": er.ms, "error": er.error, "correct": er.correct}
            for name, er in result.engines.items()
        },
    }
    print(RESULT_PREFIX + json.dumps(payload), flush=True)


def _parse_result(line: str) -> CompareResult:
    """Rebuild a `CompareResult` from the child's wire line."""
    payload = json.loads(line[len(RESULT_PREFIX) :])
    result = CompareResult(
        name=payload["name"], status=payload["status"], note=payload.get("note", "")
    )
    for name, er in payload.get("engines", {}).items():
        result.engines[name] = EngineResult(
            ms=er.get("ms"), error=er.get("error"), correct=er.get("correct")
        )
    return result


def _child_argv(case: str) -> list[str]:
    """This process's command line, aimed at exactly one case.

    Rebuilt from ``sys.argv`` rather than from the parsed namespace so every flag the
    parent was given — engines, scale, source, memory cap, spill dir — reaches the child
    without this module having to know the CLI. Only ``--isolate`` is dropped, or the
    child would recurse.
    """
    argv = [a for a in sys.argv[1:] if a != "--isolate"]
    return [sys.executable, sys.argv[0], *argv, "--isolate-case", case]


def _death(returncode: int) -> str:
    """Describe how a child that printed no result died."""
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:  # pragma: no cover - an unknown signal number
            name = f"signal {-returncode}"
        return f"killed by {name}"
    return f"exited {returncode} without a result"


def run_isolated(case_names: list[str]) -> list[CompareResult]:
    """Run each named case in its own subprocess and collect the results.

    A child that dies without printing a result yields a ``KILLED`` row rather than
    ending the run. The child's own output is forwarded on failure only, because that
    is where the traceback or the allocator's last words are, and forwarding it always
    would bury the table.

    Args:
        case_names: Case names to run, in report order. The caller has already applied
            ``--family`` / ``--only`` / ``--skip``, so every name here is meant to run.

    Returns:
        One result per name, in the same order.
    """
    results: list[CompareResult] = []
    for i, case in enumerate(case_names, start=1):
        print(f"[{i}/{len(case_names)}] {case} ...", flush=True)
        proc = subprocess.run(
            _child_argv(case),
            capture_output=True,
            text=True,
            check=False,
        )
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(RESULT_PREFIX)), None)
        if line is None:
            note = _death(proc.returncode)
            print(f"    {note}", flush=True)
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-12:]
            for entry in tail:
                print(f"    | {entry}", flush=True)
            results.append(CompareResult(name=case, status="KILLED", note=note))
            continue
        results.append(_parse_result(line))
    print()
    return results
