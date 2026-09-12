"""A whole standard benchmark suite across a GPU cluster: the device tier against the CPU engine.

The GPU tier's benchmarks up to now measured *pieces* -- one group-by, one Q6 filter, one
fan-out -- because that is what the translator covered. This runs a **complete suite** through
the public API on the same cluster twice, once with ``backend="cpu"`` and once with
``backend="gpu"``, and reports per query: whether the device actually ran it, both wall times
cold and warm, and the speedup.

Two suites, chosen because they stress opposite halves of the tier. **TPC-H** is join-heavy, so
it measures how much of a multi-relation plan the tree translator matches and what the shuffle
costs. **ClickBench** is 43 queries over one wide table, which is the shape a device is
theoretically best at and the shape whose cost is almost entirely scan and reduce -- so it is
the honest test of whether device-native Parquet reading and the group-by kernels pay.

Correctness is gated before any timing is trusted: the GPU answer must match the CPU engine's
as a multiset, to a float tolerance. A query whose device answer differs is reported as a
divergence -- a fast wrong answer is a bug.

**Queries run in a child process, one at a time, and the parent restarts it.** Not defensive
padding: the head node here has 30 GB and TPC-H q4 at sf10 is enough to have the driver
SIGKILLed by the kernel, which cannot be caught, and which took a whole 22-query run down with
it after reporting three results. A crash is a fact about one query; losing the suite to it is
a fact about the harness.

Run:
    python benchmarks/gpu_backend/cluster_suite.py                        # TPC-H sf10
    BENCH_SUITE=clickbench python benchmarks/gpu_backend/cluster_suite.py
    BENCH_TPCH_SCALE=100 python benchmarks/gpu_backend/cluster_suite.py
    BENCH_ONLY=q1,q6 python benchmarks/gpu_backend/cluster_suite.py
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print = functools.partial(print, flush=True)

#: Where each suite's parquet lives. Shared storage, so every worker reads it directly and the
#: driver never stages a row. Overridable with `BENCH_BASE`.
_BASES = {
    "tpch:1": "/mnt/cluster_storage/tpch_named_small",
    "tpch:10": "/mnt/cluster_storage/tpch_named_sf10",
    "tpch:100": "/mnt/cluster_storage/tpch_sf100",
    "clickbench:0": "/mnt/cluster_storage/clickbench_named",
}

_TPCH_TABLES = (
    "lineitem",
    "orders",
    "customer",
    "part",
    "partsupp",
    "supplier",
    "nation",
    "region",
)

#: Timed repeats after the discarded warm-up. A cold GPU run pays the worker start, the cuDF
#: import and the RMM pool build; keeping it in the reported figure measures the cluster's
#: bootstrap rather than the tier, and every one of those costs is paid once per session.
_RUNS = int(os.environ.get("BENCH_RUNS", "3"))

#: Seconds one query gets across both backends before the child is killed and recorded as such.
_QUERY_TIMEOUT_S = float(os.environ.get("BENCH_QUERY_TIMEOUT_S", "900"))


def _suite() -> str:
    return os.environ.get("BENCH_SUITE", "tpch")


def _scale() -> str:
    return os.environ.get("BENCH_TPCH_SCALE", "10") if _suite() == "tpch" else "0"


def _base() -> str:
    return os.environ.get("BENCH_BASE") or _BASES[f"{_suite()}:{_scale()}"]


def _queries() -> dict[str, str]:
    """The suite's statements, read from the module that already owns them."""
    if _suite() == "clickbench":
        from suites.standard.clickbench import QUERIES
    else:
        from suites.standard.tpch import QUERIES
    return dict(QUERIES)


def _tables(bt, base: str) -> dict:
    """The suite's relations as lazy parquet scans, so no row is read on the driver.

    TPC-H parquet in the wild is named **positionally** (`column00`, `column01`, ...) — the
    public Ray mirror is, and so is the sf100 copy on this fleet — while every TPC-H query names
    its columns. Renaming on the lazy scan is schema-on-read: pure metadata, no mirror, and the
    same projection for both arms. A source that already carries canonical names pays nothing.
    """
    if _suite() == "clickbench":
        return {"hits": bt.read.parquet(base)}
    from sources.tables import TPCH_COLUMNS

    out = {}
    for name in _TPCH_TABLES:
        scan = bt.read.parquet(f"{base}/{name}")
        want = TPCH_COLUMNS[name]
        found = list(scan.schema.names)
        if found and found[0] != want[0]:
            # Positional, and possibly one column wider than TPC-H declares — the `|`-delimited
            # generator leaves a trailing empty field. Keep the ones the queries name.
            keep = min(len(want), len(found))
            renames = dict(zip(found[:keep], want[:keep], strict=True))
            scan = scan.select(*found[:keep]).rename(renames)
        out[name] = scan
    return out


# --------------------------------------------------------------------------- #
# Parent: drive the child once per query, and survive its death.
# --------------------------------------------------------------------------- #
def main() -> int:
    base = _base()
    only = {q.strip() for q in os.environ.get("BENCH_ONLY", "").split(",") if q.strip()}
    names = [n for n in _queries() if not only or n.split("-")[-1] in only]
    out_path = os.environ.get("BENCH_OUT", "")

    print(f"# {_suite()} sf{_scale()} from {base}, {len(names)} queries, {_RUNS} timed repeats")
    rows = []
    for name in names:
        rows.append(_run_child(name, base))
        print(f"  {json.dumps(rows[-1])}")
    if out_path:
        Path(out_path).write_text(json.dumps(rows, indent=2))
    _summarize(rows)
    return 0


def _run_child(name: str, base: str) -> dict:
    """Run one query in a child process; a death is that query's result, not the suite's."""
    env = {**os.environ, "BENCH_CHILD": name, "_BENCH_BASE": base}
    # The child writes to **files**, not pipes, and leads its own process group. Both are
    # required for the timeout above to fire at all.
    #
    # `capture_output=True` hands the child a pipe, and the Ray workers and raylet it starts
    # inherit that pipe. `subprocess.run` then waits for the pipe to close as well as for the
    # child to exit, so killing the child on timeout leaves the parent blocked on a descriptor
    # its grandchildren still hold. Observed here as a suite that reported one query in
    # twenty-three minutes with a 200-second per-query bound visibly not firing.
    #
    # The new session matters for the same reason from the other side: the kill has to reach
    # the whole tree, or the next query runs on a cluster still occupied by the tasks of the
    # one that was abandoned.
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        child = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__)],
            stdout=out,
            stderr=err,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            child.wait(timeout=_QUERY_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _kill_group(child)
            return {"query": name, "harness": f"timed out after {_QUERY_TIMEOUT_S:.0f}s"}
        out.seek(0)
        err.seek(0)
        proc = subprocess.CompletedProcess(
            child.args, child.returncode or 0, out.read(), err.read()
        )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
    # A negative return code is a signal. -9 is the kernel's OOM killer, which is the one this
    # child-process design exists for and the one that leaves no traceback to report.
    reason = "OOM-killed (SIGKILL)" if proc.returncode == -9 else f"exit {proc.returncode}"
    return {"query": name, "harness": reason, "tail": tail}


def _kill_group(child: subprocess.Popen) -> None:
    """Kill the child's whole process group, then reap it.

    The child leads its own session, so this reaches the Ray processes it started. Killing the
    child alone leaves them holding the GPUs the next query is about to ask for.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(os.getpgid(child.pid), signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        child.wait(timeout=30)


def _summarize(rows: list[dict]) -> None:
    """A one-screen verdict: device coverage, correctness, and the speedup distribution."""
    ran = [r for r in rows if r.get("on_device")]
    matched = [r for r in ran if r.get("match") is True]
    diverged = [r for r in ran if r.get("match") not in (True, None)]
    speedups = sorted(r["speedup"] for r in rows if isinstance(r.get("speedup"), (int, float)))
    declines: dict[str, int] = {}
    for row in rows:
        for reason, n in (row.get("declined") or {}).items():
            declines[reason] = declines.get(reason, 0) + n
    print("\n# summary")
    print(f"#   queries              {len(rows)}")
    print(f"#   reached a device     {len(ran)}")
    print(f"#   matched the CPU      {len(matched)}")
    if diverged:
        print(f"#   DIVERGED             {[r['query'] for r in diverged]}")
    for reason, n in sorted(declines.items(), key=lambda kv: -kv[1]):
        print(f"#   declined ({n:>3}x)      {reason}")
    if speedups:
        mid = speedups[len(speedups) // 2]
        print(f"#   speedup min/med/max  {speedups[0]:.2f} / {mid:.2f} / {speedups[-1]:.2f}")
        print(f"#   faster on GPU        {sum(1 for s in speedups if s > 1.0)}/{len(speedups)}")
        print(f"#   total cpu / gpu s    {_totals(rows, 'gpu')}")
    auto = sorted(
        r["auto_speedup"] for r in rows if isinstance(r.get("auto_speedup"), (int, float))
    )
    if auto:
        routed = sum(1 for r in rows if r.get("auto_on_device"))
        mid = auto[len(auto) // 2]
        print(f"#   auto routed to GPU   {routed}/{len(rows)}")
        print(f"#   auto min/med/max     {auto[0]:.2f} / {mid:.2f} / {auto[-1]:.2f}")
        print(f"#   total cpu / auto s   {_totals(rows, 'auto')}")


def _totals(rows: list[dict], arm: str) -> str:
    """Summed warm wall time over the queries both arms answered — the suite-level figure.

    Restricted to queries with *both* timings on purpose: a sum over whichever queries each arm
    happened to finish compares two different workloads and always flatters the one that failed
    more.
    """
    key = f"{arm}_s"
    pairs = [(r["cpu_s"], r[key]) for r in rows if r.get("cpu_s") and r.get(key)]
    cpu, other = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    if not other:
        return "n/a"
    return f"{cpu:.2f} / {other:.2f} over {len(pairs)} queries ({cpu / other:.2f}x)"


# --------------------------------------------------------------------------- #
# Child: one query, both backends.
# --------------------------------------------------------------------------- #
def _child(name: str) -> int:
    from cluster_env import init_gpu_cluster

    init_gpu_cluster()
    _use_persistent_metadata()
    _capture_gpu_failures()
    import batcher as bt

    tables = _tables(bt, os.environ["_BENCH_BASE"])
    print(json.dumps(_one(bt, name, _queries()[name], tables)))
    return 0


#: Where the learned statistics live across this suite's child processes. Empty keeps the
#: engine's default, which is `in_process` — and an in-process hub is *lost when the process
#: exits*, so a harness that runs one child per query (this one does, for crash isolation)
#: can never let Kyber's GPU/CPU crossover learn anything. That is a property of the harness,
#: not of the engine, and measuring `backend="auto"` without fixing it would report a router
#: that never improves as though improvement were impossible.
_METADATA_URI = os.environ.get("BENCH_METADATA_URI", "")


def _use_persistent_metadata() -> None:
    """Point this child's `MetadataHub` at a file the next child will also read."""
    if not _METADATA_URI:
        return
    import dataclasses

    from batcher.config import active_config, set_config

    metadata = dataclasses.replace(active_config().metadata, backend="sqlite", uri=_METADATA_URI)
    # `active_config().replace(...)`, never `Config().replace(...)`. `Config()` is a **fresh
    # default**, so swapping one section in that way silently reverts every other section — and
    # this runs immediately after `init_gpu_cluster` has set `distributed.gpu_rapids_path`.
    # Wiping it sends the GPU tasks back to resolving a cuDF pip block per node, which the fan-
    # outs then decline on. The visible symptom was TPC-H q14 running at 12.2x on two suite
    # passes and declining on a third: the third was the one with a populated stats hub, so it
    # read as the learned crossover turning the accelerator off, and it was this line.
    set_config(active_config().replace(metadata=metadata))


#: Device-tier failures this child saw, in order. A decline is a fact about the query and a
#: *defect* is a fact about the tier, and the ledger records only that one was counted — the
#: sentence saying which column disagreed goes to a log the harness otherwise swallows.
_FAILURES: list[str] = []


def _capture_gpu_failures() -> None:
    """Collect what `note_gpu_failure` says, instead of losing it to a suppressed logger."""
    from batcher.api.terminal.gpu_backend import failure

    original = failure.note_gpu_failure

    def _record(step, exc):
        _FAILURES.append(f"{step}: {type(exc).__name__}: {str(exc)[:400]}")
        return original(step, exc)

    failure.note_gpu_failure = _record
    for module in ("route", "fanout", "translate", "verify"):
        target = __import__(
            f"batcher.api.terminal.gpu_backend.{module}", fromlist=["note_gpu_failure"]
        )
        if hasattr(target, "note_gpu_failure"):
            target.note_gpu_failure = _record


def _one(bt, name: str, sql: str, tables: dict) -> dict:
    """Run one query on the CPU engine and on the device, and compare.

    Reports both the cold first run and the best of `_RUNS` warm ones, because the two answer
    different questions: cold is what a one-shot job pays, warm is what the tier costs.
    """
    from batcher.api.terminal.gpu_backend.audit import gpu_ledger, reset_gpu_ledger

    rec: dict = {"query": name}
    try:
        ds = bt.sql(sql, tables)
    except Exception as exc:
        rec["parse_error"] = f"{type(exc).__name__}: {exc}"[:300]
        return rec
    results = {}
    # **The arms run in this order, and the later ones read a warmer page cache.** Each gets its
    # own discarded cold run and a best-of-`_RUNS`, which amortizes most of it — but `auto`,
    # which usually decides to run on the CPU, is executing a path the `cpu` arm has already
    # warmed. Read `auto_speedup` as "auto costs nothing", not as "auto beats the CPU engine".
    #
    # `auto` is the third arm and the one a user actually runs. `gpu` is a **forced** request:
    # it honours the caller past Kyber's small-input threshold, so it measures what the device
    # tier can do rather than what the router would choose. On a dataset small enough that the
    # CPU answers in tens of milliseconds those are very different numbers, and reporting only
    # the forced one would read as a device that loses where in fact it was never asked.
    for backend in ("cpu", "gpu", "auto"):
        reset_gpu_ledger()
        _FAILURES.clear()
        try:
            t0 = time.perf_counter()
            out = ds.collect(distributed=True, backend=backend)
            rec[f"{backend}_cold_s"] = round(time.perf_counter() - t0, 3)
            best = float("inf")
            for _ in range(_RUNS):
                t0 = time.perf_counter()
                out = ds.collect(distributed=True, backend=backend)
                best = min(best, time.perf_counter() - t0)
            rec[f"{backend}_s"] = round(best, 3)
            rec[f"{backend}_rows"] = out.num_rows
            results[backend] = out
        except Exception as exc:
            rec[f"{backend}_error"] = f"{type(exc).__name__}: {exc}"[:300]
        snap = gpu_ledger().snapshot()
        if backend == "gpu":
            rec["on_device"] = bool(snap["ran"])
            if snap["declined"]:
                rec["declined"] = snap["declined"]
            if _FAILURES:
                # Deduplicated: three repeats of one query report the same sentence three times.
                rec["failures"] = sorted(set(_FAILURES))[:3]
        elif backend == "auto":
            rec["auto_on_device"] = bool(snap["ran"])
    if "cpu" in results and "gpu" in results:
        rec["match"] = _same(results["cpu"], results["gpu"])
        if rec.get("cpu_s") and rec.get("gpu_s"):
            rec["speedup"] = round(rec["cpu_s"] / rec["gpu_s"], 2)
    if "cpu" in results and "auto" in results:
        rec["auto_match"] = _same(results["cpu"], results["auto"])
        if rec.get("cpu_s") and rec.get("auto_s"):
            rec["auto_speedup"] = round(rec["cpu_s"] / rec["auto_s"], 2)
    return rec


def _same(left, right):
    """Order-independent, float-tolerant multiset comparison of two Arrow tables."""
    from harness.compare import results_match

    ok, why = results_match(left, right)
    return True if ok else why


if __name__ == "__main__":
    # Before anything is timed, and in *both* processes: the parent publishes the summary and
    # the child produces every number in it, so guarding only one of them would leave a
    # dev-profile engine — 8-60x slower — reporting a suite total with nothing said about it.
    # Imported here rather than at the top because `envinfo` lives two directories up and is
    # only importable after the `sys.path` bootstrap above.
    from envinfo import require_release_build

    require_release_build()
    child = os.environ.get("BENCH_CHILD")
    raise SystemExit(_child(child) if child else main())
