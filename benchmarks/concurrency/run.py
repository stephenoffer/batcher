"""Measure what happens when many queries run at once — the dimension nothing else here covers.

Every other suite in ``benchmarks/`` answers "how fast is one query". This one answers
"how much work does the engine get through, and how badly does the tail suffer, as clients
are added". The reason it exists is a single measurement in ``BENCHMARK_RESULTS.md``:
16 threads produced *less* throughput than 1 (88 QPS against 124) while p50 went from
7.6 ms to 178 ms. That is the most severe result in the file and it had no harness, so it
could neither be tracked nor shown to improve.

Run::

    python3 benchmarks/concurrency/run.py --sanity           # prove the harness scales
    python3 benchmarks/concurrency/run.py --clients 1,2,4,8,16
    python3 benchmarks/concurrency/run.py --clients-as process --shape rotating
    python3 benchmarks/concurrency/run.py --rate 200 --clients 8   # open loop, real p99

**A number from here is meaningless without its four axes**, so all four are recorded in
every row and printed in every header:

- ``--clients-as thread|process`` — one session shared, or one per client.
- ``--rate N`` — open-loop Poisson arrivals. Only this mode's p99 means anything under
  saturation; a closed loop stops asking when the engine slows down.
- ``--shape repeated|rotating`` — one query text (plan-cache hits, ~0.12 ms of planning)
  or a rotation (a first-seen shape each time, 5-8 ms of planning). These measure
  different systems.
- The machine fingerprint, written as the first record of the result document.

Correctness comes first, as everywhere else here: `harness.compare` gates each case before
anything is timed, and then *every client checks every response* against the oracle
fingerprint. A wrong answer that only appears under concurrency is the failure a
throughput number would otherwise hide.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrency.client import ClientConfig, Request, run_client
from concurrency.stats import ClientStats, SweepPoint, summarize
from envinfo import machine_fingerprint, require_quiet_box, require_release_build
from signature import result_signature

# `suites`, `engines`, `context`, `harness`, and `registry` are imported lazily inside the
# functions that need them. They pull in duckdb, polars, and the corpus loaders, none of
# which `--sanity` requires — and `--sanity` is the one mode that has to run in CI, on a
# box with no comparator installed and no TPC-H data.

#: Seconds every client idles before the shared start, so process clients have finished
#: importing and building their session before any of them begins measuring.
_START_DELAY_S = 3.0

#: The floor `--sanity` demands of the harness itself. A pure-sleep workload releases the
#: GIL, so the driver should reach near-linear scaling; anything well below this means the
#: *measurement* does not scale and no engine number taken with it is meaningful.
_SANITY_MIN_EFFICIENCY = 0.7


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batcher concurrency / QPS benchmark")
    p.add_argument("--benchmark", default="tpch", help="dataset the cases come from")
    p.add_argument("--engines", default="batcher,duckdb", help="comma-separated lineup")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--source", default=None, help="override the dataset's parquet base URI")
    p.add_argument("--family", default=None)
    p.add_argument("--only", default=None, help="run only cases whose name contains this")
    p.add_argument("--clients", default="1,2,4,8,16", help="comma-separated client counts")
    p.add_argument("--clients-as", choices=("thread", "process"), default="thread")
    p.add_argument("--shape", choices=("repeated", "rotating"), default="repeated")
    p.add_argument("--rate", type=float, default=None, help="open-loop arrivals/sec per client")
    p.add_argument("--duration", type=float, default=20.0, help="seconds per sweep point")
    p.add_argument("--warmup", type=float, default=5.0)
    p.add_argument("--cooldown", type=float, default=1.0)
    p.add_argument("--out", default=None, help="write the result document here as JSON")
    p.add_argument("--sanity", action="store_true", help="self-test the harness and exit")
    p.add_argument("--allow-debug-build", action="store_true")
    p.add_argument("--allow-busy-box", action="store_true")
    return p.parse_args()


def _assert_no_ray() -> None:
    """Refuse to run with Ray up: it silently changes what is being measured.

    Merely importing Daft once made an in-memory Batcher query 23x slower, because
    ``distributed="auto"`` distributes already-resident data the moment anything has
    initialized Ray. A concurrency number taken in that state measures the scheduler.
    """
    if os.environ.get("BENCH_BATCHER_DISTRIBUTED") == "1":
        raise SystemExit("BENCH_BATCHER_DISTRIBUTED=1 is set; this harness measures in-process.")
    ray = sys.modules.get("ray")
    if ray is not None and ray.is_initialized():
        raise SystemExit("Ray is already initialized; this harness measures the in-process path.")


def _gate_and_fingerprint(ctx, case_names: list[str], engines: list[str], benchmark: str) -> dict:
    """Run the correctness gate, then take the oracle fingerprint for each case.

    Nothing is timed until every engine in the lineup agrees on the answer — the same
    contract `run.py` enforces. The fingerprint that comes out is what every client then
    checks each of its own responses against.

    Args:
        ctx: The prepared data context.
        case_names: Cases to gate.
        engines: The resolved engine lineup.
        benchmark: The dataset the cases belong to.

    Returns:
        Case name mapped to the expected result fingerprint.

    Raises:
        SystemExit: If any case fails the correctness gate.
    """
    from harness import compare
    from registry import REGISTRY

    signatures: dict[str, list] = {}
    for name in case_names:
        case = REGISTRY.select(dataset=benchmark, name=name)[0]
        fns = case.build(ctx)
        print(f"gating {name} ...", flush=True)
        result = compare(name, fns, engines, runs=1)
        if result.status in ("FAILED", "ERROR"):
            raise SystemExit(f"correctness gate failed for {name}: {result.note}")
        oracle = next((e for e in ("duckdb", "polars") if fns.get(e)), engines[0])
        fn = fns[oracle]
        assert fn is not None  # the gate above proved it produced a result
        signatures[name] = result_signature(fn())
    return signatures


def _run_threads(
    requests: list[Request], n: int, config_for: Callable[[int], ClientConfig]
) -> list[ClientStats]:
    """Run `n` clients as threads sharing one engine session."""
    barrier = time.perf_counter() + _START_DELAY_S
    results: list[ClientStats | None] = [None] * n
    threads = [
        threading.Thread(
            target=lambda i=i: results.__setitem__(i, run_client(config_for(i), requests, barrier)),
            name=f"bench-client-{i}",
            daemon=True,
        )
        for i in range(n)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [r for r in results if r is not None]


def _run_processes(args: argparse.Namespace, engine: str, cases: list[str], sigs: dict, n: int):
    """Run `n` clients as subprocesses, each with its own engine session."""
    here = Path(__file__).parent
    payload = json.dumps({k: sigs[k] for k in cases})
    procs = []
    for i in range(n):
        cmd = [
            sys.executable,
            str(here / "client.py"),
            "--engine", engine,
            "--benchmark", args.benchmark,
            "--scale", str(args.scale),
            "--cases", ",".join(cases),
            "--client-id", str(i),
            "--duration", str(args.duration),
            "--warmup", str(args.warmup),
            "--cooldown", str(args.cooldown),
            "--seed", "1",
            "--start-in", str(_START_DELAY_S + 10.0),  # room for N interpreters to boot
            "--signatures", payload,
        ]  # fmt: skip
        if args.source:
            cmd += ["--source", args.source]
        if args.rate is not None:
            cmd += ["--rate", str(args.rate)]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True))

    out: list[ClientStats] = []
    for proc in procs:
        stdout, _ = proc.communicate(timeout=args.duration + 180)
        line = (stdout or "").strip().splitlines()[-1:]
        if not line:
            continue
        doc = json.loads(line[0])
        if doc.get("error") or "stats" not in doc:
            print(f"  client {doc.get('client_id')} failed: {doc.get('error')}")
            continue
        out.append(ClientStats(**doc["stats"]))
    return out


def _sweep(args: argparse.Namespace, ctx, engine: str, cases: list[str], sigs: dict):
    """Sweep the client counts for one engine, returning a sweep point per count."""
    from registry import REGISTRY

    counts = [int(c) for c in args.clients.split(",")]
    requests: list[Request] = []
    if ctx is not None:
        for name in cases:
            case = REGISTRY.select(dataset=args.benchmark, name=name)[0]
            fn = case.build(ctx).get(engine)
            if fn is not None:
                requests.append((name, fn, sigs[name]))
        if not requests:
            return []

    label = f"{'+'.join(cases)}" if args.shape == "rotating" else cases[0]
    points: list[SweepPoint] = []
    baseline: float | None = None
    for n in counts:
        print(f"  {engine} x{n} ...", flush=True)

        def config_for(i: int) -> ClientConfig:
            return ClientConfig(
                client_id=i,
                duration_s=args.duration,
                warmup_s=args.warmup,
                cooldown_s=args.cooldown,
                rate=args.rate,
                seed=1,
            )

        if args.clients_as == "thread":
            stats = _run_threads(requests, n, config_for)
        else:
            stats = _run_processes(args, engine, cases, sigs, n)
        point = summarize(
            stats,
            case=label,
            engine=engine,
            mode=args.clients_as,
            shape=args.shape,
            loop="open" if args.rate is not None else "closed",
            baseline_qps=baseline,
        )
        if n == 1:
            baseline = point.qps
        points.append(point)
    return points


def _print_table(points: list[SweepPoint]) -> None:
    """Print the sweep, one row per (engine, client count)."""
    headers = ["engine", "clients", "qps", "scale_eff", "fair", "p50", "p99", "p999", "max", "cold"]
    rows = [
        [
            p.engine,
            str(p.clients),
            f"{p.qps:.1f}",
            f"{p.scaling_efficiency:.2f}",
            f"{p.fairness:.2f}",
            f"{p.p50_ms:.1f}",
            f"{p.p99_ms:.1f}",
            f"{p.p999_ms:.1f}",
            f"{p.max_ms:.1f}",
            f"{p.cold_ms:.1f}",
        ]
        for p in points
    ]
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(c)) for w, c in zip(widths, row, strict=True)]

    def fmt(cells: list[str]) -> str:
        pairs = enumerate(zip(cells, widths, strict=True))
        return "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in pairs)

    print(fmt(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        print(fmt(row))
    for p in points:
        if p.errors:
            print(f"\n[{p.engine} x{p.clients}] {len(p.errors)} error(s), first: {p.errors[0]}")


def _sanity(args: argparse.Namespace) -> int:
    """Prove the harness itself scales before trusting it about anything else.

    Runs a workload that only sleeps, which releases the GIL, so N clients should reach
    near-linear throughput. If this fails, the driver is the bottleneck and every engine
    number it would produce is a measurement of the driver.
    """
    import pyarrow as pa

    table = pa.table({"x": [1]})
    sig = result_signature(table)

    def sleeper() -> pa.Table:
        time.sleep(0.01)
        return table

    requests: list[Request] = [("sleep-10ms", sleeper, sig)]
    args.duration, args.warmup, args.cooldown = 4.0, 1.0, 0.5
    points = []
    baseline = None

    def config_for(i: int) -> ClientConfig:
        return ClientConfig(i, args.duration, args.warmup, args.cooldown, args.rate, 1)

    for n in (1, 4):
        stats = _run_threads(requests, n, config_for)
        point = summarize(
            stats,
            case="sanity",
            engine="sleep",
            mode="thread",
            shape="repeated",
            loop="closed",
            baseline_qps=baseline,
        )
        if n == 1:
            baseline = point.qps
        points.append(point)
    _print_table(points)
    efficiency = points[-1].scaling_efficiency
    if efficiency < _SANITY_MIN_EFFICIENCY:
        print(
            f"\nFAIL: the harness itself only reached {efficiency:.2f} scaling efficiency at "
            f"4 clients (floor {_SANITY_MIN_EFFICIENCY}). Any engine number taken with this "
            "driver would measure the driver."
        )
        return 1
    print(f"\nOK: harness scales at {efficiency:.2f} efficiency on a pure-sleep workload.")
    return 0


def main() -> int:
    args = _parse_args()
    require_release_build(allow_debug=args.allow_debug_build)
    if args.sanity:
        return _sanity(args)

    _assert_no_ray()
    require_quiet_box(allow_busy=args.allow_busy_box)

    import engines as engines_mod
    import suites  # noqa: F401  (import registers every benchmark)
    from context import Context
    from registry import REGISTRY

    names = [n.strip() for n in args.engines.split(",")]
    engines = engines_mod.resolve(names)
    ctx = Context.build(args.benchmark, args.scale, engines, args.source)
    selected = REGISTRY.select(dataset=args.benchmark, family=args.family, name=args.only)
    if not selected:
        print("no benchmarks matched the selection.")
        return 0
    cases = [c.name for c in selected] if args.shape == "rotating" else [selected[0].name]

    fp = machine_fingerprint()
    loop = "open" if args.rate else "closed"
    print(f"concurrency benchmark  ({fp['engine_profile']} engine, {fp['git_sha']})")
    print(f"  {fp['cpu_count_available']} cores available of {fp['cpu_count_logical']}")
    print(f"  mode={args.clients_as} shape={args.shape} loop={loop}")
    print(f"  cases: {', '.join(cases)}\n")

    signatures = _gate_and_fingerprint(ctx, cases, [e.name for e in engines], args.benchmark)
    print()

    points: list[SweepPoint] = []
    shared_ctx = ctx if args.clients_as == "thread" else None
    for engine in engines:
        points += _sweep(args, shared_ctx, engine.name, cases, signatures)
    print()
    _print_table(points)

    if args.out:
        document = {
            "machine": fp,
            "config": vars(args),
            "points": [p.as_dict() for p in points],
        }
        Path(args.out).write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.out}")

    wrong = [p for p in points if any("WRONG ANSWER" in e for e in p.errors)]
    if wrong:
        print(f"\n{len(wrong)} sweep point(s) produced a WRONG ANSWER under concurrency.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
