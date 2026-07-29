"""One client: a request loop that checks every answer and records every latency.

The loop is shared by both client modes. In ``--clients-as thread`` the driver calls
`run_client` directly on a thread against a shared engine session; in
``--clients-as process`` a subprocess builds its own session and calls the same function.
Keeping one loop is what makes the two modes comparable, and the distinction between them
is not cosmetic: the Reyden measurements in ``BENCHMARK_RESULTS.md`` found processes
reaching 113 QPS where threads reached 88, so which one you ran decides the answer.

Two properties of this loop are load-bearing and easy to get wrong:

**Every response is verified, not just the first.** A wrong answer that appears only at
16-way concurrency is precisely the failure a throughput number hides, so each request
checks its result against the oracle fingerprint taken before the run. Verification is a
fixed-size comparison (`signature.result_signature`) rather than the full multiset gate,
because it runs thousands of times.

**Open-loop latency is measured from the intended arrival, not the dispatch.** A closed
loop only issues request k+1 after k returns, so when the engine slows down the client
politely stops asking — and the recorded p99 improves while the real one collapses. That
is coordinated omission, and it is why any tail-latency claim has to come from
``--rate``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrency.stats import ClientStats, steady_state
from signature import result_signature, signatures_match

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["ClientConfig", "Request", "run_client"]

#: One thing a client can ask for: a case name, the callable that runs it, and the
#: fingerprint its answer must match.
Request = tuple[str, Callable[[], "pa.Table"], list]


@dataclass(frozen=True)
class ClientConfig:
    """How one client should drive the engine."""

    client_id: int
    duration_s: float
    warmup_s: float
    cooldown_s: float
    #: Arrivals per second for this client, or None for a closed loop.
    rate: float | None = None
    #: Seeds the arrival process. Varied per client so sixteen clients do not fire in
    #: lockstep, which would measure a thundering herd rather than a Poisson load.
    seed: int = 0
    #: Stop after this many consecutive failures rather than filling a log with the same
    #: error for the whole duration.
    max_consecutive_errors: int = 5


def _rss_peak_mb() -> float:
    """Peak resident set size of this process, in MiB.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS; the heuristic below picks the
    interpretation that is not absurd. In thread mode every client reports the same
    process-wide figure, which is correct — there is only one process to attribute.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1024.0 if sys.platform != "darwin" else raw / (1024.0 * 1024.0)


def run_client(config: ClientConfig, requests: list[Request], start_barrier: float) -> ClientStats:
    """Drive `requests` for `config.duration_s` and return this client's samples.

    The client cycles through `requests`, so the driver controls the query shape purely by
    what it passes: one entry is ``--shape repeated`` (plan-cache hits, the serving shape),
    many entries is ``--shape rotating`` (a first-seen shape per request, which pays the
    5-8 ms cold optimizer cost the plan cache otherwise hides).

    Args:
        config: This client's loop parameters.
        requests: The cycle of case name, callable, and expected fingerprint.
        start_barrier: An absolute `time.perf_counter` value every client waits for, so
            all of them begin measuring at the same instant.

    Returns:
        This client's latencies, error strings, and steady-state accounting.
    """
    rng = random.Random(config.seed + config.client_id)
    starts: list[float] = []
    latencies: list[float] = []
    errors: list[str] = []
    consecutive = 0
    index = 0

    while time.perf_counter() < start_barrier:
        time.sleep(0.0005)
    origin = start_barrier
    deadline = origin + config.duration_s
    # In an open loop the schedule advances independently of how long requests take, which
    # is the entire point: falling behind it *is* the queueing delay being measured.
    next_arrival = origin

    while True:
        if config.rate is not None:
            next_arrival += rng.expovariate(config.rate)
            if next_arrival >= deadline:
                break
            now = time.perf_counter()
            if next_arrival > now:
                time.sleep(next_arrival - now)
            issued_at = next_arrival
        else:
            issued_at = time.perf_counter()
            if issued_at >= deadline:
                break

        name, fn, expected = requests[index % len(requests)]
        index += 1
        try:
            table = fn()
            elapsed_ms = (time.perf_counter() - issued_at) * 1000.0
            ok, why = signatures_match(expected, result_signature(table))
            if not ok:
                # A concurrency-only wrong answer. Record it and keep going: how *often*
                # it happens is itself the finding.
                errors.append(f"{name}: WRONG ANSWER: {why}")
            starts.append(issued_at - origin)
            latencies.append(elapsed_ms)
            consecutive = 0
        except Exception as exc:
            consecutive += 1
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            if consecutive >= config.max_consecutive_errors:
                break

    warm, steady = steady_state(
        starts, latencies, warmup_s=config.warmup_s, cooldown_s=config.cooldown_s
    )
    steady_seconds = max(0.0, config.duration_s - config.warmup_s - config.cooldown_s)
    return ClientStats(
        client_id=config.client_id,
        requests=len(latencies),
        steady_requests=len(steady),
        steady_seconds=steady_seconds,
        latencies_ms=steady,
        cold_ms=warm,
        rss_peak_mb=_rss_peak_mb(),
        errors=errors,
    )


# --------------------------------------------------------------------------- #
# Subprocess entry point (--clients-as process)
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="one isolated concurrency-benchmark client")
    p.add_argument("--engine", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--source", default=None)
    p.add_argument("--cases", required=True, help="comma-separated case names to cycle")
    p.add_argument("--client-id", type=int, required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--warmup", type=float, required=True)
    p.add_argument("--cooldown", type=float, required=True)
    p.add_argument("--rate", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-in", type=float, default=1.0, help="seconds until the shared start")
    p.add_argument("--signatures", required=True, help="JSON: case name -> expected fingerprint")
    return p.parse_args()


def main() -> int:
    """Build an isolated session, run the loop, and print one JSON line of samples."""
    args = _parse_args()
    out: dict[str, object] = {"client_id": args.client_id, "error": None}
    try:
        import engines as engines_mod
        import suites  # noqa: F401  (import registers every benchmark)
        from context import Context
        from registry import REGISTRY

        engines = engines_mod.resolve([args.engine])
        ctx = Context.build(args.benchmark, args.scale, engines, args.source)
        signatures = json.loads(args.signatures)
        wanted = args.cases.split(",")
        requests: list[Request] = []
        for name in wanted:
            case = REGISTRY.select(dataset=args.benchmark, name=name)[0]
            fn = case.build(ctx)[args.engine]
            if fn is None:
                continue
            requests.append((name, fn, signatures[name]))
        if not requests:
            raise RuntimeError(f"engine {args.engine!r} expresses none of {wanted}")

        config = ClientConfig(
            client_id=args.client_id,
            duration_s=args.duration,
            warmup_s=args.warmup,
            cooldown_s=args.cooldown,
            rate=args.rate,
            seed=args.seed,
        )
        stats = run_client(config, requests, time.perf_counter() + args.start_in)
        out["stats"] = {
            "client_id": stats.client_id,
            "requests": stats.requests,
            "steady_requests": stats.steady_requests,
            "steady_seconds": stats.steady_seconds,
            "latencies_ms": stats.latencies_ms,
            "cold_ms": stats.cold_ms,
            "rss_peak_mb": stats.rss_peak_mb,
            "errors": stats.errors[:20],
        }
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
