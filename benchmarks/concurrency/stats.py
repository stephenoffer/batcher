"""Turning a pile of request latencies into the four numbers that mean something.

Deliberately *not* best-of-N. `harness.bench` reports the fastest of five runs, which is
the right statistic for "how fast can this query go" and the wrong one for "what happens
when sixteen of them run at once": it reports the luckiest request and erases the tail that
is the entire subject. Everything here is computed over the full steady-state distribution.

The four numbers, and why each is here:

- **percentiles** — p50 says whether the median user is served; p99 says whether anyone is.
  Under saturation those diverge by orders of magnitude, which a mean hides completely.
- **scaling efficiency** — `qps(n) / (n * qps(1))`. The one number that would have caught
  the finding in ``BENCHMARK_RESULTS.md`` where 16 threads produced *less* throughput than
  1 (88 vs 124 QPS). Perfect scaling is 1.0; anything below 1/n means adding clients made
  things actively worse.
- **fairness** — `min_client_qps / max_client_qps`. Aggregate throughput can look healthy
  while one client starves. This is also the only multi-tenancy signal the benchmark can
  currently produce.
- **steady state** — a run's first requests pay warm-up (~2-3.6 s of control plane lands on
  whichever shape runs first) and its last requests run against a draining box. Both are
  excluded from the percentiles and reported separately, rather than averaged in.

Every function here is pure so it can be unit-tested against hand-computed answers, which
matters more than usual: a bug in the *harness* has twice produced a confident wrong
conclusion in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ClientStats",
    "SweepPoint",
    "fairness",
    "percentile",
    "scaling_efficiency",
    "steady_state",
    "summarize",
]


def percentile(values: list[float], q: float) -> float:
    """The `q`-th percentile of `values` by linear interpolation, `q` in [0, 100].

    Uses the same definition as ``numpy.percentile``'s default so a reader who checks the
    number against numpy gets the same answer. Interpolating rather than picking the
    nearest rank matters at p99 on a few hundred samples, where nearest-rank quantizes the
    answer to whichever single request happened to land there.

    Args:
        values: Samples, in any order. Must be non-empty.
        q: The percentile to compute, from 0 to 100.

    Returns:
        The interpolated percentile.

    Raises:
        ValueError: If `values` is empty.
    """
    if not values:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def steady_state(
    starts: list[float],
    latencies: list[float],
    *,
    warmup_s: float,
    cooldown_s: float,
) -> tuple[list[float], list[float]]:
    """Split requests into the warm-up prefix and the steady-state middle.

    The window is defined in *wall time relative to the run*, not in request count,
    because a slow client issues fewer requests in the same warm-up period and a
    count-based cut would trim it far harder than a fast one — biasing exactly the client
    whose behavior the fairness number is about.

    Args:
        starts: Each request's start time, in seconds since the run began.
        latencies: Each request's duration in milliseconds, index-aligned with `starts`.
        warmup_s: Seconds at the start of the run to exclude.
        cooldown_s: Seconds at the end of the run to exclude.

    Returns:
        ``(warmup_latencies, steady_latencies)``. Either may be empty — a run too short to
        have a steady state must say so rather than silently report its warm-up.
    """
    if not starts:
        return [], []
    end = max(starts)
    warm: list[float] = []
    steady: list[float] = []
    for start, latency in zip(starts, latencies, strict=True):
        if start < warmup_s:
            warm.append(latency)
        elif start <= end - cooldown_s:
            steady.append(latency)
    return warm, steady


@dataclass(frozen=True)
class ClientStats:
    """One client's contribution to a sweep point."""

    client_id: int
    requests: int
    steady_requests: int
    steady_seconds: float
    latencies_ms: list[float]
    cold_ms: list[float]
    rss_peak_mb: float
    errors: list[str]

    @property
    def qps(self) -> float:
        """Completed steady-state requests per second for this client alone."""
        if self.steady_seconds <= 0:
            return 0.0
        return self.steady_requests / self.steady_seconds


def fairness(clients: list[ClientStats]) -> float:
    """The ratio of the slowest client's throughput to the fastest client's.

    1.0 is perfect equality. A low value with healthy aggregate QPS is the signature of
    one client monopolizing the engine while another starves, which no aggregate number
    shows.

    Args:
        clients: Every client's stats for one sweep point.

    Returns:
        ``min_qps / max_qps``, or 0.0 when the fastest client completed nothing.
    """
    rates = [c.qps for c in clients]
    if not rates or max(rates) <= 0:
        return 0.0
    return min(rates) / max(rates)


def scaling_efficiency(qps_at_n: float, qps_at_1: float, n: int) -> float:
    """How much of the ideal `n`-fold speedup the run actually achieved.

    1.0 is linear scaling. **Below `1/n` means adding clients reduced total throughput** —
    the box is not merely failing to scale, the extra work is destructive. That is the
    regime `BENCHMARK_RESULTS.md` recorded at 16 threads, and the reason this number is
    tracked rather than derived on demand.

    Args:
        qps_at_n: Aggregate throughput with `n` clients.
        qps_at_1: Aggregate throughput with one client.
        n: The client count.

    Returns:
        The efficiency ratio, or 0.0 when the single-client baseline is unusable.
    """
    if qps_at_1 <= 0 or n <= 0:
        return 0.0
    return qps_at_n / (n * qps_at_1)


@dataclass(frozen=True)
class SweepPoint:
    """The aggregate result for one (case, client count) cell of the sweep."""

    case: str
    engine: str
    clients: int
    mode: str
    shape: str
    loop: str
    qps: float
    scaling_efficiency: float
    fairness: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    p999_ms: float
    max_ms: float
    cold_ms: float
    steady_requests: int
    rss_peak_mb: float
    errors: list[str]

    def as_dict(self) -> dict[str, object]:
        """A JSON-serializable view, for the result document."""
        return {
            "case": self.case,
            "engine": self.engine,
            "clients": self.clients,
            "mode": self.mode,
            "shape": self.shape,
            "loop": self.loop,
            "qps": round(self.qps, 3),
            "scaling_efficiency": round(self.scaling_efficiency, 4),
            "fairness": round(self.fairness, 4),
            "p50_ms": round(self.p50_ms, 3),
            "p90_ms": round(self.p90_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "p999_ms": round(self.p999_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "cold_ms": round(self.cold_ms, 3),
            "steady_requests": self.steady_requests,
            "rss_peak_mb": round(self.rss_peak_mb, 1),
            "errors": self.errors[:10],
        }


def summarize(
    clients: list[ClientStats],
    *,
    case: str,
    engine: str,
    mode: str,
    shape: str,
    loop: str,
    baseline_qps: float | None,
) -> SweepPoint:
    """Fold every client's samples into one sweep point.

    Latencies are pooled across clients before taking percentiles rather than averaging
    per-client percentiles: the pooled p99 is the experience of the unluckiest *request*,
    which is what a tail-latency budget is written against, while an average of p99s is
    not a percentile of anything.

    Args:
        clients: Every client's stats for this cell.
        case: The benchmark case name.
        engine: The engine under test.
        mode: ``thread`` or ``process``.
        shape: ``repeated`` or ``rotating``.
        loop: ``closed`` or ``open``.
        baseline_qps: Aggregate QPS at one client, or None when this *is* the baseline.

    Returns:
        The aggregated sweep point.
    """
    pooled = [ms for c in clients for ms in c.latencies_ms]
    cold = [ms for c in clients for ms in c.cold_ms]
    total_qps = sum(c.qps for c in clients)
    n = len(clients)
    errors = [e for c in clients for e in c.errors]
    if not pooled:
        # Every request failed or the run was too short to have a steady state. Report
        # zeros rather than crashing, so one bad cell does not discard the whole sweep.
        return SweepPoint(
            case=case,
            engine=engine,
            clients=n,
            mode=mode,
            shape=shape,
            loop=loop,
            qps=0.0,
            scaling_efficiency=0.0,
            fairness=0.0,
            p50_ms=0.0,
            p90_ms=0.0,
            p99_ms=0.0,
            p999_ms=0.0,
            max_ms=0.0,
            cold_ms=percentile(cold, 50) if cold else 0.0,
            steady_requests=0,
            rss_peak_mb=max((c.rss_peak_mb for c in clients), default=0.0),
            errors=errors or ["no steady-state requests completed"],
        )
    return SweepPoint(
        case=case,
        engine=engine,
        clients=n,
        mode=mode,
        shape=shape,
        loop=loop,
        qps=total_qps,
        scaling_efficiency=(
            1.0 if baseline_qps is None else scaling_efficiency(total_qps, baseline_qps, n)
        ),
        fairness=fairness(clients),
        p50_ms=percentile(pooled, 50),
        p90_ms=percentile(pooled, 90),
        p99_ms=percentile(pooled, 99),
        p999_ms=percentile(pooled, 99.9),
        max_ms=max(pooled),
        cold_ms=percentile(cold, 50) if cold else 0.0,
        steady_requests=sum(c.steady_requests for c in clients),
        rss_peak_mb=max(c.rss_peak_mb for c in clients),
        errors=errors,
    )
