"""The concurrency harness's aggregation, checked against hand-computed answers.

These are worth more than their size suggests. Twice in this repo a *harness* bug produced
a confident wrong conclusion — Daft was run single-process in the distributed lineup, and
the correctness-mismatch line printed engine names and values in opposite order for
months. A number that comes out of `summarize` is going to be quoted in a document, so
the arithmetic behind it is pinned here rather than trusted.

Every expected value below is computed by hand in the assertion's comment, not by running
the code and pasting what it said.
"""

from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(_REPO, "benchmarks"))

from concurrency.stats import (  # noqa: E402  (needs the sys.path above)
    ClientStats,
    fairness,
    percentile,
    scaling_efficiency,
    steady_state,
    summarize,
)

pytestmark = pytest.mark.unit


def _client(client_id: int, latencies: list[float], *, requests: int | None = None) -> ClientStats:
    """A client that spent exactly 10 steady seconds producing `latencies`."""
    n = len(latencies) if requests is None else requests
    return ClientStats(
        client_id=client_id,
        requests=n,
        steady_requests=n,
        steady_seconds=10.0,
        latencies_ms=latencies,
        cold_ms=[],
        rss_peak_mb=100.0,
        errors=[],
    )


class TestPercentile:
    def test_single_sample_is_itself(self) -> None:
        assert percentile([7.0], 99) == 7.0

    def test_matches_numpy_linear_interpolation(self) -> None:
        # [1..10], p50 sits at position (10-1)*0.5 = 4.5, i.e. halfway between the 5th and
        # 6th values (5 and 6) -> 5.5. This is numpy.percentile's default definition.
        assert percentile([float(i) for i in range(1, 11)], 50) == pytest.approx(5.5)

    def test_endpoints_are_exact(self) -> None:
        values = [3.0, 1.0, 2.0]
        assert percentile(values, 0) == 1.0
        assert percentile(values, 100) == 3.0

    def test_tail_sees_the_outlier(self) -> None:
        # The whole reason percentiles replaced best-of-N: slow requests must move the
        # tail and must not move p50. With 2 slow requests in 100, p99's position is
        # 99*0.99 = 98.01, which lands inside the slow pair -> the full 1000 ms.
        values = [10.0] * 98 + [1000.0, 1000.0]
        assert percentile(values, 50) == pytest.approx(10.0)
        assert percentile(values, 99) == pytest.approx(1000.0)

    def test_a_lone_outlier_is_interpolated_not_reported_whole(self) -> None:
        # Worth pinning because it surprises people: with exactly ONE slow request in 100,
        # p99 sits at position 98.01 and interpolates 99% of the way from 10 to 1000 ->
        # 19.9, not 1000. That is correct for this percentile definition, and it is why
        # `max_ms` is reported alongside: p99 alone can under-report a single stall.
        values = [10.0] * 99 + [1000.0]
        assert percentile(values, 99) == pytest.approx(19.9)
        assert max(values) == 1000.0

    def test_empty_raises_rather_than_returning_zero(self) -> None:
        # A zero here would read as "instant", which is the opposite of "no data".
        with pytest.raises(ValueError, match="empty"):
            percentile([], 50)


class TestSteadyState:
    def test_splits_on_wall_time_not_request_count(self) -> None:
        # Requests at t = 0, 1, ..., 9 s; warm-up 2 s, cooldown 1 s. The run's last start
        # is 9.0, so the window is [2.0, 8.0]: starts 2..8 inclusive -> 7 requests.
        starts = [float(i) for i in range(10)]
        latencies = [float(i * 10) for i in range(10)]
        warm, steady = steady_state(starts, latencies, warmup_s=2.0, cooldown_s=1.0)
        assert warm == [0.0, 10.0]  # t = 0, 1
        assert steady == [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]  # t = 2..8

    def test_a_run_shorter_than_its_warmup_has_no_steady_state(self) -> None:
        # Must report an empty steady state rather than silently reporting the warm-up,
        # which would publish cold-start numbers as though they were throughput.
        starts = [0.0, 0.5]
        warm, steady = steady_state(starts, [1.0, 2.0], warmup_s=5.0, cooldown_s=1.0)
        assert warm == [1.0, 2.0]
        assert steady == []

    def test_empty_input(self) -> None:
        assert steady_state([], [], warmup_s=1.0, cooldown_s=1.0) == ([], [])


class TestFairness:
    def test_equal_clients_are_perfectly_fair(self) -> None:
        clients = [_client(i, [1.0] * 100) for i in range(4)]
        assert fairness(clients) == pytest.approx(1.0)

    def test_a_starved_client_shows_up(self) -> None:
        # 10 requests against 100 in the same 10 s window -> 1.0 qps vs 10.0 qps.
        clients = [_client(0, [1.0] * 100), _client(1, [1.0] * 10)]
        assert fairness(clients) == pytest.approx(0.1)

    def test_no_completed_work_is_zero_not_a_crash(self) -> None:
        assert fairness([]) == 0.0
        assert fairness([_client(0, [], requests=0)]) == 0.0


class TestScalingEfficiency:
    def test_linear_scaling_is_one(self) -> None:
        assert scaling_efficiency(400.0, 100.0, 4) == pytest.approx(1.0)

    def test_the_inversion_this_harness_exists_to_catch(self) -> None:
        # The measurement from BENCHMARK_RESULTS.md: 16 clients produced 88 QPS where 1
        # produced 124. Efficiency 88/(16*124) = 0.0444 — and, critically, that is below
        # 1/16 = 0.0625, which is the threshold meaning the extra clients did not merely
        # fail to help, they destroyed throughput.
        efficiency = scaling_efficiency(88.0, 124.0, 16)
        assert efficiency == pytest.approx(0.04435, abs=1e-4)
        assert efficiency < 1.0 / 16

    def test_degenerate_baselines_are_zero(self) -> None:
        assert scaling_efficiency(100.0, 0.0, 4) == 0.0
        assert scaling_efficiency(100.0, 10.0, 0) == 0.0


class TestSummarize:
    def test_pools_latencies_across_clients(self) -> None:
        # Pooling matters: client A is uniformly fast and client B uniformly slow. The
        # pooled p99 must see B's tail. An average of the two clients' p99s would not.
        clients = [_client(0, [1.0] * 99), _client(1, [500.0] * 99)]
        point = summarize(
            clients,
            case="c",
            engine="batcher",
            mode="thread",
            shape="repeated",
            loop="closed",
            baseline_qps=None,
        )
        assert point.max_ms == 500.0
        assert point.p99_ms > 100.0
        assert point.qps == pytest.approx(19.8)  # (99 + 99) / 10 s
        assert point.clients == 2

    def test_baseline_none_means_this_is_the_baseline(self) -> None:
        point = summarize(
            [_client(0, [1.0] * 10)],
            case="c",
            engine="batcher",
            mode="thread",
            shape="repeated",
            loop="closed",
            baseline_qps=None,
        )
        assert point.scaling_efficiency == 1.0

    def test_a_dead_cell_reports_zeros_and_says_why(self) -> None:
        # One failed cell must not discard the rest of the sweep, but it must never be
        # mistaken for a fast one either.
        point = summarize(
            [_client(0, [], requests=0)],
            case="c",
            engine="batcher",
            mode="thread",
            shape="repeated",
            loop="closed",
            baseline_qps=100.0,
        )
        assert point.qps == 0.0
        assert point.p99_ms == 0.0
        assert point.errors == ["no steady-state requests completed"]

    def test_axes_survive_into_the_record(self) -> None:
        # A QPS number without its four axes is meaningless, so as_dict() must carry them.
        point = summarize(
            [_client(0, [2.0] * 50)],
            case="q6",
            engine="batcher",
            mode="process",
            shape="rotating",
            loop="open",
            baseline_qps=None,
        )
        doc = point.as_dict()
        assert doc["mode"] == "process"
        assert doc["shape"] == "rotating"
        assert doc["loop"] == "open"
        assert doc["clients"] == 1
