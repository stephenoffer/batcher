"""The `cpu-underutilized` insight reads the whole-run CPU, not the per-operator ratio.

`QueryUsage.cores_busy` is `cpu_ms / wall_ms` for the entire execution, and its own docstring
calls it "the one figure a per-operator utilization ratio cannot be summed into". The rule
summed exactly that -- and the per-operator hardware fields hold **only on a materializing
executor**, while the streaming executor is the default. There every operator reports
`cpu_ms == elapsed_ms` (one thread's accounting) beside a `threads` count for the whole pool,
so the derived `cpu_util` is ~`1 / threads` no matter what the query did.

Measured before the fix: a 10M-row `GROUP BY` whose process burned 177 ms of CPU in 14.3 ms of
wall clock -- twelve cores -- was reported as "CPU 2% utilized".
"""

from __future__ import annotations

from batcher.observe.insights.resources import idle_cpu


def _streaming_ops(threads: int = 61, elapsed: float = 30.0):
    """Operators as the *streaming* executor reports them: one thread's CPU, whole-pool count."""
    return [
        {
            "kind": "aggregate",
            "elapsed_ms": elapsed,
            "cpu_ms": elapsed,
            "threads": threads,
            "cpu_util": 1.0 / threads,
            "measured": True,
        },
        {
            "kind": "scan",
            "elapsed_ms": 0.05,
            "cpu_ms": 0.05,
            "threads": threads,
            "cpu_util": 1.0 / threads,
            "measured": True,
        },
    ]


def _profile(cores_busy: float, wall_ms: float = 30.0):
    return {
        "usage": {"cores_busy": cores_busy, "cpu_ms": cores_busy * wall_ms, "wall_ms": wall_ms},
        "total_ms": wall_ms,
    }


def _fired(found):
    return [i for i in found if i.rule == "cpu-underutilized"]


def test_a_saturated_streaming_run_is_not_called_idle():
    """The regression. Per-operator ratio says 1.6%; the run actually used most of the box."""
    from batcher._internal.hardware import available_cpu_count

    busy = available_cpu_count() * 0.85
    found = idle_cpu(_profile(busy), _streaming_ops(), 30.0)
    assert not _fired(found), (
        "a run using 85% of the machine was reported idle -- the rule is reading the "
        "per-operator ratio, which the streaming executor does not populate"
    )


def test_a_genuinely_serial_run_is_still_called_idle():
    """The positive control: the finding must survive for the case it exists to catch."""
    found = idle_cpu(_profile(1.0), _streaming_ops(), 30.0)
    assert _fired(found), "a one-core run over 30 ms should still be reported"


def test_the_reported_number_is_the_whole_run_share():
    from batcher._internal.hardware import available_cpu_count

    cores = available_cpu_count()
    found = _fired(idle_cpu(_profile(cores * 0.10), _streaming_ops(), 30.0))
    assert found and "10%" in found[0].title, f"expected the true share, got {found[0].title!r}"


def test_it_falls_back_to_the_per_operator_ratio_when_usage_is_unmeasured():
    """A materializing run with no whole-run reading keeps the old, sound path."""
    ops = [
        {
            "kind": "aggregate",
            "elapsed_ms": 30.0,
            "cpu_ms": 3.0,
            "threads": 8,
            "cpu_util": 0.05,
            "measured": True,
        }
    ]
    assert _fired(idle_cpu({"total_ms": 30.0}, ops, 30.0)), (
        "with no usage block the per-operator ratio is all there is and must still be used"
    )


def test_a_trivial_run_reports_nothing():
    assert not idle_cpu(_profile(1.0), _streaming_ops(), 1.0)
