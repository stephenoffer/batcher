"""What a finding is, the thresholds rules compare against, and prose helpers.

Split out so a rule module imports one thing and every threshold has exactly one home — a
constant duplicated between two rule families is how two findings start disagreeing about
what "slow" means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Insight", "count", "gib"]

#: Actual/estimated rows beyond which Kyber planned for the wrong query. 10x is the figure
#: `OpProfile.est_error` documents as the adaptive controller's own "badly wrong" mark.
_EST_ERROR_HIGH = 10.0
_EST_ERROR_LOW = 0.1
#: Share of total runtime that makes one operator *the* bottleneck worth naming.
_BOTTLENECK_SHARE = 0.6
#: CPU utilization below which a run is leaving the box idle (the docs' target is >90%).
_CPU_IDLE = 0.4
#: Peak memory as a share of budget above which the next run of this shape may not fit.
#: Set to `MemoryConfig.hard_limit`, the point where the engine itself starts spilling —
#: **not** to the 80% utilization target. Warning at 80% would flag the healthy 80–90% band
#: the engine is deliberately aiming for, training users to ignore the one reading that
#: matters. High memory use is the goal; running out of it is the finding.
_MEMORY_TIGHT = 0.9
#: Peak memory as a share of budget below which the envelope went largely unclaimed. Well
#: under the >80% target so a merely-imperfect run stays silent and only real waste speaks.
_MEMORY_IDLE = 0.5
#: Run-queue length per core above which co-tenants, not the plan, explain an idle-looking
#: run. Set just above 1.0 (exact saturation) so a busy-but-not-oversubscribed box is silent.
_LOAD_CONTENDED = 1.25
#: Share of CFS periods throttled above which the cgroup quota is the binding constraint.
_THROTTLED_HIGH = 0.05
#: A run shorter than this is dominated by fixed control-plane cost; tuning it is noise.
_TRIVIAL_MS = 25.0
#: A scan smaller than this is not worth a pushdown conversation whatever its selectivity.
_WIDE_SCAN_MIN_ROWS = 100_000
#: Scanned:kept ratio at which a filter is selective enough that skipping files would pay.
_WIDE_SCAN_RATIO = 50
#: A join is "exploding" only well above noise, and only when output clearly exceeds input.
_EXPLODING_JOIN_MIN_ROWS = 1_000
_EXPLODING_JOIN_RATIO = 2.0
#: A filter is "late" when it discards most rows and costly work ran beneath it.
_LATE_FILTER_MIN_ROWS = 10_000
_LATE_FILTER_KEEP = 0.25
_LATE_FILTER_MIN_MS = 20.0
#: Time is a "long tail" when no step owns much of it, across enough steps to matter.
_LONG_TAIL_MIN_MS = 50.0
_LONG_TAIL_MIN_OPS = 6
_LONG_TAIL_MAX_SHARE = 0.30
#: Planning dominates when the steps account for little of the wall clock.
_PLANNING_MIN_MS = 5.0
_PLANNING_SHARE = 0.6


@dataclass(frozen=True, slots=True)
class Insight:
    """One finding: what was observed, what it means, and what to do about it.

    `severity` is ``"critical"`` | ``"warning"`` | ``"info"`` — the same three the dashboard
    colors by, and deliberately not a numeric score, because a score invites sorting by a
    precision these heuristics do not have.
    """

    severity: str
    #: A short kebab-case rule id, stable across releases so advice can be suppressed.
    rule: str
    title: str
    #: The measured numbers the rule fired on, phrased so the user can verify them.
    evidence: str
    #: The concrete change to make.
    action: str
    #: The operator this concerns, or empty for a whole-query finding.
    op: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "rule": self.rule,
            "title": self.title,
            "evidence": self.evidence,
            "action": self.action,
            "op": self.op,
            "detail": self.detail,
        }


def count(value: Any) -> str:
    """A compact SI-style row count for prose, tolerant of None."""
    if value is None:
        return "an unknown number of"
    number = float(value)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(number) >= limit:
            return f"{number / limit:.1f}{suffix}"
    return f"{number:.0f}"


def gib(value: int) -> str:
    """A compact binary size, e.g. ``1.4 GiB``."""
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size:.0f} B"
        size /= 1024
    return f"{size:.1f} GiB"
