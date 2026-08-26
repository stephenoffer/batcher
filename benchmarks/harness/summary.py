"""The suite geomean, its inputs, and how much precision it is entitled to.

Every headline figure this project quotes — "TPC-H sf1 0.922", "ClickBench 0.620" — is a
geometric mean of per-query ``batcher_ms / engine_ms`` ratios. Until now the harness printed
the per-query table and **nothing computed the geomean at all**: it was worked out by hand
from the printed rows, which leaves three things unrecorded and one of them is load-bearing.

*Which rows went in.* A hand-computed mean over a table containing ``FAILED``, ``PARTIAL``,
``DIVERGENT`` and ``n/c`` rows depends entirely on which the person included, and nothing
wrote that decision down. Two figures produced a month apart need not have the same
denominator, and a suite whose coverage changed can move the geomean without any engine
changing.

*How many samples.* ``_runs_for`` picks best-of-5, best-of-3 or best-of-2 by scale, so the
same suite at two scales is not the same measurement, and the figure carries no note of it.

*How repeatable it is.* This is the one that has already produced a false claim. The
operator-mix geomean measured three times on one box gave 0.619 / 0.598 / 0.594 — a **4.1%
spread**, with one case varying 1.78x — while the record said a run "reproduces its recorded
0.620 to within 0.001, which is the evidence that this box and the recorded board agree."
Agreement to 0.16% under a 4% spread is a coincidence, and the sentence was reading it as
corroboration. A figure quoted to three decimals claims a precision no single run can
support.

So this module computes the geomean *and* what it is made of, and :func:`format_summary`
prints them together. `run.py --repeat N` then re-runs a selection and reports the spread,
which is what makes "reproduces to within X" a checkable claim rather than a hopeful one.

Nothing here decides what is quotable. It records what was measured so a reader can.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .compare import CompareResult

__all__ = ["Summary", "format_repeats", "format_summary", "geomean", "summarize"]

#: Statuses whose ratios are excluded from the geomean, and why. A ``DIVERGENT`` row is
#: excluded for the same reason its ratio is withheld from the table: the two engines did
#: not compute the same answer, so dividing their times compares unlike work.
_EXCLUDED = {
    "FAILED": "results disagreed",
    "ERROR": "no engine produced a result",
    "KILLED": "the case took its process down",
    "DIVERGENT": "a recorded semantic difference — unlike work, so unlike times",
}


def geomean(values: list[float]) -> float:
    """The geometric mean of `values`, or NaN when there is nothing to average."""
    positive = [v for v in values if v > 0 and math.isfinite(v)]
    if not positive:
        return float("nan")
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


@dataclass
class Summary:
    """One comparator's geomean and the exact set of rows behind it."""

    engine: str
    value: float
    included: int
    excluded: dict[str, int] = field(default_factory=dict)

    def provenance(self) -> str:
        """A one-line account of what the number is a mean *of*."""
        if not self.excluded:
            return f"{self.included} of {self.included} cases"
        dropped = ", ".join(f"{n} {status}" for status, n in sorted(self.excluded.items()))
        return f"{self.included} of {self.included + sum(self.excluded.values())} cases ({dropped})"


def summarize(results: list[CompareResult], engines: list[str]) -> list[Summary]:
    """The geomean of ``batcher/<engine>`` per comparator, with its inputs counted.

    A case contributes only when both engines produced a timing *and* the row was not
    excluded by status. Every exclusion is counted rather than silently dropped, because a
    geomean over an unstated denominator is not comparable to another one.

    Args:
        results: Every case's comparison result.
        engines: The resolved lineup, in report order.

    Returns:
        One :class:`Summary` per comparator, in lineup order. Empty when Batcher is absent.
    """
    if "batcher" not in engines:
        return []
    out: list[Summary] = []
    for engine in (e for e in engines if e != "batcher"):
        ratios: list[float] = []
        excluded: dict[str, int] = {}
        for r in results:
            b, c = r.engines.get("batcher"), r.engines.get(engine)
            if b is None or c is None or not b.ms or not c.ms:
                continue
            if r.status in _EXCLUDED:
                excluded[r.status] = excluded.get(r.status, 0) + 1
                continue
            if b.correct is False or c.correct is False:
                excluded["FAILED"] = excluded.get("FAILED", 0) + 1
                continue
            ratios.append(b.ms / c.ms)
        out.append(Summary(engine, geomean(ratios), len(ratios), excluded))
    return out


def format_summary(summaries: list[Summary], runs: int) -> str:
    """The geomean block printed under the table, stating what it is a mean of."""
    if not summaries:
        return ""
    lines = ["", f"geomean of b/<engine>, best-of-{runs}:"]
    for s in summaries:
        value = "n/a" if math.isnan(s.value) else f"{s.value:.3f}"
        lines.append(f"  b/{s.engine:<12} {value:>7}   {s.provenance()}")
    lines.append(
        "  A single run does not support three decimals. Use `--repeat N` before quoting a\n"
        "  figure, and quote the median to a precision that run's own spread supports."
    )
    return "\n".join(lines)


def format_repeats(per_run: list[list[Summary]]) -> str:
    """The spread across repeated runs — what makes a precision claim checkable.

    Args:
        per_run: One :func:`summarize` result per repetition, in order.

    Returns:
        A block giving min / median / max and the spread as a percentage of the median,
        per comparator. The spread is the number to quote a figure against: a geomean
        reported to a precision finer than it is asserting a stability nobody measured.
    """
    if len(per_run) < 2:
        return ""
    engines = [s.engine for s in per_run[0]]
    lines = ["", f"geomean across {len(per_run)} repeats:"]
    for i, engine in enumerate(engines):
        vals = sorted(run[i].value for run in per_run if not math.isnan(run[i].value))
        if not vals:
            continue
        lo, hi = vals[0], vals[-1]
        mid = vals[len(vals) // 2]
        spread = (hi - lo) / mid * 100 if mid else float("nan")
        lines.append(
            f"  b/{engine:<12} min {lo:.3f}  median {mid:.3f}  max {hi:.3f}   spread {spread:.1f}%"
        )
    lines.append(
        "  Quote the median, to a precision THIS spread supports. A 3% spread cannot\n"
        "  distinguish 0.699 from 0.705, and 'reproduces to within 0.001' across it is a\n"
        "  coincidence rather than corroboration.\n"
        "  A spread belongs to the suite that produced it, not to the box: measured here,\n"
        "  22 TPC-H queries gave 2.9% while a 12-query subset gave 6.4% on the SAME loaded\n"
        "  box, because a geomean over more cases averages more noise away. Do not carry a\n"
        "  spread from one suite to another."
    )
    return "\n".join(lines)
