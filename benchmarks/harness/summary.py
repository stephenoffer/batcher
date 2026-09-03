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

__all__ = [
    "Summary",
    "case_ratios",
    "digits_for",
    "format_repeats",
    "format_summary",
    "format_unstable",
    "geomean",
    "summarize",
]

#: Statuses whose ratios are excluded from the geomean, and why. A ``DIVERGENT`` row is
#: excluded for the same reason its ratio is withheld from the table: the two engines did
#: not compute the same answer, so dividing their times compares unlike work.
_EXCLUDED = {
    "FAILED": "results disagreed",
    "ERROR": "no engine produced a result",
    "KILLED": "the case took its process down",
    "DIVERGENT": "a recorded semantic difference — unlike work, so unlike times",
    "DEGENERATE": "every engine returned nothing; the timing is of no work",
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


#: E[range] = d2(k) * sigma for k samples — the control-chart constant. A range under-states
#: the spread, and by a factor that depends on how many samples produced it: at k=2 the
#: expected range is only 1.13 sigma, at k=6 it is 2.53. Dividing by d2 turns an observed
#: range into an estimate of the spread that does not reward taking fewer samples.
_D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534}


def digits_for(value: float, spread_pct: float | None, samples: int = 3) -> int:
    """How many decimals `value` is entitled to, given the measured spread across repeats.

    Printing a figure to more decimals than its repeatability supports is the defect this
    module exists for, and printing the *warning* beside a three-decimal number does not fix
    it: the number and its qualification part company the moment either is copied into a
    document. So the precision is derived rather than advised.

    The figure is quoted to the decimal place of the uncertainty's leading digit — the
    metrological convention, where the last digit shown is the uncertain one. A 2.9% spread
    on 0.694 gives ~0.010, so two decimals: 0.69. A 0.3% spread gives ~0.001 and earns a third.

    The uncertainty is **not** the half-range, and the reason is the failure this argument
    was added for. An observed range under-states the true spread, by a factor that depends
    on how many samples produced it: `E[range] = d2(k) * sigma`, and `d2` runs 1.13 at k=2
    to 2.53 at k=6. So a two-run range looks *tighter* than a three-run one on the same
    instrument — measured on JOB, 1.5% at n=2 against 3.5% at n=3 — and a rule that read the
    range directly would award the under-sampled run **more** decimals. It did: n=2 would
    have printed 1.050 and n=3 1.05, three digits of confidence on the sample that was wrong.

    Dividing the range by `d2(samples)` removes that, continuously and without a special
    case for k=2. It is also the honest general statement, which a k=2 branch was not: a
    spread from any finite k is a lower bound that tightens with k — n=3 is not "fine", it
    is merely less optimistic than n=2, and the correction says so by how much.

    The uncertainty is rounded to one significant figure before its place is taken, and that
    is not cosmetic. A 2.88% spread puts it at 0.0099998 — a hair under 0.01 — and taking
    the place directly gives *three* digits for an uncertainty of one hundredth. The boundary
    is real rather than floating-point noise, and it sits exactly where the suites this is
    built for actually measure: TPC-H came in at 2.9%.

    With no spread measured (`spread_pct is None`, a single pass), two decimals is the floor:
    nothing here has ever repeated tighter than about 3%, so a third digit from one run is
    asserting a stability that has never been observed on this suite or any other.

    Args:
        value: The geomean.
        spread_pct: `(max - min) / median * 100` across repeats, or None for a single pass.
        samples: How many passes produced that range. Fewer samples mean a more optimistic
            range and therefore a larger corrected uncertainty.

    Returns:
        A decimal count, at least 1 and at most 4.
    """
    if not math.isfinite(value) or value <= 0:
        return 2
    if spread_pct is None:
        return 2
    d2 = _D2.get(max(2, samples), 2.534)
    uncertainty = abs(value) * max(spread_pct, 1e-9) / 100.0 / d2
    if uncertainty <= 0:
        return 4
    exponent = math.floor(math.log10(uncertainty))
    leading = round(uncertainty / 10**exponent)  # 1 significant figure
    if leading >= 10:  # e.g. 9.6 -> 10, which is the next decade up
        exponent += 1
    return max(1, min(4, -exponent))


def format_summary(summaries: list[Summary], runs: int, repeated: bool = False) -> str:
    """The geomean block printed under the table, stating what it is a mean of.

    Args:
        summaries: One per comparator, from :func:`summarize`.
        runs: The best-of-N this pass used.
        repeated: Whether more than one pass is being made. When it is,
            :func:`format_repeats` prints the spread and the per-pass figures are
            intermediate, so they keep three decimals; when it is not, this is the only
            number a reader will see and it is quoted to what one pass can support.
    """
    if not summaries:
        return ""
    lines = ["", f"geomean of b/<engine>, best-of-{runs}:"]
    for s in summaries:
        if math.isnan(s.value):
            value = "n/a"
        else:
            value = f"{s.value:.{3 if repeated else digits_for(s.value, None)}f}"
        lines.append(f"  b/{s.engine:<12} {value:>7}   {s.provenance()}")
    if not repeated:
        lines.append(
            "  SINGLE RUN — quoted to two decimals, which is what one pass supports. Run\n"
            "  `--repeat N` to measure this suite's own spread and earn a third digit."
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
        d = digits_for(mid, spread, samples=len(per_run))
        lines.append(
            f"  b/{engine:<12} min {lo:.3f}  median {mid:.3f}  max {hi:.3f}"
            f"   spread {spread:.1f}%   ->  quote {mid:.{d}f}"
        )
    lines.append(
        f"  Spread is an observed range over {len(per_run)} passes, corrected for that count\n"
        "  before the precision is derived: a range under-states the spread and does so more\n"
        "  the fewer passes produced it. Measured on JOB, n=2 gave 1.5% and n=3 gave 3.5% on\n"
        "  the same instrument, and the tighter figure made a 4.5% 'improvement' look real.\n"
        "  A spread from any finite number of passes is a lower bound that tightens with it."
    )
    lines.append(
        "  The `quote` column is the median at the precision this spread earns, derived\n"
        "  rather than advised — a warning printed beside a three-decimal number does not\n"
        "  survive being copied, and the digits do. A 3% spread cannot\n"
        "  distinguish 0.699 from 0.705, and 'reproduces to within 0.001' across it is a\n"
        "  coincidence rather than corroboration.\n"
        "  A spread belongs to the suite that produced it, not to the box: measured here,\n"
        "  22 TPC-H queries gave 2.9% while a 12-query subset gave 6.4% on the SAME loaded\n"
        "  box, because a geomean over more cases averages more noise away. Do not carry a\n"
        "  spread from one suite to another."
    )
    return "\n".join(lines)


def case_ratios(per_run: list[list[CompareResult]], engine: str) -> dict[str, list[float]]:
    """Every case's ``batcher/<engine>`` ratio, one entry per pass."""
    out: dict[str, list[float]] = {}
    for results in per_run:
        for r in results:
            b, c = r.engines.get("batcher"), r.engines.get(engine)
            if b is None or c is None or not b.ms or not c.ms:
                continue
            if r.status in _EXCLUDED or b.correct is False or c.correct is False:
                continue
            out.setdefault(r.name, []).append(b.ms / c.ms)
    return out


def format_unstable(per_run: list[list[CompareResult]], engines: list[str], top: int = 8) -> str:
    """The least reproducible per-query ratios — the rows most likely to be over-read.

    The geomean carries a spread; the per-query rows above it do not, and that asymmetry is
    backwards. Measured on TPC-H sf1 over five whole passes, the **median per-query ratio
    moved 19% and the worst moved 61%**, while the geomean over the same 22 moved 6%. Averaging
    22 queries cancels most of the noise; a single row keeps all of it.

    q6 is the case that makes it concrete: it ranged 0.76 to 1.57 across five passes. Printed
    from one pass it reads as a comfortable Batcher win or a clear Batcher loss depending
    only on which pass was printed, and the table gives the reader nothing to tell those
    apart. So the rows that move most are named, rather than left looking as solid as the
    aggregate they sit under.
    """
    if len(per_run) < 2 or "batcher" not in engines:
        return ""
    lines: list[str] = []
    for engine in (e for e in engines if e != "batcher"):
        ratios = case_ratios(per_run, engine)
        spreads = []
        for name, vals in ratios.items():
            if len(vals) < 2:
                continue
            v = sorted(vals)
            mid = v[len(v) // 2]
            if mid:
                spreads.append(((v[-1] - v[0]) / mid * 100, name, v[0], mid, v[-1]))
        if not spreads:
            continue
        spreads.sort(reverse=True)
        lines.append("")
        lines.append(f"least reproducible b/{engine} rows, across {len(per_run)} passes:")
        for pct, name, lo, mid, hi in spreads[:top]:
            flips = "  <- crosses 1.00" if lo < 1.0 < hi else ""
            lines.append(
                f"  {name:<14} min {lo:.2f}  median {mid:.2f}  max {hi:.2f}"
                f"   spread {pct:>4.0f}%{flips}"
            )
        med = sorted(s[0] for s in spreads)[len(spreads) // 2]
        lines.append(
            f"  median per-query spread {med:.0f}% over {len(spreads)} cases. A single-pass\n"
            "  per-query ratio is far less reproducible than the geomean above it — quote a\n"
            "  row only if it was repeated, and never a row that crosses 1.00."
        )
    return "\n".join(lines)
