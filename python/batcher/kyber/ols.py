"""Shared OLS sufficient statistics for Kyber's learned crossover models.

Several learned thresholds are the solution of "where does line A overtake line B" —
the GPU/CPU backend crossover (`kyber.gpu.adaptive`) and the broadcast / sort-merge
crossovers (`kyber.learned_tuning.crossover`). Each folds `(x, y)` observations into
the same running statistics plus an observed x-range, then fits a line.

Only the *pure* halves live here: `ols_update` builds the new statistics dict and
`fit_ols` solves it. Each caller keeps its own write sink (one uses
`hub.put_keyed_param`, the other routes through `plan_cache.record_write` so a learned
write invalidates cached plans), because *where* the statistics are persisted is a
per-model decision — but *how a line is identified from them* must not be.

That split is the point. These two `_fit` bodies were copies, and they diverged: the
relative-spread guard below was added to one after a measured failure and never
reached the other, so the same clustered-sample garbage fit that had been diagnosed
and fixed in one model was still live in the other. One implementation, one guard.
"""

from __future__ import annotations

__all__ = ["MIN_SAMPLES", "fit_ols", "ols_update"]

# Trust a fit only after enough spread-out samples — single-run timings are noisy, so a
# handful of points can swing the intercept wildly. This keeps the (already well-chosen)
# config default in charge until there is real evidence, and the loop stays conservative.
MIN_SAMPLES = 8

# The x-spread a bucket needs, as a fraction of its magnitude, before a line is identifiable
# from it. 1% of the largest observed input: enough that the samples describe genuinely
# different input sizes rather than one size measured repeatedly.
_MIN_RELATIVE_SPREAD = 0.01

# The fraction of the response's variance the line must explain before it is trusted.
#
# Spread in `x` makes a slope *identifiable*; it does not make it *real*. Eight runs spread
# across a decade of input sizes whose times are dominated by unrelated noise still produce a
# line, and that line's intercept is the whole numerator of a crossover threshold. R² is the
# direct measure of whether the line explains anything: at 0.5 the input size accounts for
# half the variation in time, which for a cost model that is *claiming* time is linear in rows
# is a low bar to clear — and a bucket that cannot clear it should not be moving a threshold.
_MIN_R_SQUARED = 0.5


def ols_update(s: dict, x: float, y: float) -> dict:
    """The OLS statistics in `s` extended by one `(x, y)` observation.

    Pure — returns a new dict for the caller to persist through its own sink. `xmin`/`xmax`
    track the observed x-range so `fit_ols` can tell a genuinely spread-out sample from a
    cluster of runs at one input size, which cannot identify a slope however many there are.

    The accumulated state is **centered** (running means and co-moments), not the textbook
    power sums `(Σx, Σy, Σx², Σxy)`. Power sums are mergeable and compact, but recovering the
    slope from them requires forming `n·Σx² - (Σx)²` — a subtraction of two numbers that agree
    to fourteen digits whenever the runs are clustered, so the difference is float noise and
    the fitted line is noise scaled by it. This is the same cancellation the variance and
    covariance aggregates were rewritten to avoid, and the same fix: accumulate the centered
    quantity directly, with West's weighted-update form so the state stays streaming.

    A dict in the old power-sum shape is not upgraded in place (its centered moments cannot
    be recovered without cancellation, which is the whole point); it is discarded and the
    bucket restarts. A learned threshold is an optimization, so a rebuild costs a few runs of
    the config default — never a wrong answer.

    Args:
        s: The prior statistics, or an empty dict to start a new bucket.
        x: The observation's independent value (typically a row count).
        y: The observation's dependent value (typically a wall time in ms).

    Returns:
        The updated statistics dict.
    """
    prior = s if _is_centered(s) else {}
    n = float(prior.get("n", 0.0)) + 1.0
    mx = float(prior.get("mx", 0.0))
    my = float(prior.get("my", 0.0))
    m2x = float(prior.get("m2x", 0.0))
    m2y = float(prior.get("m2y", 0.0))
    cxy = float(prior.get("cxy", 0.0))
    dx = x - mx
    dy = y - my
    mx += dx / n
    my += dy / n
    # Each co-moment is updated with the deviation from the *old* mean times the deviation
    # from the *new* one — Welford's identity, which is exact and never forms a difference of
    # large sums.
    m2x += dx * (x - mx)
    m2y += dy * (y - my)
    cxy += dx * (y - my)
    return {
        "n": n,
        "mx": mx,
        "my": my,
        "m2x": m2x,
        "m2y": m2y,
        "cxy": cxy,
        "xmin": min(float(prior.get("xmin", x)), x),
        "xmax": max(float(prior.get("xmax", x)), x),
    }


def fit_ols(s: dict) -> tuple[float, float] | None:
    """`(intercept, slope)` from the accumulated statistics, or `None` when unidentifiable.

    Returns `None` rather than a garbage line whenever the samples cannot support a fit, so
    every caller falls back to its config default. Three things must hold:

    * **enough samples** — a handful of single-run timings can swing an intercept wildly;
    * **enough spread in `x`** — runs clustered at one input size cannot identify a slope
      however many of them there are. Measured: 8 runs at ~10M rows spread over 3 rows, with
      ±0.5 ms of ordinary jitter on a 100 ms measurement, fit an intercept of 954,873 (true
      100) and a sign-flipped slope;
    * **the line must explain the data** — `R² = C_xy² / (M2x·M2y)`, the fraction of the
      response's variance attributable to `x`. Spread makes a slope identifiable; it does not
      make it real, and a threshold moved by a line that explains nothing is worse than the
      default it replaced.

    Args:
        s: The accumulated statistics, as produced by `ols_update`.

    Returns:
        The fitted `(intercept, slope)`, or `None` if the data cannot identify a line.
    """
    if not _is_centered(s):
        return None  # legacy power-sum state: rebuild rather than fit from it
    n = float(s.get("n", 0.0))
    xmin, xmax = float(s.get("xmin", 0.0)), float(s.get("xmax", 0.0))
    if n < MIN_SAMPLES or xmax <= xmin:
        return None
    if xmax - xmin < _MIN_RELATIVE_SPREAD * abs(xmax):
        return None
    m2x = float(s.get("m2x", 0.0))
    m2y = float(s.get("m2y", 0.0))
    cxy = float(s.get("cxy", 0.0))
    if m2x <= 0.0:
        return None
    if m2y > 0.0 and (cxy * cxy) / (m2x * m2y) < _MIN_R_SQUARED:
        return None
    slope = cxy / m2x
    return float(s.get("my", 0.0)) - slope * float(s.get("mx", 0.0)), slope


def _is_centered(s: dict) -> bool:
    """Whether `s` carries the centered state (as opposed to a legacy power-sum dict)."""
    return "m2x" in s and "cxy" in s
