"""Shared OLS sufficient statistics for Kyber's learned crossover models.

Several learned thresholds are the solution of "where does line A overtake line B" —
the GPU/CPU backend crossover (`kyber.gpu.adaptive`) and the broadcast / sort-merge
crossovers (`kyber.learned_tuning.crossover`). Each folds `(x, y)` observations into
the same five OLS sufficient statistics plus an observed x-range, then fits a line.

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
# from it. 1% of the largest observed input: enough that `n*sxx - sx*sx` is signal rather than
# float noise, and easily cleared by any bucket that has genuinely seen different input sizes.
_MIN_RELATIVE_SPREAD = 0.01


def ols_update(s: dict, x: float, y: float) -> dict:
    """The OLS sufficient statistics in `s` extended by one `(x, y)` observation.

    Pure — returns a new dict for the caller to persist through its own sink. `xmin`/`xmax`
    track the observed x-range so `fit_ols` can tell a genuinely spread-out sample from a
    cluster of runs at one input size, which cannot identify a slope however many there are.

    Args:
        s: The prior statistics, or an empty dict to start a new bucket.
        x: The observation's independent value (typically a row count).
        y: The observation's dependent value (typically a wall time in ms).

    Returns:
        The updated statistics dict.
    """
    return {
        "n": int(s.get("n", 0)) + 1,
        "sx": float(s.get("sx", 0.0)) + x,
        "sy": float(s.get("sy", 0.0)) + y,
        "sxx": float(s.get("sxx", 0.0)) + x * x,
        "sxy": float(s.get("sxy", 0.0)) + x * y,
        "xmin": min(float(s.get("xmin", x)), x),
        "xmax": max(float(s.get("xmax", x)), x),
    }


def fit_ols(s: dict) -> tuple[float, float] | None:
    """`(intercept, slope)` from OLS sufficient statistics, or `None` when unidentifiable.

    Returns `None` rather than a garbage line whenever the samples cannot support a fit —
    too few of them, or too clustered — so every caller falls back to its config default.

    The spread test is deliberately **relative**, not merely non-zero. `denom = n*sxx - sx*sx`
    is the unstable textbook form: with runs clustered near one input size, `n*sxx` and `sx*sx`
    agree to ~14 significant digits and their difference is float noise. The `denom <= 0` guard
    catches the negative half of that noise but not the small-positive half, so an *absolute*
    `xmax > xmin` lets a garbage line through. Measured: 8 runs at ~10M rows spread over 3 rows,
    with +/-0.5 ms of ordinary timing jitter on a 100 ms measurement, fit intercept 954,873
    (true 100) and a sign-flipped slope — and that intercept is the whole numerator of a
    crossover.

    Args:
        s: The accumulated statistics, as produced by `ols_update`.

    Returns:
        The fitted `(intercept, slope)`, or `None` if the data cannot identify a line.
    """
    n = int(s.get("n", 0))
    xmin, xmax = float(s.get("xmin", 0.0)), float(s.get("xmax", 0.0))
    if n < MIN_SAMPLES or xmax <= xmin:
        return None
    if xmax - xmin < _MIN_RELATIVE_SPREAD * abs(xmax):
        return None
    sx, sy, sxx, sxy = s.get("sx", 0.0), s.get("sy", 0.0), s.get("sxx", 0.0), s.get("sxy", 0.0)
    denom = n * sxx - sx * sx
    if denom <= 0.0:
        return None
    slope = (n * sxy - sx * sy) / denom
    return (sy - slope * sx) / n, slope
