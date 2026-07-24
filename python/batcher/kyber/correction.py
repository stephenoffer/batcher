"""Turning a window of measured q-errors into one cardinality-correction factor.

`learning` collects, per plan signature, the log ratio `log(actual / estimated)` of every
execution Core has measured. This module answers the question that history poses: given
those samples, by how much should the *next* structural estimate for that shape be
multiplied?

The naive answer — the plain geometric mean of the window — treats a noisy signature and a
consistent one identically, and treats a measurement from ten runs ago exactly like the one
that just landed. Both matter, because the thing being corrected is not stationary: the
structural estimator sharpens as the column-statistics loop learns NDVs and quantiles, and
the data itself drifts. So the estimator here does three things the plain mean does not:
it weights recent samples more heavily, it refuses to trust a mean the samples do not
actually support, and it clips a single wild run before it can move anything.
"""

from __future__ import annotations

import math

__all__ = ["correction_factor"]

# Prior standard deviation of the true log-correction, in nats. `ln 2` says: before seeing
# any evidence, a signature's structural estimate is expected to be within about a factor of
# two of the truth, and being off by 4x is a two-sigma event. It is deliberately loose —
# the loop exists because structural estimation *is* wrong — but it is finite, which is what
# makes a handful of inconsistent samples shrink toward "no correction" instead of being
# extrapolated into a plan-flipping factor.
_PRIOR_LOG_SIGMA = math.log(2.0)

# Fraction of the window over which a sample's weight halves: the newest sample weighs 1.0
# and the oldest of a full window weighs 0.5. That is deliberately gentle. Recency weighting
# exists so a correction tracks a genuine shift rather than waiting for the stale samples to
# fall off the end of the window, but it costs a property the flat mean has for free —
# symmetric noise no longer cancels exactly, because the newest sample of an alternating
# sequence outweighs its opposite. At a half-life of one window that residual is under a few
# percent (harmless, and shrinkage removes most of what remains); at half a window it reaches
# 8%, which is a correction invented out of pure noise.
_HALF_LIFE_FRACTION = 1.0


def correction_factor(logs: list[float], min_samples: int, max_factor: float) -> float:
    """The multiplicative correction implied by a window of log q-errors, or 1.0.

    The samples are oldest-first. Three steps, each of which the plain geometric mean skips:

    **Clip, then average.** Every sample is first clipped to `±log(max_factor)`. Clamping the
    *result* (which is also done, as a backstop) lets one absurd run — an operator that
    produced a million times its estimate because a UDF exploded — drag the mean the whole
    way to the clamp and hold it there for the rest of the window. Clipping the input bounds
    any single sample's influence to one window slot's worth.

    **Weight by recency.** Samples decay exponentially with a half-life of half the window,
    so the correction follows a real shift in the data rather than averaging it against a
    stale regime.

    **Shrink toward no correction.** The weighted mean `μ̂` has standard error `s/√n_eff`
    over the weighted sample. Under a normal prior centered on zero (no correction) with
    variance `τ²`, the posterior mean is

    ``μ̂ · τ² / (τ² + s²/n_eff)``

    which is the classical shrinkage estimator. Its behaviour is exactly what a cardinality
    correction wants: a signature whose runs consistently miss by 8x has `s ≈ 0`, so the
    factor passes through undiminished from its very first samples; a signature whose runs
    scatter between 0.2x and 5x has a large `s`, and its mean — which is mostly noise — is
    pulled back toward 1.0 rather than being stamped onto the next plan.

    Args:
        logs: `log(actual / estimated)` per execution, oldest first.
        min_samples: Below this many samples no correction is emitted at all.
        max_factor: The largest multiplicative correction allowed, in either direction.

    Returns:
        The factor to multiply a structural row estimate by; 1.0 when nothing is warranted.
    """
    n = len(logs)
    if n < min_samples or n == 0 or max_factor <= 1.0:
        return 1.0
    limit = math.log(max_factor)
    clipped = [min(limit, max(-limit, x)) for x in logs]
    weights = _recency_weights(n)
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return 1.0
    mean = sum(w * x for w, x in zip(weights, clipped, strict=True)) / total_weight
    # Kish's effective sample size: the number of equally-weighted samples carrying the same
    # information as this weighted set. Using `n` here would overstate the evidence, since the
    # decayed tail of the window contributes far less than one sample each.
    n_eff = total_weight**2 / sum(w * w for w in weights)
    if n_eff <= 1.0:
        return 1.0  # a single effective observation cannot establish a systematic bias
    # Weighted (Bessel-corrected via the effective size) variance of the log q-errors.
    variance = sum(w * (x - mean) ** 2 for w, x in zip(weights, clipped, strict=True))
    variance = variance / total_weight * (n_eff / (n_eff - 1.0))
    prior_variance = _PRIOR_LOG_SIGMA**2
    shrink = prior_variance / (prior_variance + variance / n_eff)
    factor = math.exp(mean * shrink)
    return min(max_factor, max(1.0 / max_factor, factor))


def _recency_weights(n: int) -> list[float]:
    """Exponentially decaying weights for `n` oldest-first samples (newest weighs 1.0)."""
    half_life = max(1.0, _HALF_LIFE_FRACTION * n)
    return [0.5 ** ((n - 1 - i) / half_life) for i in range(n)]
