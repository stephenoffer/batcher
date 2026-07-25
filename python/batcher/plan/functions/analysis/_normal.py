"""The standard normal quantile, on the neutral side of the layer boundary.

Both a normal-output `QuantileTransformer` (in `ml`) and the confidence-interval half-widths
(in `plan.functions.analysis.inference`) need an inverse normal CDF on the driver, at
plan-build time, for a handful of scalars. `plan` cannot import `ml`, so this lives here --
the lowest layer both callers can see -- rather than being pasted into each.

Driver-side only: it turns a Python float into a Python float while a plan is being built, so
the engine never evaluates an inverse CDF per row.
"""

from __future__ import annotations

import math

from batcher._internal.errors import PlanError

__all__ = ["normal_ppf"]


def normal_ppf(probability: float) -> float:
    """The standard normal quantile (inverse CDF) at `probability`.

    Acklam's rational approximation, accurate to about 1.15e-9 over the whole range. Used
    to precompute constants for a normal-output `QuantileTransformer`, so the engine never
    evaluates an inverse CDF per row and the transform needs no SciPy.

    Args:
        probability: A probability strictly between 0 and 1.

    Returns:
        The value whose standard normal CDF is `probability`.

    Raises:
        PlanError: If `probability` is not strictly between 0 and 1.

    Examples:
        .. doctest::

            >>> from batcher.plan.functions.analysis._normal import normal_ppf
            >>> round(normal_ppf(0.975), 4)
            1.96
    """
    if not 0.0 < probability < 1.0:
        raise PlanError(f"normal_ppf needs a probability in (0, 1), got {probability}")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    low, high = 0.02425, 1 - 0.02425
    if probability < low:
        q = math.sqrt(-2 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if probability > high:
        q = math.sqrt(-2 * math.log(1 - probability))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = probability - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )
