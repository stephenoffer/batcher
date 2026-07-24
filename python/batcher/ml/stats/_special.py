"""Distribution tail probabilities for hypothesis testing, in dependency-free Python.

A hypothesis test collapses a whole column to one aggregated statistic and then asks a single
question of a reference distribution: how surprising is this value under the null. That last
step is pure scalar math on one number, so it belongs in the control plane and needs no engine
pass and no third-party runtime dependency. This module is that step — the regularized
incomplete beta and gamma functions and the three survival functions built on them (Student's
t, Fisher's F, chi-squared).

The implementations are the standard continued-fraction and series expansions (Numerical
Recipes, Lentz's method for the continued fractions), which are accurate to roughly machine
precision across the range a test needs. They are checked against SciPy's `stats` survival
functions in the tests; SciPy is the oracle, never a runtime dependency.
"""

from __future__ import annotations

import math

__all__ = ["chi2_sf", "f_sf", "normal_two_sided_p", "students_t_two_sided_p"]

_EPS = 3.0e-16
_FPMIN = 1.0e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function, by modified Lentz's method."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function ``I_x(a, b)`` in ``[0, 1]``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _gammainc_upper(s: float, x: float) -> float:
    """The regularized upper incomplete gamma ``Q(s, x) = P(X > x)`` for a Gamma(s) variable."""
    if x <= 0.0:
        return 1.0
    if x < s + 1.0:
        # Lower series P(s, x), then complement.
        term = 1.0 / s
        total = term
        n = s
        for _ in range(300):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * _EPS:
                break
        lower = total * math.exp(-x + s * math.log(x) - math.lgamma(s))
        return 1.0 - lower
    # Continued fraction for Q(s, x) directly (Lentz).
    b = x + 1.0 - s
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, 300):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h * math.exp(-x + s * math.log(x) - math.lgamma(s))


def students_t_two_sided_p(t: float, df: float) -> float:
    """The two-sided p-value ``P(|T| >= |t|)`` for a Student's t with `df` degrees of freedom."""
    if df <= 0:
        return math.nan
    if math.isinf(t):
        return 0.0
    return _betai(0.5 * df, 0.5, df / (df + t * t))


def normal_two_sided_p(z: float) -> float:
    """The two-sided p-value ``P(|Z| >= |z|)`` for a standard normal, via ``erfc``."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def f_sf(f: float, df1: float, df2: float) -> float:
    """The upper-tail probability ``P(X >= f)`` for an F distribution with ``(df1, df2)`` d.f."""
    if f <= 0.0:
        return 1.0
    if math.isinf(f):
        return 0.0
    return _betai(0.5 * df2, 0.5 * df1, df2 / (df2 + df1 * f))


def chi2_sf(x: float, df: float) -> float:
    """The upper-tail probability ``P(X >= x)`` for a chi-squared with `df` degrees of freedom."""
    if x <= 0.0:
        return 1.0
    return _gammainc_upper(0.5 * df, 0.5 * x)
