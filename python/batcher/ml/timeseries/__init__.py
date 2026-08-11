"""Time-series analysis: what a series' own past says about it.

A time series is the case an ordinary correlation cannot see — the signal is in how a column
relates to *itself* at an offset, which needs an ordering before it means anything. Everything
here therefore takes an `order_by` column and reduces an ordered window to aggregates the
engine computes; none of it materializes the series in the control plane.

`diagnostics`
    The autocorrelation function and the tests built on it: ACF and PACF, Ljung-Box,
    Durbin-Watson, and the scaled forecast error (MASE) that judges a model against a
    seasonal-naive baseline.

The package keeps `batcher.ml.timeseries.<name>` working as a flat import path — this is a
façade over the modules beside it, not a new namespace to learn.
"""

from __future__ import annotations

from batcher.ml.timeseries.diagnostics import (
    autocorrelation,
    autocorrelations,
    durbin_watson,
    ljung_box,
    mean_absolute_scaled_error,
    partial_autocorrelation,
    partial_autocorrelations,
)

__all__ = [
    "autocorrelation",
    "autocorrelations",
    "durbin_watson",
    "ljung_box",
    "mean_absolute_scaled_error",
    "partial_autocorrelation",
    "partial_autocorrelations",
]
