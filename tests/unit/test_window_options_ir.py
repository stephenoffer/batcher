"""The `opts` object a series or `qcut` window function carries on the JSON IR.

It mirrors `bc_ir::WindowOpts`, whose every field defaults to the historical behaviour, so
the contract is: a function using no option serializes byte-identically to before (no
`opts` key at all), and one using an option sends only that option. The Rust side's
`missing_keys_take_the_historical_defaults` test holds the other half.
"""

from __future__ import annotations

import pytest

from batcher import col
from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.nodes import WindowOptions
from batcher.plan.logical.window import WindowFuncSpec

pytestmark = pytest.mark.unit


def test_defaults_send_no_opts_key():
    spec = WindowFuncSpec("ewm_mean", col("x"), "e", alpha=0.5, opts=WindowOptions())
    assert "opts" not in spec.to_ir()
    assert "opts" not in WindowFuncSpec("interpolate", col("x"), "i").to_ir()


def test_only_the_set_options_are_sent():
    ewm = WindowFuncSpec(
        "ewm_var",
        col("x"),
        "e",
        alpha=0.5,
        ignore_nulls=True,
        opts=WindowOptions(adjust=False, min_periods=3),
    ).to_ir()
    assert ewm["opts"] == {"adjust": False, "min_periods": 3}
    assert ewm["ignore_nulls"] is True
    interp = WindowFuncSpec(
        "interpolate", col("x"), "i", opts=WindowOptions(max_gap=2, by_value=True)
    ).to_ir()
    assert interp["opts"] == {"max_gap": 2.0, "by_value": True}
    qcut = WindowFuncSpec(
        "qcut", col("x"), "q", opts=WindowOptions(probs=(0.0, 0.5, 1.0), drop_duplicates=True)
    ).to_ir()
    assert qcut["opts"] == {"probs": ["0.0", "0.5", "1.0"], "drop_duplicates": True}


@pytest.mark.parametrize(
    ("func", "kwargs", "match"),
    [
        ("sum", {"opts": WindowOptions(adjust=False)}, "adjust"),
        ("ewm_mean", {"half_life": 2.0, "opts": WindowOptions(min_periods=2)}, "adjust"),
        ("ewm_mean", {"alpha": 0.5, "opts": WindowOptions(max_gap=1)}, "max_gap"),
        ("interpolate", {"opts": WindowOptions(probs=(0.0, 1.0))}, "probabilities"),
        ("qcut", {}, "exactly one of"),
        ("lag", {"ignore_nulls": True}, "ignore_nulls"),
        ("ewm_mean", {"half_life": 2.0, "ignore_nulls": True}, "ignore_nulls"),
    ],
)
def test_an_option_a_function_would_ignore_is_refused(func, kwargs, match):
    with pytest.raises(PlanError, match=match):
        WindowFuncSpec(func, col("x"), "w", **kwargs)


def test_qcut_probabilities_follow_pandas_arithmetic():
    """`q=10` is numpy's ``linspace(0, 1, 11)`` passed through ``p * 100 / 100``, the route
    pandas takes through ``numpy.percentile``; the bits decide a value on an edge."""
    np = pytest.importorskip("numpy")
    probs = col("x").qcut(10).opts.probs
    want = tuple(float(p) for p in (np.linspace(0, 1, 11) * 100.0) / 100.0)
    assert probs == want
