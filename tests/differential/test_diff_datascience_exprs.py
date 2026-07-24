"""The data-science `Expr` and `Dataset` surface matches DuckDB.

`diff`/`pct_change`/`rank`/`is_duplicated`/`is_unique`/`rolling_*` are pure desugarings
over the window operator (they exist because a `WindowExpr` now composes like a scalar
— see `test_diff_window_compose`), and `corr`/`cov`/`median`/`quantile`/`tail` are
scalar terminals over the existing aggregates. Nothing here adds IR, so the oracle
check is that each desugaring computes exactly what the equivalent SQL does —
including on the null and tie edges where a hand-rolled version would drift.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered
from batcher import col

pytestmark = pytest.mark.differential


def _t():
    return pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5, 6], type=pa.int64()),
            "g": pa.array(["a", "a", "a", "b", "b", "b"], type=pa.string()),
            "x": pa.array([10, 20, 20, 7, None, 21], type=pa.int64()),
        }
    )


@pytest.fixture
def ds(duck):
    t = _t()
    duck.register("t", t)
    return bt.from_arrow(t)


def test_diff_matches_duckdb(duck, ds):
    got = ds.select("id", d=col("x").diff(order_by=["id"])).to_arrow()
    assert_same(got, duck.sql("SELECT id, x - lag(x, 1) OVER (ORDER BY id) AS d FROM t"))


def test_diff_partitioned_matches_duckdb(duck, ds):
    got = ds.select("id", d=col("x").diff(partition_by=["g"], order_by=["id"])).to_arrow()
    assert_same(
        got,
        duck.sql("SELECT id, x - lag(x, 1) OVER (PARTITION BY g ORDER BY id) AS d FROM t"),
    )


def test_diff_negative_n_looks_forward_matches_duckdb(duck, ds):
    got = ds.select("id", d=col("x").diff(-1, order_by=["id"])).to_arrow()
    assert_same(got, duck.sql("SELECT id, x - lead(x, 1) OVER (ORDER BY id) AS d FROM t"))


def test_pct_change_matches_duckdb(duck, ds):
    got = ds.select("id", p=col("x").pct_change(order_by=["id"])).to_arrow()
    assert_same(
        got,
        duck.sql("SELECT id, x / lag(x, 1) OVER (ORDER BY id) - 1 AS p FROM t"),
    )


@pytest.mark.parametrize(
    ("method", "sql_fn"),
    [("min", "rank"), ("dense", "dense_rank"), ("ordinal", "row_number")],
)
def test_rank_methods_match_duckdb(duck, ds, method, sql_fn):
    """Ties: `min` leaves a gap, `dense` does not, `ordinal` breaks them."""
    got = ds.select("id", r=col("x").rank(method)).to_arrow()
    assert_same(got, duck.sql(f"SELECT id, {sql_fn}() OVER (ORDER BY x) AS r FROM t"))


def test_rank_descending_partitioned_matches_duckdb(duck, ds):
    got = ds.select("id", r=col("x").rank(descending=True, partition_by=["g"])).to_arrow()
    assert_same(
        got,
        duck.sql("SELECT id, rank() OVER (PARTITION BY g ORDER BY x DESC) AS r FROM t"),
    )


def test_rank_rejects_unknown_method(ds):
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="method must be one of"):
        col("x").rank("average")


def test_is_duplicated_matches_duckdb(duck, ds):
    """Nulls form their own group, so `count(1)` (not `count(x)`) is the oracle."""
    got = ds.select("id", d=col("x").is_duplicated()).to_arrow()
    assert_same(got, duck.sql("SELECT id, count(1) OVER (PARTITION BY x) > 1 AS d FROM t"))


def test_is_unique_matches_duckdb(duck, ds):
    got = ds.select("id", u=col("x").is_unique()).to_arrow()
    assert_same(got, duck.sql("SELECT id, count(1) OVER (PARTITION BY x) = 1 AS u FROM t"))


def test_is_unique_in_filter_matches_duckdb(duck, ds):
    got = ds.filter(col("x").is_unique()).to_arrow()
    assert_same(
        got,
        duck.sql(
            "SELECT id, g, x FROM "
            "(SELECT *, count(1) OVER (PARTITION BY x) AS c FROM t) WHERE c = 1"
        ),
    )


@pytest.mark.parametrize(
    ("method", "sql_fn"),
    [
        ("rolling_sum", "sum"),
        ("rolling_mean", "avg"),
        ("rolling_min", "min"),
        ("rolling_max", "max"),
        ("rolling_count", "count"),
    ],
)
def test_rolling_matches_duckdb_partial_leading_windows(duck, ds, method, sql_fn):
    """Without `min_periods` the leading rows aggregate a partial frame, as SQL does."""
    got = ds.select("id", r=getattr(col("x"), method)(3, order_by=["id"])).to_arrow()
    assert_same(
        got,
        duck.sql(
            f"SELECT id, {sql_fn}(x) OVER (ORDER BY id "
            "ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS r FROM t"
        ),
    )


def test_rolling_min_periods_nulls_the_short_frames(duck, ds):
    """`min_periods` nulls a row whose frame holds too few non-null values — the
    Polars default, expressed as a `count` guard over the same frame."""
    got = ds.select("id", r=col("x").rolling_sum(3, min_periods=3, order_by=["id"])).to_arrow()
    assert_same(
        got,
        duck.sql(
            "SELECT id, CASE WHEN count(x) OVER w >= 3 THEN sum(x) OVER w END AS r "
            "FROM t WINDOW w AS (ORDER BY id ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)"
        ),
    )


def test_rolling_partitioned_matches_duckdb(duck, ds):
    got = ds.select("id", r=col("x").rolling_sum(2, partition_by=["g"], order_by=["id"])).to_arrow()
    assert_same(
        got,
        duck.sql(
            "SELECT id, sum(x) OVER (PARTITION BY g ORDER BY id "
            "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS r FROM t"
        ),
    )


def test_rolling_composes_with_arithmetic(duck, ds):
    """A rolling aggregate is an ordinary operand — here, deviation from the moving mean."""
    got = ds.select("id", dev=col("x") - col("x").rolling_mean(3, order_by=["id"])).to_arrow()
    assert_same(
        got,
        duck.sql(
            "SELECT id, x - avg(x) OVER (ORDER BY id "
            "ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS dev FROM t"
        ),
    )


@pytest.mark.parametrize(("size", "mp"), [(0, None), (-1, None), (3, 0), (3, 4)])
def test_rolling_rejects_bad_frames(size, mp):
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError):
        col("x").rolling_sum(size, min_periods=mp)


def test_min_periods_reuses_one_window_node():
    """The value window appears in both `CASE` branches; the hoist must share it."""
    from batcher.plan.expr_rewrite import hoist_windows

    _, hoisted = hoist_windows([col("x").rolling_sum(3, min_periods=3)])
    assert sorted(w.func for _, w in hoisted) == ["count", "sum"]


def test_fill_nan_matches_duckdb(duck):
    """NaN is replaced; null stays null (NaN is a float value, not a null)."""
    t = pa.table({"x": pa.array([1.0, float("nan"), None, 4.0], type=pa.float64())})
    duck.register("f", t)
    got = bt.from_arrow(t).select(r=col("x").fill_nan(0.0)).to_arrow()
    assert_same(got, duck.sql("SELECT CASE WHEN isnan(x) THEN 0.0 ELSE x END AS r FROM f"))


# --- Dataset scalar terminals -------------------------------------------------


def test_corr_matches_duckdb(duck, ds):
    assert ds.corr("id", "x") == pytest.approx(duck.sql("SELECT corr(id, x) FROM t").fetchone()[0])


def test_cov_sample_and_population_match_duckdb(duck, ds):
    assert ds.cov("id", "x") == pytest.approx(
        duck.sql("SELECT covar_samp(id, x) FROM t").fetchone()[0]
    )
    assert ds.cov("id", "x", ddof=0) == pytest.approx(
        duck.sql("SELECT covar_pop(id, x) FROM t").fetchone()[0]
    )


def test_cov_rejects_bad_ddof(ds):
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="ddof must be 0"):
        ds.cov("id", "x", ddof=2)


def test_median_matches_duckdb(duck, ds):
    assert ds.median("x") == pytest.approx(duck.sql("SELECT median(x) FROM t").fetchone()[0])


@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.9, 1.0])
def test_quantile_matches_duckdb(duck, ds, q):
    assert ds.quantile("x", q) == pytest.approx(
        duck.sql(f"SELECT quantile_cont(x, {q}) FROM t").fetchone()[0]
    )


def test_quantile_rejects_out_of_range_q(ds):
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match=r"q must be in \[0, 1\]"):
        ds.quantile("x", 1.5)


def test_tail_matches_duckdb_offset(duck, ds):
    # Ordered: both sides sort on `id`, so row order is part of what `tail` promises and the
    # comparison has to see it. `assert_same` is order-independent and would pass on a
    # `tail` that returned the right two rows the wrong way round.
    got = ds.sort("id").tail(2).to_arrow()
    assert_same_ordered(got, duck.sql("SELECT * FROM t ORDER BY id OFFSET 4"))


def test_tail_larger_than_relation_returns_everything(ds):
    assert ds.tail(100).count() == 6


def test_tail_zero_is_empty(ds):
    assert ds.tail(0).count() == 0


def test_tail_rejects_negative_n(ds):
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="non-negative"):
        ds.tail(-1)


def test_dataset_truthiness_is_rejected(ds):
    """`if ds:` must not silently execute a full count."""
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="truth value of a Dataset is ambiguous"):
        bool(ds)


def test_silu_and_gelu_match_torch_and_closed_form():
    """`.silu()` and `.gelu()` (tanh approximation) — the modern transformer activations,
    composed from engine primitives. Checked against the closed form and, if available,
    torch's own functions."""
    import math

    xs = [-2.0, -0.5, 0.0, 0.5, 2.0]
    t = pa.table({"x": xs})
    out = bt.from_arrow(t).select(s=col("x").silu(), g=col("x").gelu()).to_pydict()

    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    def gelu_tanh(x):
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + math.tanh(c * (x + 0.044715 * x**3)))

    assert out["s"] == pytest.approx([x * sigmoid(x) for x in xs])
    assert out["g"] == pytest.approx([gelu_tanh(x) for x in xs])

    torch = pytest.importorskip("torch")
    xt = torch.tensor(xs)
    assert out["s"] == pytest.approx(torch.nn.functional.silu(xt).tolist(), rel=1e-6)
    assert out["g"] == pytest.approx(
        torch.nn.functional.gelu(xt, approximate="tanh").tolist(), rel=1e-6
    )


def test_mish_hardsigmoid_hardswish_match_torch():
    """`.mish()`, `.hardsigmoid()`, `.hardswish()` — more activations composed from engine
    primitives. Checked against the closed form and torch's own functions."""
    import math

    xs = [-3.0, -1.0, 0.0, 1.0, 2.0]
    t = pa.table({"x": xs})
    out = (
        bt.from_arrow(t)
        .select(m=col("x").mish(), hs=col("x").hardsigmoid(), hw=col("x").hardswish())
        .to_pydict()
    )

    def mish(x):
        return x * math.tanh(math.log1p(math.exp(x)))

    def hsig(x):
        return min(1.0, max(0.0, (x + 3.0) / 6.0))

    assert out["m"] == pytest.approx([mish(x) for x in xs])
    assert out["hs"] == pytest.approx([hsig(x) for x in xs])
    assert out["hw"] == pytest.approx([x * hsig(x) for x in xs])

    torch = pytest.importorskip("torch")
    import torch.nn.functional as functional

    xt = torch.tensor(xs)
    assert out["m"] == pytest.approx(functional.mish(xt).tolist(), rel=1e-6)
    assert out["hs"] == pytest.approx(functional.hardsigmoid(xt).tolist(), rel=1e-6)
    assert out["hw"] == pytest.approx(functional.hardswish(xt).tolist(), rel=1e-6)


def test_leaky_relu_and_elu_match_torch():
    """`.leaky_relu()` / `.elu()` (parametric) — composed via `when`, checked vs torch."""
    import math

    xs = [-2.0, -0.5, 0.0, 1.0, 2.0]
    t = pa.table({"x": xs})
    out = (
        bt.from_arrow(t)
        .select(
            l=col("x").leaky_relu(),
            l2=col("x").leaky_relu(0.2),
            e=col("x").elu(),
            e2=col("x").elu(0.5),
        )
        .to_pydict()
    )

    def leaky(x, s):
        return x if x > 0 else s * x

    def elu(x, a):
        return x if x > 0 else a * (math.exp(x) - 1.0)

    assert out["l"] == pytest.approx([leaky(x, 0.01) for x in xs])
    assert out["l2"] == pytest.approx([leaky(x, 0.2) for x in xs])
    assert out["e"] == pytest.approx([elu(x, 1.0) for x in xs])
    assert out["e2"] == pytest.approx([elu(x, 0.5) for x in xs])

    torch = pytest.importorskip("torch")
    import torch.nn.functional as functional

    xt = torch.tensor(xs)
    assert out["l"] == pytest.approx(functional.leaky_relu(xt, 0.01).tolist(), rel=1e-6)
    assert out["e2"] == pytest.approx(functional.elu(xt, 0.5).tolist(), rel=1e-6)


def test_hardtanh_softsign_tanhshrink_match_torch():
    """`.hardtanh()` / `.softsign()` / `.tanhshrink()` — more activations vs closed form + torch."""
    import math

    xs = [-2.0, -0.5, 0.0, 0.5, 2.0]
    t = pa.table({"x": xs})
    out = (
        bt.from_arrow(t)
        .select(h=col("x").hardtanh(), s=col("x").softsign(), ts=col("x").tanhshrink())
        .to_pydict()
    )
    assert out["h"] == pytest.approx([min(1.0, max(-1.0, x)) for x in xs])
    assert out["s"] == pytest.approx([x / (1.0 + abs(x)) for x in xs])
    assert out["ts"] == pytest.approx([x - math.tanh(x) for x in xs])

    torch = pytest.importorskip("torch")
    import torch.nn.functional as functional

    xt = torch.tensor(xs)
    assert out["h"] == pytest.approx(functional.hardtanh(xt).tolist(), rel=1e-6)
    assert out["s"] == pytest.approx(functional.softsign(xt).tolist(), rel=1e-6)
    assert out["ts"] == pytest.approx(functional.tanhshrink(xt).tolist(), rel=1e-6)
