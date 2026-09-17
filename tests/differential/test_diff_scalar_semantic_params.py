"""The scalar parameters that restore another engine's meaning, and the scalar bugfixes, vs DuckDB.

Every default is DuckDB's, and each test pins the default beside the parameter, so a
parameter cannot drift the default. A parameterised form is checked against the DuckDB
expression that spells the same meaning: ``round_even`` for ties-to-even, ``IS NOT DISTINCT
FROM`` for null-safe membership, a framed window aggregate under a ``CASE`` for Polars'
running nulls, ``<>`` for boolean xor, ``bin``/``hex`` for two's-complement digits. The
fixtures carry nulls, NaN, ``-0.0``, exact ties, negative operands and 64-bit boundaries.

The competitor half of the same claims (Polars, Daft, Spark's documented examples) is
`test_diff_scalar_list_competitor_params.py`.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

_I64_MIN, _I64_MAX = -(2**63), 2**63 - 1


def _types(table: pa.Table) -> dict[str, pa.DataType]:
    return {f.name: f.type for f in table.schema}


# --- round(mode="half_to_even") --------------------------------------------------------------

_FLOATS = [2.5, -2.5, 0.5, -0.5, 1.5, 3.5, 0.25, 1.25, 2.675, 1.005, -0.0, 1e308, None, 7.0]


@pytest.fixture
def floats(duck):
    tbl = pa.table({"id": list(range(len(_FLOATS))), "x": pa.array(_FLOATS, pa.float64())})
    duck.register("f", tbl)
    return tbl


@pytest.mark.parametrize("digits", [None, 0, 1, 2, -1])
def test_round_half_to_even_matches_round_even(duck, floats, digits):
    d = 0 if digits is None else digits
    out = bt.from_arrow(floats).select(
        "id",
        away=col("x").round(digits),
        even=col("x").round(digits, mode="half_to_even"),
    )
    assert_same(
        out.collect(),
        duck.sql(f"SELECT id, round(x, {d}) AS away, round_even(x, {d}) AS even FROM f"),
    )


def test_round_half_to_even_keeps_negative_zero_and_nan(duck):
    tbl = pa.table({"x": pa.array([-0.5, float("nan"), -0.0], pa.float64())})
    got = bt.from_arrow(tbl).select(r=col("x").round(mode="half_to_even")).to_pydict()["r"]
    want = duck.sql("SELECT round_even(x, 0) AS r FROM tbl").fetchall()
    assert [math.copysign(1.0, v) for v in (got[0], got[2])] == [
        math.copysign(1.0, w[0]) for w in (want[0], want[2])
    ]
    assert math.isnan(got[1]) and math.isnan(want[1][0])


def test_round_half_to_even_on_integers_rounds_to_even_tens_and_stays_int64(duck):
    ints = [25, -25, 15, -15, 35, 26, 0, None]
    tbl = pa.table({"id": list(range(len(ints))), "x": pa.array(ints, pa.int64())})
    duck.register("i", tbl)
    out = bt.from_arrow(tbl).select("id", r=col("x").round(-1, mode="half_to_even")).collect()
    assert _types(out)["r"] == pa.int64()
    # DuckDB's `round_even(BIGINT, -1)` answers DOUBLE; the values are what is compared.
    assert_same(out, duck.sql("SELECT id, round_even(x, -1)::BIGINT AS r FROM i"))
    # Past 2^53 DuckDB's DOUBLE loses the answer (`_I64_MIN + 5` comes back as `_I64_MIN`);
    # the integer path keeps it exact.
    edge = bt.from_arrow(pa.table({"x": [_I64_MIN + 5, 2**53 + 7]}))
    got = edge.select(r=col("x").round(-1, mode="half_to_even")).to_pydict()["r"]
    assert got == [_I64_MIN + 8, 2**53 + 8]


def test_round_rejects_an_unknown_mode():
    with pytest.raises(bt.PlanError, match="half_to_even"):
        col("x").round(mode="bankers")


# --- boolean xor (bugfix) ------------------------------------------------------------------


def test_boolean_xor_is_boolean_and_null_propagating(duck):
    a = [True, True, False, False, None, True, None]
    b = [True, False, True, False, True, None, None]
    tbl = pa.table({"id": list(range(7)), "a": pa.array(a), "b": pa.array(b)})
    duck.register("t", tbl)
    out = bt.from_arrow(tbl).select("id", x=col("a") ^ col("b"), rx=True ^ col("b")).collect()
    assert _types(out) == {"id": pa.int64(), "x": pa.bool_(), "rx": pa.bool_()}
    # DuckDB has no BOOLEAN xor (`xor(BOOLEAN, BOOLEAN)` is a binder error); `<>` over two
    # booleans is exclusive-or with nulls propagating.
    assert_same(out, duck.sql("SELECT id, a <> b AS x, true <> b AS rx FROM t"))


def test_integer_xor_is_still_bitwise(duck):
    tbl = pa.table({"a": [6, -1, _I64_MAX, None], "b": [3, 0, -1, 5]})
    out = bt.from_arrow(tbl).select("a", "b", x=col("a") ^ col("b")).collect()
    assert _types(out)["x"] == pa.int64()
    assert_same(out, duck.sql("SELECT a, b, xor(a, b) AS x FROM tbl"))


# --- to_base(twos_complement=True) --------------------------------------------------------

_BASE = [0, 1, 5, -1, -8, 255, -256, _I64_MAX, _I64_MIN, None]


@pytest.fixture
def ints(duck):
    tbl = pa.table({"id": list(range(len(_BASE))), "n": pa.array(_BASE, pa.int64())})
    duck.register("n", tbl)
    return tbl


def test_twos_complement_binary_and_hex_match_duckdb_bin_and_hex(duck, ints):
    out = bt.from_arrow(ints).select(
        "id",
        b=col("n").to_base(2, twos_complement=True),
        h=col("n").to_base(16, twos_complement=True),
        plain=col("n").to_base(2),
    )
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT id, bin(n) AS b, upper(hex(n)) AS h, "
            "CASE WHEN n < 0 THEN '-' || bin(-(n::HUGEINT)) ELSE bin(n) END AS plain FROM n"
        ),
    )


@pytest.mark.parametrize("radix", [4, 8, 32])
def test_twos_complement_other_power_of_two_radixes(ints, radix):
    got = bt.from_arrow(ints).select(r=col("n").to_base(radix, twos_complement=True)).to_pydict()
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUV"

    def render(v: int) -> str:
        v &= 2**64 - 1
        out = ""
        while True:
            v, r = divmod(v, radix)
            out = digits[r] + out
            if v == 0:
                return out

    assert got["r"] == [None if v is None else render(v) for v in _BASE]


def test_twos_complement_needs_a_power_of_two():
    with pytest.raises(bt.PlanError, match="power of two"):
        col("n").to_base(10, twos_complement=True)


# --- is_in(nulls_equal=True) --------------------------------------------------------------


@pytest.mark.parametrize("values", [[1, None], [1], [None], [], [1.0, 3]])
def test_null_safe_membership_matches_is_not_distinct_from(duck, values):
    xs = [1, 2, None, 3, -0.0 if values == [1.0, 3] else 0]
    tbl = pa.table({"id": list(range(5)), "x": pa.array(xs, pa.float64())})
    duck.register("m", tbl)
    got = bt.from_arrow(tbl).select(
        "id",
        sql=col("x").is_in(values),
        safe=col("x").is_in(values, nulls_equal=True),
        not_in=~col("x").is_in(values, nulls_equal=True),
    )

    def literal(v: object) -> str:
        return "NULL" if v is None else repr(v)

    safe = " OR ".join(f"x IS NOT DISTINCT FROM {literal(v)}" for v in values) or "false"
    sql_in = f"x IN ({', '.join(literal(v) for v in values)})" if values else "false"
    assert_same(
        got.collect(),
        duck.sql(f"SELECT id, {sql_in} AS sql, ({safe}) AS safe, NOT ({safe}) AS not_in FROM m"),
    )


# --- engine-compatible hashes --------------------------------------------------------------


def test_spark_hash_documented_examples():
    """`pyspark/sql/functions/builtin.py::hash`: `hash('ABC')` is -757602832 and
    `hash('ABC', 'DEF')` is 599895104 under Spark's seed 42."""
    ds = bt.from_pydict({"c1": ["ABC"], "c2": ["DEF"]})
    got = ds.select(
        one=bt.hash_rows(col("c1"), seed=42, algorithm="murmur3"),
        two=bt.hash_rows(col("c1"), col("c2"), seed=42, algorithm="murmur3"),
        expr=col("c1").hash(42, algorithm="murmur3"),
    ).to_pydict()
    assert got == {"one": [-757602832], "two": [599895104], "expr": [-757602832]}


def test_iceberg_bucket_hash_matches_the_spec_and_reference_murmur3():
    """Iceberg spec, Appendix B: int/long 34 → 2017239379, "iceberg" → 1210000089. Every other
    value is held to the reference Murmur3_x86_32 (`mmh3`) over Iceberg's byte encoding."""
    mmh3 = pytest.importorskip("mmh3")
    import struct

    longs = [34, 0, -1, _I64_MAX, _I64_MIN, 17486, None]
    strings = ["iceberg", "", "a", "ab", "abc", "abcd", "abcde", "ünïcode", None]
    got = (
        bt.from_arrow(pa.table({"n": pa.array([*longs, 0, 0], pa.int64()), "s": pa.array(strings)}))
        .select(
            hn=col("n").hash(algorithm="iceberg"),
            hs=col("s").hash(algorithm="iceberg"),
            bn=col("n").hash_bucket(16, algorithm="iceberg"),
            bs=col("s").hash_bucket(7, algorithm="iceberg"),
        )
        .to_pydict()
    )
    want_n = [None if v is None else mmh3.hash(struct.pack("<q", v), 0) for v in longs] + [
        mmh3.hash(struct.pack("<q", 0), 0)
    ] * 2
    want_s = [None if v is None else mmh3.hash(v.encode(), 0) for v in strings]
    assert got["hn"] == want_n
    assert got["hs"] == want_s
    assert got["hn"][0] == 2017239379 and got["hs"][0] == 1210000089
    assert got["bn"] == [None if h is None else (h & 0x7FFFFFFF) % 16 for h in want_n]
    assert got["bs"] == [None if h is None else (h & 0x7FFFFFFF) % 7 for h in want_s]


def test_hash_algorithms_decline_what_their_engine_does_not_hash():
    with pytest.raises(bt.PlanError, match="exactly one"):
        bt.hash_rows(col("a"), col("b"), algorithm="iceberg")
    with pytest.raises(bt.PlanError, match="algorithm"):
        bt.hash_rows(col("a"), algorithm="sha1")
    with pytest.raises(bt.PlanError, match="no seed"):
        col("a").hash_bucket(4, seed=1, algorithm="iceberg")
    ds = bt.from_pydict({"f": [1.5]})
    with pytest.raises(Exception, match="iceberg"):
        ds.select(h=col("f").hash(algorithm="iceberg")).to_pydict()


def test_default_hash_is_unchanged_and_serializes_without_an_algorithm():
    ir = bt.hash_rows(col("a")).to_ir()
    assert "algorithm" not in ir
    got = bt.from_pydict({"a": [0, 1, -1]}).select(h=col("a").hash()).to_pydict()["h"]
    # The pinned `eval::hash::tests::golden_digests_are_stable` values.
    assert got == [7776768183763457969, 460422991341443459, -1695834481542859942]


# --- cum_*(reverse=, propagate_nulls=) ----------------------------------------------------

_RUN = [3, None, -2, 5, None, 0, 7]


@pytest.fixture
def run(duck):
    tbl = pa.table(
        {
            "id": list(range(len(_RUN))),
            "g": ["a", "a", "a", "b", "b", "b", "b"],
            "x": pa.array(_RUN, pa.int64()),
        }
    )
    duck.register("r", tbl)
    return tbl


@pytest.mark.parametrize(
    ("method", "agg"),
    [("cum_sum", "sum"), ("cum_min", "min"), ("cum_max", "max"), ("cum_prod", "product")],
)
def test_running_aggregates_reverse_and_propagate_nulls(duck, run, method, agg):
    def build(**kw):
        return getattr(col("x"), method)(partition_by="g", order_by="id", **kw)

    out = bt.from_arrow(run).select(
        "id",
        plain=build(),
        rev=build(reverse=True),
        keep=build(propagate_nulls=True),
        both=build(reverse=True, propagate_nulls=True),
    )
    over = "OVER (PARTITION BY g ORDER BY id ROWS BETWEEN"
    fwd = f"{agg}(x) {over} UNBOUNDED PRECEDING AND CURRENT ROW)"
    bwd = f"{agg}(x) {over} CURRENT ROW AND UNBOUNDED FOLLOWING)"
    assert_same(
        out.collect(),
        duck.sql(
            f"SELECT id, {fwd} AS plain, {bwd} AS rev, "
            f"CASE WHEN x IS NULL THEN NULL ELSE {fwd} END AS keep, "
            f"CASE WHEN x IS NULL THEN NULL ELSE {bwd} END AS both FROM r"
        ),
    )


# --- rank(method="average"|"max", propagate_nulls=) ---------------------------------------


def test_rank_average_max_and_null_rows(duck):
    xs = [3.0, 1.0, 3.0, None, float("nan"), -0.0, 0.0, 3.0]
    tbl = pa.table({"id": list(range(len(xs))), "x": pa.array(xs, pa.float64())})
    duck.register("k", tbl)
    out = bt.from_arrow(tbl).select(
        "id",
        mn=col("x").rank(),
        mx=col("x").rank("max"),
        avg=col("x").rank("average"),
        desc=col("x").rank("average", descending=True),
        keep=col("x").rank("average", propagate_nulls=True),
    )
    peers = "count(*) OVER (PARTITION BY x)"
    assert_same(
        out.collect(),
        duck.sql(
            f"SELECT id, rank() OVER (ORDER BY x) AS mn, "
            f"rank() OVER (ORDER BY x) + {peers} - 1 AS mx, "
            f"rank() OVER (ORDER BY x) + ({peers} - 1) / 2 AS avg, "
            f"rank() OVER (ORDER BY x DESC NULLS LAST) + ({peers} - 1) / 2 AS desc, "
            f"CASE WHEN x IS NULL THEN NULL ELSE rank() OVER (ORDER BY x) + ({peers} - 1) / 2 "
            "END AS keep FROM k"
        ),
    )


def test_rank_average_is_float64():
    out = bt.from_pydict({"x": [1, 1]}).select(r=col("x").rank("average")).collect()
    assert _types(out)["r"] == pa.float64()


# --- rolling variance: one NaN no longer poisons the partition (bugfix) --------------------


def test_rolling_std_nan_only_poisons_its_own_windows(duck):
    xs = [1.0, float("nan"), 3.0, 4.0, 6.0, 1e9, 1e9 + 2]
    tbl = pa.table({"id": list(range(len(xs))), "x": pa.array(xs, pa.float64())})
    got = (
        bt.from_arrow(tbl)
        .select(
            s=col("x").rolling_std(2, order_by="id"),
            v0=col("x").rolling_var(3, ddof=0, order_by="id"),
        )
        .to_pydict()
    )
    clean = pa.table({"id": [2, 3, 4, 5, 6], "x": [3.0, 4.0, 6.0, 1e9, 1e9 + 2]})
    duck.register("c", clean)
    want = dict(
        duck.sql(
            "SELECT id, stddev_samp(x) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) "
            "FROM c"
        ).fetchall()
    )
    assert got["s"][0] is None  # one value: no sample deviation
    assert math.isnan(got["s"][1]) and math.isnan(got["s"][2])  # the windows holding the NaN
    for i in (4, 5, 6):  # the windows after it, which the old centring turned to NaN
        assert got["s"][i] == pytest.approx(want[i], rel=1e-9)
    assert got["v0"][5] == pytest.approx(statistics_pvariance([4.0, 6.0, 1e9]), rel=1e-9)


def statistics_pvariance(values: list[float]) -> float:
    import statistics

    return statistics.pvariance(values)


# --- bare strings name columns in coalesce/greatest/least/arctan2 (breaking) ---------------


def test_bare_strings_are_column_names(duck):
    tbl = pa.table({"a": [None, 1.0, -0.0, None], "b": [2.0, None, 0.0, None]})
    duck.register("s", tbl)
    out = bt.from_arrow(tbl).select(
        c=bt.coalesce("a", "b"),
        g=bt.greatest("a", "b"),
        l=bt.least("a", col("b")),
        t=bt.arctan2("a", "b"),
        lit=bt.coalesce("a", bt.lit(9.0)),
    )
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT coalesce(a, b) AS c, greatest(a, b) AS g, least(a, b) AS l, "
            "atan2(a, b) AS t, coalesce(a, 9.0) AS lit FROM s"
        ),
    )


# --- over(order_by=[(key, descending, nulls_first)]) (bugfix) ------------------------------


def test_over_accepts_a_nulls_first_order_key(duck):
    xs = [None, 2, 1, None, 3]
    tbl = pa.table({"id": list(range(5)), "x": pa.array(xs, pa.int64())})
    duck.register("w", tbl)
    out = bt.from_arrow(tbl).select(
        "id",
        first=bt.row_number().over(order_by=[(col("x"), False, True), "id"]),
        last=bt.row_number().over(order_by=[("x", True, False), "id"]),
        run=col("x").sum().over(order_by=[("x", False, True), "id"]),
    )
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT id, row_number() OVER (ORDER BY x ASC NULLS FIRST, id) AS first, "
            "row_number() OVER (ORDER BY x DESC NULLS LAST, id) AS last, "
            "sum(x) OVER (ORDER BY x ASC NULLS FIRST, id) AS run FROM w"
        ),
    )


# --- bt.udf applied to a column fails loudly (bugfix) --------------------------------------


def test_udf_applied_to_an_expression_raises():
    shout = bt.udf(lambda s: s)
    with pytest.raises(bt.PlanError, match="not to a column expression"):
        shout(col("a"))
    with pytest.raises(bt.PlanError, match="not to a column expression"):
        shout(col("a").sum())
    # A Dataset is still the target it was built for.
    assert shout(bt.from_pydict({"a": [1]})).to_pydict() == {"a": [1]}
