"""The five relational operators `test_diff_operator_matrix.py` never reached.

That file pins every execution path against every edge-case input for the operators it
enumerates, and its docstring records the four wrong-answer bugs that lived in exactly
that gap. It does not enumerate all of them. Lowering each builder to IR shows five
`RelOp` tags with no cross-product coverage at all:

* ``unnest``     — `explode()`, including the `outer` / `index` (posexplode) variants;
* ``unpivot``    — wide-to-long reshape;
* ``sample``     — both the streaming `fraction` form and the breaker `n` form;
* ``asof_join``  — nearest-match, both directions, with and without a `by` group;
* ``range_join`` — the inequality-join rewrite.

Each already has a per-operator file, and each of those tests the operator on
`collect()` alone. That is the same shape the matrix docstring calls out: "the
per-operator tests each covered their own operator on its own happy path, and the
*combinations* were nobody's job". These five are cardinality-changing operators, which
is the property that makes a scheduling difference visible — a row that a spilled or
streamed path emits once too often, or drops at a morsel boundary, is a wrong answer
that no `collect()`-only test can see.

`test_diff_operator_matrix_coverage.py` is what keeps this list from going stale: it
lowers every builder in both files and fails if any `RelOp` tag is left uncovered, so a
new operator cannot land without a row here.

Input shapes are `base` / `empty` / `single` as in the sibling matrix, and the inputs
carry nulls, empty lists, duplicate keys, and unmatched rows on both join sides.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

from _harness import assert_same, assert_tables_equal

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")

# --- inputs -------------------------------------------------------------------------

#: Reshape input: a list column carrying both an empty list and a null (the two cases
#: `explode` treats identically by default and differently under `outer`), two same-typed
#: integer measure columns for `unpivot` with a null in a different row each, and a pair of
#: float measures carrying `-0.0`, `0.0` and NaN.
#:
#: The float pair is here because `unpivot` *moves values between columns*, which is where
#: a float identity quietly changes: `-0.0` canonicalized to `0.0`, or a NaN normalized to
#: a single payload, is invisible to a row-count assertion and to every integer column. It
#: is the same value class that split one group in two in `agg_float_key`.
RESHAPE = pa.table(
    {
        "id": pa.array([1, 2, 3, 4, 5], pa.int64()),
        "xs": pa.array([[1, 2], [], None, [3], [4, 5, 6]], pa.list_(pa.int64())),
        "w1": pa.array([10, 20, None, 40, 50], pa.int64()),
        "w2": pa.array([1, None, 3, 4, 5], pa.int64()),
        "f1": pa.array([-0.0, 0.0, float("nan"), None, 2.5], pa.float64()),
        "f2": pa.array([0.0, float("nan"), -0.0, 1.5, None], pa.float64()),
    }
)

#: ASOF left side: duplicate keys (5 twice), a null key that can never match, and a key
#: past the end of the right side so the backward/forward directions disagree on it.
ASOF_LEFT = pa.table(
    {
        "t": pa.array([1, 5, 5, 10, None, 20], pa.int64()),
        "g": pa.array(["a", "a", "b", "b", "a", None]),
        "lv": pa.array([1, 2, 3, 4, 5, 6], pa.int64()),
    }
)
ASOF_RIGHT = pa.table(
    {
        "t": pa.array([2, 5, 6, None, 15], pa.int64()),
        "g": pa.array(["a", "a", "b", "a", "b"]),
        "rv": pa.array(["p", "q", "r", "s", "u"]),
    }
)

#: Range-join sides: points against half-open intervals, with a null on each axis (never
#: matches), a point on an interval boundary (the half-open edge), and intervals that
#: overlap so a point matches more than one.
POINTS = pa.table(
    {
        "x": pa.array([0, 4, 5, 10, None, 25], pa.int64()),
        "pid": pa.array(["a", "b", "c", "d", "e", "f"]),
    }
)
INTERVALS = pa.table(
    {
        "lo": pa.array([0, 4, 10, None, 100], pa.int64()),
        "hi": pa.array([5, 20, 11, 50, 200], pa.int64()),
        "iid": pa.array(["i1", "i2", "i3", "i4", "i5"]),
    }
)


def _shapes(table: pa.Table) -> dict[str, pa.Table]:
    """The four input shapes every operator here is run against.

    `multibatch` repeats the input past the 16,384-row morsel, so `iter_batches()` yields
    more than one batch and `collect(spill=True)` has something to split. Without it the
    three "paths" are three names for a single batch, and a row dropped or double-emitted
    at a morsel boundary — the failure mode these cardinality-changing operators are most
    prone to — cannot be observed at all.
    """
    repeats = 16_384 // table.num_rows + 2 if table.num_rows else 1
    return {
        "base": table,
        "empty": table.slice(0, 0),
        "single": table.slice(0, 1),
        "multibatch": pa.concat_tables([table] * repeats),
    }


RESHAPE_SHAPES = _shapes(RESHAPE)
ASOF_LEFT_SHAPES = _shapes(ASOF_LEFT)
POINT_SHAPES = _shapes(POINTS)


def _stream(ds) -> pa.Table:
    """`iter_batches()` collected back into a table (the streaming scheduling)."""
    batches = list(ds.iter_batches())
    if not batches:
        return ds.collect().slice(0, 0)
    return pa.Table.from_batches(batches, schema=batches[0].schema)


def _assert_paths_agree(build, table: pa.Table) -> pa.Table:
    """`collect()`, `collect(spill=True)` and `iter_batches()` are one semantics.

    Returns the `collect()` result so a caller can go on to compare it to DuckDB.
    """
    oracle = build(bt.from_arrow(table)).collect()
    assert_tables_equal(build(bt.from_arrow(table)).collect(spill=True), oracle)
    assert_tables_equal(_stream(build(bt.from_arrow(table))), oracle)
    return oracle


# --- unnest (explode) and unpivot ---------------------------------------------------

#: operator -> (build, DuckDB SQL over `t`, or None where no SQL spelling matches).
RESHAPE_OPS: dict[str, tuple] = {
    # DuckDB's `unnest` in the select list drops a row whose list is empty or null, which
    # is `explode()`'s default. The projection pins column identity too: `explode` replaces
    # the list column in place rather than appending.
    "explode": (
        lambda d: d.explode("xs").select(bt.col("id"), bt.col("xs"), bt.col("w1")),
        "SELECT id, unnest(xs) AS xs, w1 FROM t",
    ),
    # `outer=True` keeps the empty-list and null-list rows with a NULL element (Spark
    # `explode_outer`); DuckDB has no one-expression spelling, so this is path-only.
    "explode_outer": (lambda d: d.explode("xs", outer=True), None),
    # `index` is Spark `posexplode` — the 0-based position within each row's own list, and
    # the thing that lets chunks be reassembled after a shuffle. A path that reorders or
    # restarts the counter is a wrong answer this column makes visible.
    "explode_posexplode": (lambda d: d.explode("xs", index="i"), None),
    "explode_aliased": (lambda d: d.explode("xs", alias="x"), None),
    # `INCLUDE NULLS` is required: DuckDB's bare UNPIVOT drops a melted NULL, Batcher keeps
    # it (Polars/pandas `melt` semantics). Measured on duckdb 1.5.4 — the default oracle
    # spelling silently disagrees on `w1`/`w2`'s nulls, so it is the wrong oracle here.
    "unpivot": (
        lambda d: d.select(bt.col("id"), bt.col("w1"), bt.col("w2")).unpivot(index=["id"]),
        "SELECT id, variable, value FROM t UNPIVOT INCLUDE NULLS (value FOR variable IN (w1, w2))",
    ),
    "unpivot_explicit_on": (
        lambda d: d.select(bt.col("id"), bt.col("w1"), bt.col("w2")).unpivot(
            index=["id"], on=["w1"], variable_name="var", value_name="val"
        ),
        "SELECT id, var, val FROM t UNPIVOT INCLUDE NULLS (val FOR var IN (w1))",
    ),
    # The float pair: `unpivot` must carry `-0.0`, `0.0` and NaN across the reshape as the
    # values they are. A `-0.0` arriving as `0.0` changes no row count and no integer column.
    "unpivot_float_measures": (
        lambda d: d.select(bt.col("id"), bt.col("f1"), bt.col("f2")).unpivot(index=["id"]),
        "SELECT id, variable, value FROM t UNPIVOT INCLUDE NULLS (value FOR variable IN (f1, f2))",
    ),
    # No `index`: every non-melted column is an identifier. Reshape over a single melted
    # column is the degenerate case a wide-to-long rewrite is most likely to mishandle.
    "unpivot_inferred_index": (
        lambda d: d.select(bt.col("id"), bt.col("w1")).unpivot(on=["w1"]),
        "SELECT id, variable, value FROM t UNPIVOT INCLUDE NULLS (value FOR variable IN (w1))",
    ),
}


@pytest.mark.parametrize("op", sorted(RESHAPE_OPS))
@pytest.mark.parametrize("shape", sorted(RESHAPE_SHAPES))
def test_reshape_paths_agree(op, shape):
    """Every scheduling of a cardinality-changing reshape emits the same rows."""
    build, _ = RESHAPE_OPS[op]
    _assert_paths_agree(build, RESHAPE_SHAPES[shape])


@pytest.mark.parametrize("op", sorted(o for o, (_, sql) in RESHAPE_OPS.items() if sql))
@pytest.mark.parametrize("shape", sorted(RESHAPE_SHAPES))
def test_reshape_matches_duckdb(duck, op, shape):
    """...and `collect()` matches the external oracle on every edge-case input."""
    build, sql = RESHAPE_OPS[op]
    table = RESHAPE_SHAPES[shape]
    duck.register("t", table)
    assert_same(build(bt.from_arrow(table)).collect(), duck.sql(sql))


@pytest.mark.parametrize("shape", sorted(RESHAPE_SHAPES))
def test_unpivot_carries_signed_zero_and_nan_through_the_reshape(shape):
    """`-0.0` must arrive as `-0.0`, and a NaN as a NaN, on every path.

    Neither the DuckDB comparison nor the path-vs-path one can assert this: `assert_same`
    and `assert_tables_equal` both run values through `_harness._coerce`, which folds
    `-0.0` into `0` and every NaN into one sentinel *on purpose*, so that SQL's
    "all NaNs are one value" grouping semantics compare correctly. That normalization is
    right for those comparisons and blind for this one, so the sign bit is read directly.
    """
    table = RESHAPE_SHAPES[shape]
    plan = lambda d: d.select(bt.col("id"), bt.col("f1"), bt.col("f2")).unpivot(index=["id"])  # noqa: E731

    def signature(values: list[float | None]) -> tuple[int, int, int, int]:
        """(negative zeros, positive zeros, NaNs, nulls) — the classes `_coerce` erases."""
        return (
            sum(1 for v in values if v == 0.0 and v is not None and math.copysign(1.0, v) < 0),
            sum(1 for v in values if v == 0.0 and v is not None and math.copysign(1.0, v) > 0),
            sum(1 for v in values if v is not None and math.isnan(v)),
            sum(1 for v in values if v is None),
        )

    expected = signature(table.column("f1").to_pylist() + table.column("f2").to_pylist())
    for out in (
        plan(bt.from_arrow(table)).collect(),
        plan(bt.from_arrow(table)).collect(spill=True),
        _stream(plan(bt.from_arrow(table))),
    ):
        got = signature(out.column("value").to_pylist())
        assert got == expected, (
            f"unpivot changed the float value classes "
            f"(neg-zero, pos-zero, nan, null): {got} != {expected}"
        )


@pytest.mark.parametrize("shape", sorted(RESHAPE_SHAPES))
def test_explode_outer_keeps_exactly_the_rows_default_explode_drops(shape):
    """The `outer` contract, stated as the difference between the two forms.

    A document that chunked to nothing disappearing along with its id is the bug `outer`
    exists to prevent, so the row count relationship is the contract: `outer` emits one
    row per element, plus exactly one row for each null-or-empty list.
    """
    table = RESHAPE_SHAPES[shape]
    ds = bt.from_arrow(table)
    inner = ds.explode("xs").collect().num_rows
    outer = ds.explode("xs", outer=True).collect().num_rows
    lists = table.column("xs").to_pylist()
    empty_or_null = sum(1 for v in lists if v is None or len(v) == 0)
    assert outer == inner + empty_or_null, (
        f"outer explode emitted {outer} rows; expected {inner} + {empty_or_null} kept rows"
    )


@pytest.mark.parametrize("shape", sorted(RESHAPE_SHAPES))
def test_posexplode_index_is_each_element_position_in_its_own_list(shape):
    """`index` must be the position within the row's list, not a running counter.

    A global counter passes any row-count assertion and any unordered comparison, and is
    wrong the moment two rows are exploded — so the positions are asserted directly, on
    all three paths.

    The expectation is built from the input rather than from a per-`id` lookup: `id` is
    not unique in the `multibatch` shape, and a lookup keyed on it silently compares one
    row's positions against 3,278 rows' worth of them.
    """
    table = RESHAPE_SHAPES[shape]
    expected: dict[int, list[int | None]] = {}
    for row_id, xs in zip(
        table.column("id").to_pylist(), table.column("xs").to_pylist(), strict=True
    ):
        # A null or empty list is kept by `outer=True` as one row with a NULL position.
        positions = [None] if xs is None or len(xs) == 0 else list(range(len(xs)))
        expected.setdefault(row_id, []).extend(positions)
    for positions in expected.values():
        positions.sort(key=lambda p: (p is None, p))

    plan = lambda d: d.explode("xs", outer=True, index="i")  # noqa: E731
    for out in (
        plan(bt.from_arrow(table)).collect(),
        plan(bt.from_arrow(table)).collect(spill=True),
        _stream(plan(bt.from_arrow(table))),
    ):
        actual: dict[int, list[int | None]] = {}
        d = out.to_pydict()
        for row_id, idx in zip(d["id"], d["i"], strict=True):
            actual.setdefault(row_id, []).append(idx)
        for positions in actual.values():
            positions.sort(key=lambda p: (p is None, p))
        assert actual == expected, (
            f"posexplode positions disagree with the input's list lengths: "
            f"{ {k: v for k, v in actual.items() if expected.get(k) != v} }"
        )


# --- sample -------------------------------------------------------------------------

#: `sample` has no SQL oracle: it keeps rows by a stable seeded hash, which is a Batcher
#: contract rather than a shared one. What *is* checkable is that the contract holds on
#: every path — and the contract is the reason the operator is safe to distribute.
SAMPLE_OPS: dict[str, object] = {
    "sample_fraction": lambda d: d.sample(0.5, seed=7),
    "sample_fraction_all": lambda d: d.sample(1.0, seed=7),
    "sample_fraction_none": lambda d: d.sample(0.0, seed=7),
    "sample_n": lambda d: d.sample(n=2, seed=7),
    "sample_n_over_cardinality": lambda d: d.sample(n=99, seed=7),
    "sample_frac_pandas_spelling": lambda d: d.sample(frac=0.5, random_state=7),
}


@pytest.mark.parametrize("op", sorted(SAMPLE_OPS))
@pytest.mark.parametrize("shape", sorted(RESHAPE_SHAPES))
def test_sample_paths_agree(op, shape):
    """A seeded sample is partition-independent, so scheduling cannot change the result.

    This is the property that makes `sample` correct distributed, and the one a
    row-count- or fraction-based implementation would silently break: it would still
    return "about half the rows" on every path, and a different half on each.
    """
    _assert_paths_agree(SAMPLE_OPS[op], RESHAPE_SHAPES[shape])


@pytest.mark.parametrize("op", sorted(SAMPLE_OPS))
@pytest.mark.parametrize("shape", sorted(RESHAPE_SHAPES))
def test_sample_returns_a_subset_of_its_input(op, shape):
    """Sampling selects rows; it never fabricates, duplicates, or alters one."""
    table = RESHAPE_SHAPES[shape]
    out = SAMPLE_OPS[op](bt.from_arrow(table)).collect()
    assert out.column_names == table.column_names
    original = [str(r) for r in table.to_pylist()]
    for row in (str(r) for r in out.to_pylist()):
        assert row in original, f"sampled row not present in the input: {row}"
    assert out.num_rows <= table.num_rows


@pytest.mark.parametrize("shape", sorted(RESHAPE_SHAPES))
def test_sample_n_keeps_exactly_n_rows_when_the_input_has_them(shape):
    """The `n` form is an exact count, not an approximation."""
    table = RESHAPE_SHAPES[shape]
    out = bt.from_arrow(table).sample(n=2, seed=7).collect()
    assert out.num_rows == min(2, table.num_rows)


@pytest.mark.parametrize("shape", sorted(RESHAPE_SHAPES))
def test_the_same_seed_samples_the_same_rows_twice(shape):
    """Reproducibility is the documented reason `seed` exists."""
    table = RESHAPE_SHAPES[shape]
    first = bt.from_arrow(table).sample(0.5, seed=11).collect()
    second = bt.from_arrow(table).sample(0.5, seed=11).collect()
    assert_tables_equal(second, first)


def test_a_different_seed_can_select_a_different_subset():
    """The seed must actually reach the hash — a sample that ignores it is reproducible
    for the wrong reason, and every determinism assertion above would still pass."""
    wide = pa.table({"id": pa.array(list(range(200)), pa.int64())})
    subsets = {
        tuple(bt.from_arrow(wide).sample(0.5, seed=s).collect().column("id").to_pylist())
        for s in range(6)
    }
    assert len(subsets) > 1, "every seed selected the same rows; the seed is being ignored"


# --- asof join ----------------------------------------------------------------------

ASOF_OPS: dict[str, tuple] = {
    # DuckDB spells the nearest-backward match `ASOF LEFT JOIN ... ON l.t >= r.t`.
    "asof_backward": (
        lambda left, right: left.join_asof(right, on="t", direction="backward").select(
            bt.col("t"), bt.col("lv"), bt.col("rv")
        ),
        "SELECT l.t AS t, l.lv AS lv, r.rv AS rv FROM l ASOF LEFT JOIN r ON l.t >= r.t",
    ),
    "asof_forward": (
        lambda left, right: left.join_asof(right, on="t", direction="forward").select(
            bt.col("t"), bt.col("lv"), bt.col("rv")
        ),
        "SELECT l.t AS t, l.lv AS lv, r.rv AS rv FROM l ASOF LEFT JOIN r ON l.t <= r.t",
    ),
    # The `by` group is an exact match *inside* the nearest match: a row may only pair
    # with a right row of the same group, however much nearer an out-of-group row is.
    "asof_by_group": (
        lambda left, right: left.join_asof(right, on="t", by="g", direction="backward").select(
            bt.col("t"), bt.col("g"), bt.col("lv"), bt.col("rv")
        ),
        "SELECT l.t AS t, l.g AS g, l.lv AS lv, r.rv AS rv "
        "FROM l ASOF LEFT JOIN r ON l.g = r.g AND l.t >= r.t",
    ),
}


@pytest.mark.parametrize("op", sorted(ASOF_OPS))
@pytest.mark.parametrize("shape", sorted(ASOF_LEFT_SHAPES))
@pytest.mark.parametrize("right_empty", [False, True])
def test_asof_paths_agree(op, shape, right_empty):
    """Every scheduling of the nearest-match join agrees, including against an empty side."""
    build, _ = ASOF_OPS[op]
    left = ASOF_LEFT_SHAPES[shape]
    right = ASOF_RIGHT.slice(0, 0) if right_empty else ASOF_RIGHT

    def plan(d):
        return build(d, bt.from_arrow(right))

    _assert_paths_agree(plan, left)


@pytest.mark.parametrize("op", sorted(ASOF_OPS))
@pytest.mark.parametrize("shape", sorted(ASOF_LEFT_SHAPES))
@pytest.mark.parametrize("right_empty", [False, True])
def test_asof_matches_duckdb(duck, op, shape, right_empty):
    """...and matches DuckDB's own ASOF join on each of them."""
    build, sql = ASOF_OPS[op]
    left = ASOF_LEFT_SHAPES[shape]
    right = ASOF_RIGHT.slice(0, 0) if right_empty else ASOF_RIGHT
    duck.register("l", left)
    duck.register("r", right)
    out = build(bt.from_arrow(left), bt.from_arrow(right)).collect()
    assert_same(out, duck.sql(sql))


@pytest.mark.parametrize("shape", sorted(ASOF_LEFT_SHAPES))
@pytest.mark.parametrize("direction", ["backward", "forward"])
def test_asof_is_left_style_so_every_left_row_survives(shape, direction):
    """An unmatched left row is null-extended, never dropped.

    A nearest-match implementation that filters instead of null-extending loses exactly
    the rows with no match — the null key and the out-of-range key here — which an
    unordered comparison against a *correspondingly wrong* expectation would not catch.
    """
    left = ASOF_LEFT_SHAPES[shape]
    out = (
        bt.from_arrow(left)
        .join_asof(bt.from_arrow(ASOF_RIGHT), on="t", direction=direction)
        .collect()
    )
    assert out.num_rows == left.num_rows
    assert sorted(map(str, out.column("lv").to_pylist())) == sorted(
        map(str, left.column("lv").to_pylist())
    )


@pytest.mark.parametrize("shape", sorted(ASOF_LEFT_SHAPES))
def test_asof_null_keys_never_match(shape):
    """NULL is not "nearest" to anything — it matches no right row, in either direction."""
    left = ASOF_LEFT_SHAPES[shape]
    for direction in ("backward", "forward"):
        out = (
            bt.from_arrow(left)
            .join_asof(bt.from_arrow(ASOF_RIGHT), on="t", direction=direction)
            .collect()
            .to_pydict()
        )
        for key, matched in zip(out["t"], out["rv"], strict=True):
            if key is None:
                assert matched is None, f"a NULL key matched {matched!r}"


# --- range join ---------------------------------------------------------------------

#: The inequality-join rewrite only fires through a SQL predicate, so these are spelled as
#: SQL. Two conjuncts is the interval-containment shape; one is the sorted-suffix scan.
RANGE_QUERIES: dict[str, str] = {
    "interval_containment": ("SELECT pid, iid FROM pt, iv WHERE pt.x >= iv.lo AND pt.x < iv.hi"),
    "single_inequality": "SELECT pid, iid FROM pt, iv WHERE pt.x < iv.hi",
    "both_closed": "SELECT pid, iid FROM pt, iv WHERE pt.x >= iv.lo AND pt.x <= iv.hi",
    "reversed_operands": "SELECT pid, iid FROM pt, iv WHERE iv.lo <= pt.x AND iv.hi > pt.x",
}


def _ir_ops(ir: object) -> set[str]:
    """Every ``op`` tag appearing in a lowered plan."""
    found: set[str] = set()
    if isinstance(ir, dict):
        op = ir.get("op")
        if isinstance(op, str):
            found.add(op)
        for value in ir.values():
            found |= _ir_ops(value)
    elif isinstance(ir, list):
        for value in ir:
            found |= _ir_ops(value)
    return found


@pytest.mark.parametrize("name", sorted(RANGE_QUERIES))
@pytest.mark.parametrize("shape", sorted(POINT_SHAPES))
def test_range_join_paths_agree(name, shape):
    """Every scheduling of the range join agrees on every point-side shape."""
    points = POINT_SHAPES[shape]
    query = RANGE_QUERIES[name]

    # `bt.sql` binds its own inputs, so each path gets a freshly built plan rather than a
    # rebound one — `_assert_paths_agree` takes a `from_arrow` builder and does not fit.
    def build():
        return bt.sql(query, pt=points, iv=INTERVALS)

    oracle = build().collect()
    assert_tables_equal(build().collect(spill=True), oracle)
    assert_tables_equal(_stream(build()), oracle)


@pytest.mark.parametrize("name", sorted(RANGE_QUERIES))
@pytest.mark.parametrize("shape", sorted(POINT_SHAPES))
def test_range_join_matches_duckdb(duck, name, shape):
    """...and matches DuckDB, which runs the same predicate over a cross product."""
    points = POINT_SHAPES[shape]
    query = RANGE_QUERIES[name]
    duck.register("pt", points)
    duck.register("iv", INTERVALS)
    assert_same(bt.sql(query, pt=points, iv=INTERVALS).collect(), duck.sql(query))


@pytest.mark.parametrize("name", sorted(RANGE_QUERIES))
def test_the_range_join_rewrite_actually_fires(name):
    """A rewrite that silently declines passes every result assertion while changing
    nothing, so the plan shape is asserted alongside the rows."""
    from batcher.kyber.optimizer import optimize_logical

    ds = bt.sql(RANGE_QUERIES[name], pt=POINTS, iv=INTERVALS)
    ops = _ir_ops(optimize_logical(ds._plan).to_ir())
    assert "range_join" in ops, f"expected a range_join in the plan, got {sorted(ops)}"
