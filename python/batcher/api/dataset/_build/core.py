"""Plan-construction helpers behind the thinner `Dataset` methods.

`Dataset` stays a thin fluent builder (the v2 maintainability contract): its heavier
methods (`window`) and the frame-level convenience sugar (`fill_null`/`drop_nulls`/
`cast`) delegate their bodies here, mirroring how terminal ops live in `terminal.py`.
These functions take the `Dataset` and return a new one via its own public methods,
so they add no new IR — the sugar lowers to existing `select`/`with_columns`/`filter`.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.api._join_helpers import _as_key_expr
from batcher.plan.expr_ir import Col, nullif, when
from batcher.plan.expr_ir.selectors import Selector, expand_selectors
from batcher.plan.ir_tags import RUNNING_AGGREGATES, WINDOW_AGGREGATES, WINDOW_FRAMEABLE
from batcher.plan.logical import (
    Distinct,
    Sample,
    SortKeySpec,
    Unnest,
    Unpivot,
    Window,
    WindowFrame,
    WindowFuncSpec,
)

if TYPE_CHECKING:
    from batcher.api.dataset.frame import Dataset
    from batcher.plan.expr_ir import Expr


def expand_selector_expr(ds: Dataset, expr: Expr) -> list[tuple[str, Expr]]:
    """Expand a selector-bearing expression against `ds`'s columns and schema.

    The one place the relational layer resolves a `Selector` into concrete columns,
    so `select`, `with_columns`, and `drop` all agree on what a selector means.
    """
    return expand_selectors(expr, ds._plan.available_columns(), ds._plan.available_schema())


def selector_columns(ds: Dataset, selector: Selector) -> list[str]:
    """The columns of `ds` a bare `Selector` matches, in the dataset's column order."""
    return selector.matched_columns(ds._plan.available_columns(), ds._plan.available_schema())


def build_window(
    ds: Dataset,
    *,
    partition_by: list[str | Expr],
    order_by: list[str | tuple[str, bool] | Expr],
    functions: dict[str, str | tuple[str, str]],
    frame: tuple[int | None, int | None] | tuple[int | None, int | None, str] | None,
) -> Dataset:
    """Construct a `Window` node (see `Dataset.window` for the contract)."""
    if not functions:
        raise PlanError("window() requires at least one function")

    wframe = WindowFrame(*frame) if frame is not None else None
    part_keys = tuple(_as_key_expr(k) for k in partition_by)

    order_specs: list[SortKeySpec] = []
    for key in order_by:
        # A bare name or expression, `(name, descending)`, or `(name, descending,
        # nulls_first)` — the third element places the nulls and defaults to SQL's last.
        name, desc, *rest = key if isinstance(key, tuple) else (key, False)
        nulls_first = bool(rest[0]) if rest else False
        order_specs.append(SortKeySpec(_as_key_expr(name), bool(desc), nulls_first))

    specs: list[WindowFuncSpec] = []
    for alias, spec in functions.items():
        if isinstance(spec, str):
            specs.append(WindowFuncSpec(spec, None, alias))
        elif isinstance(spec, tuple) and spec and spec[0] == "ntile":
            # ntile: a no-input ranking function whose bucket count rides in `offset`
            # — spelled ``("ntile", n)`` since it takes a count, not a column.
            if len(spec) != 2:
                raise PlanError(f"window function {alias!r}: ntile takes ('ntile', n)")
            specs.append(WindowFuncSpec("ntile", None, alias, int(spec[1]), None))
        elif isinstance(spec, tuple):
            # (func, column) or, for lag/lead, (func, column, offset).
            if len(spec) == 2:
                func, column = spec
                offset = 1
            elif len(spec) == 3:
                func, column, offset = spec
            else:
                raise PlanError(
                    f"window function {alias!r} must be (func, column) or (func, column, offset)"
                )
            # `mean` is the canonical DataFrame spelling (matches Expr.mean()); the
            # window engine names the aggregate `avg`, so accept both here.
            if func == "mean":
                func = "avg"
            # Dropping the frame is right for a function that *has* no frame in SQL either
            # (ranking, lag/lead, the fills), and wrong for an aggregate, where SQL defines
            # the framed answer and returning the unframed one is a wrong result rather than
            # a missing feature. `stddev`, `var` and `count_distinct` are exactly that case:
            # `window(frame=(-1, 0), functions={"w": ("stddev", "f")})` silently returned the
            # *running* standard deviation, which is what DuckDB returns for the query without
            # the frame. Refuse instead, and name the frameable set so the error is actionable.
            if wframe is not None and func in WINDOW_AGGREGATES - WINDOW_FRAMEABLE:
                raise PlanError(
                    f"window aggregate {func!r} does not support an explicit frame yet — "
                    f"drop the frame for the running form, or use one of "
                    f"{sorted(WINDOW_AGGREGATES & WINDOW_FRAMEABLE)}"
                )
            fn_frame = wframe if func in WINDOW_FRAMEABLE else None
            specs.append(WindowFuncSpec(func, _as_key_expr(column), alias, int(offset), fn_frame))
        else:
            raise PlanError(
                f"window function {alias!r} must be a string or (func, column[, offset]) tuple"
            )

    return ds._derive(Window(ds._plan, part_keys, tuple(order_specs), tuple(specs)))


_RANDOM_MODULUS = 2147483647  # 2^31 - 1 (prime): the uniform denominator.


def build_with_random(ds: Dataset, name: str, *, seed: int, normal: bool) -> Dataset:
    """Add a reproducible pseudo-random column keyed by ``(seed, row index)``.

    Pure desugaring: a `with_row_index` provides a stable per-row key, an xxhash of
    ``seed:salt:index`` provides a well-distributed integer, and that maps to a
    uniform ``[0, 1)`` (or, with `normal`, a standard normal via Box-Muller from two
    independent hashes). Keyed on the stable index, so it is reproducible and matches
    on the single-node and parallel paths.
    """
    from batcher.plan.expr_ir import Col, lit
    from batcher.plan.functions.string import concat_ws

    rid = f"__bc_random_idx_{name}"

    def uniform_int(salt: str) -> Expr:
        keyed = concat_ws(":", lit(f"{seed}:{salt}"), Col(rid).cast("string"))
        h = keyed.str.xxhash64()  # well-distributed Int64
        return ((h % _RANDOM_MODULUS) + _RANDOM_MODULUS) % _RANDOM_MODULUS  # [0, M)

    if normal:
        import math

        u1 = (uniform_int("a") + 1).cast("float64") / float(_RANDOM_MODULUS + 1)  # (0, 1]
        u2 = uniform_int("b").cast("float64") / float(_RANDOM_MODULUS)  # [0, 1)
        expr = (-2.0 * u1.ln()).sqrt() * (2.0 * math.pi * u2).cos()  # Box-Muller
    else:
        expr = uniform_int("u").cast("float64") / float(_RANDOM_MODULUS)  # [0, 1)
    return ds.with_row_index(rid).with_columns(**{name: expr}).drop(rid)


def split_key(ds: Dataset, key: list[str] | None, seed: int) -> Expr:
    """A reproducible uniform ``[0, 1)`` per row, derived from the row's **content**.

    Deliberately not `with_random`, which keys on `with_row_index`: a row index is a
    `RowId` node, and `RowId` has no distributed implementation and is not streamable,
    so a split built on it would silently pin an ML pipeline to one node. Hashing the
    row's own values instead makes the split a pure row-wise `Filter` — the shape the
    distributed executor treats as embarrassingly parallel and the streaming engine
    accepts — and makes it *partition-independent*: the same row lands in the same part
    however the data is laid out, which `Dataset.sample` already relies on.

    With `key` given, only those columns are hashed. That is better on three counts and
    is what to reach for on a real corpus: the split survives a schema change (adding or
    recomputing a feature column does not reshuffle rows between train and test), and it
    hashes one column rather than all of them. Without `key` the split is still correct
    and reproducible — just tied to every value in the row.
    """
    from batcher.plan.expr_ir.constructors import hash_rows

    columns = key if key is not None else list(ds.columns)
    # A typed row hash: no per-value string rendering, and no dependence on how a float
    # prints. `seed` keys the digest, so a new seed is a new split.
    digest = hash_rows(*(Col(c) for c in columns), seed=seed)  # Int64, may be negative
    positive = ((digest % _RANDOM_MODULUS) + _RANDOM_MODULUS) % _RANDOM_MODULUS
    return positive.cast("float64") / float(_RANDOM_MODULUS)


def build_random_split(
    ds: Dataset, fractions: list[float], *, seed: int, key: list[str] | None = None
) -> list[Dataset]:
    """Partition rows into disjoint random subsets sized by `fractions`.

    Each row's uniform ``[0, 1)`` (`split_key`) is compared against the cumulative
    boundaries of `fractions`, so the parts are disjoint, cover every row, and are
    stable across runs, partitions, and the parallel/distributed/streaming paths. Rows
    land in a part in expectation, not exactly: the sizes are binomial around
    ``fraction * n``, as with any hash-keyed split.

    Each part is a `Filter` over the input — no extra column, no row index, no shuffle.
    """
    if not fractions:
        raise PlanError("random_split(): fractions must be non-empty")
    if any(f <= 0 for f in fractions):
        raise PlanError(f"random_split(): every fraction must be > 0, got {fractions}")
    total = sum(fractions)
    if abs(total - 1.0) > 1e-9:
        raise PlanError(f"random_split(): fractions must sum to 1.0, got {total}")
    if key is not None:
        unknown = [c for c in key if c not in ds.columns]
        if unknown:
            raise PlanError(f"random_split(): unknown key column(s) {unknown}")

    if len(fractions) == 1:
        return [ds]  # the whole dataset; no predicate to evaluate

    parts: list[Dataset] = []
    lo = 0.0
    last = len(fractions) - 1
    for i, fraction in enumerate(fractions):
        hi = lo + fraction
        u = split_key(ds, key, seed)
        # Only the interior parts need both bounds: the first is bounded below by 0 and
        # the last above by 1, and the last takes everything left so float drift on the
        # cumulative boundaries can never drop a row.
        if i == 0:
            keep = u < hi
        elif i == last:
            keep = u >= lo
        else:
            keep = (u >= lo) & (u < hi)
        parts.append(ds.filter(keep))
        lo = hi
    return parts


def build_train_test_split(
    ds: Dataset, test_size: float, *, seed: int, key: list[str] | None = None
) -> tuple[Dataset, Dataset]:
    """Split into a train and a test `Dataset` — `test_size` is the test fraction."""
    if not 0.0 < test_size < 1.0:
        raise PlanError(f"train_test_split(): test_size must be in (0, 1), got {test_size}")
    train, test = build_random_split(ds, [1.0 - test_size, test_size], seed=seed, key=key)
    return train, test


# Python builtins and NumPy/pandas dtype objects accepted where a Batcher dtype name
# is expected, so `astype(float)` and `astype({"x": int})` read the way pandas spells
# them. Widths follow the FFI boundary's normalization (Int*/Float* → 64-bit).
_PY_TYPE_DTYPES: dict[Any, str] = {
    int: "int64",
    float: "float64",
    str: "string",
    bool: "boolean",
    bytes: "binary",
}


def _dtype_name(dtype: Any) -> str:
    """Normalize a dtype specification to the string the IR expects."""
    if isinstance(dtype, str):
        return dtype
    if dtype in _PY_TYPE_DTYPES:
        return _PY_TYPE_DTYPES[dtype]
    # A pyarrow DataType (or anything else that names itself) stringifies to the
    # same vocabulary the cast expression already understands.
    name = getattr(dtype, "__name__", None) or str(dtype)
    if name in _PY_TYPE_DTYPES.values() or not isinstance(dtype, type):
        return name
    raise PlanError(
        f"cast(): cannot interpret {dtype!r} as a dtype; pass a dtype name such as "
        "'int64', a Python type (int/float/str/bool), or a pyarrow DataType"
    )


def build_cast(ds: Dataset, dtypes: str | type | dict[str, Any], *, strict: bool = True) -> Dataset:
    """Cast columns — one dtype for every column, or per-column via a dict.

    A dtype is a Batcher dtype name, a Python type (``int``, ``float``, ``str``,
    ``bool``), or a pyarrow `DataType`. `strict=False` selects ``TRY_CAST`` (NULL on
    an unconvertible value).
    """

    def _cast(name: str, dtype: Any) -> Expr:
        e = Col(name)
        target = _dtype_name(dtype)
        return e.cast(target) if strict else e.try_cast(target)

    if isinstance(dtypes, dict):
        unknown = set(dtypes) - set(ds.columns)
        if unknown:
            raise PlanError(f"cast(): unknown column(s) {sorted(unknown)}")
        return ds.with_columns(**{c: _cast(c, t) for c, t in dtypes.items()})
    return ds.with_columns(**{c: _cast(c, dtypes) for c in ds.columns})


@dataclass(frozen=True, slots=True)
class RepartitionSpec:
    """How the next `write` should lay out its output files.

    - `num_files`: produce exactly this many files (rows split evenly).
    - `by`: Hive-partition the output by these column values (one subtree per value).
    - `target_size_mb`: coalesce into files of roughly this many megabytes.

    `num_files` and `target_size_mb` are resolved to a per-file row cap *after* the
    result materializes (so no extra counting pass), and may combine with `by`.
    """

    num_files: int | None = None
    by: tuple[str, ...] = ()
    target_size_mb: float | None = None


def _all_bounded(ds: Dataset) -> bool:
    """Whether every source behind `ds` is bounded."""
    from batcher.io.source import is_bounded

    return all(is_bounded(s) for s in ds._sources)


def build_distinct(
    ds: Dataset,
    subset: list[str],
    keep: str,
    order_by: str | list[str] | list[tuple[str, bool]] | None,
) -> Dataset:
    """Keep one row per `subset` key, as a single mergeable `Distinct` reduction.

    `keep="first"`/`"last"` keep the row minimizing/maximizing `order_by`; `keep="any"`
    keeps an arbitrary one, expressed as *no* ordering at all.

    This used to lower to ``row_number() OVER (PARTITION BY subset ORDER BY ...) = 1``,
    which is the same answer computed the expensive way: a full sort of every partition and
    a materialized rank column over the whole relation, to select one row from each. The
    `Distinct` node is one hash pass and one gather, it stays bounded under spill, and it
    pre-reduces on the map side of a shuffle instead of sending every row across the wire.

    `keep="any"` passing no ordering is not a loosening. Ordering by the subset keys made
    every row in a partition a tie, so which one survived was already unspecified — the old
    docstring's claim that the choice was "deterministic and partition-independent" was not
    something the plan could deliver.
    """
    unknown = set(subset) - set(ds.columns)
    if unknown:
        raise PlanError(f"distinct(): unknown subset column(s) {sorted(unknown)}")
    if keep not in ("first", "last", "any"):
        raise PlanError(f"distinct(): keep must be 'first'/'last'/'any', got {keep!r}")
    # Refused here, by name, rather than at execution as a generic breaker. The generic
    # message advises "restructure to ... a single top-level aggregate / distinct", which
    # is the thing the caller just wrote -- a keyed dedup holds one row per key for the life
    # of the query, and no restructuring of it streams.
    if not _all_bounded(ds):
        raise PlanError(
            "distinct(subset=...) cannot stream: keeping one row per key across an "
            "unbounded input needs the key set held for the life of the query, and which "
            "row wins is decided by an ordering over rows that have not arrived. Use "
            "drop_duplicates_within_watermark(subset, event_time=..., lateness=...), whose "
            "state the watermark bounds, or distinct() with no subset when the whole row "
            "is the key."
        )

    order: list[tuple[str, bool]] = []
    if keep != "any":
        if order_by is None:
            raise PlanError(f"distinct(keep={keep!r}) requires order_by")
        keys = [order_by] if isinstance(order_by, str) else list(order_by)
        descending = keep == "last"
        order = [(k, descending) if isinstance(k, str) else (k[0], k[1] ^ descending) for k in keys]
    unknown_order = {name for name, _ in order} - set(ds.columns)
    if unknown_order:
        raise PlanError(f"distinct(): unknown order_by column(s) {sorted(unknown_order)}")

    # `nulls_first=False` for both directions, matching the window lowering this replaced
    # (and so the differential results it was checked against): a null ordering value loses
    # to any real one whichever way the comparison runs.
    specs = tuple(SortKeySpec(Col(name), descending=desc) for name, desc in order)
    return ds._derive(Distinct(ds._plan, tuple(subset), specs))


def build_explode(
    ds: Dataset,
    column: str,
    alias: str | None,
    *,
    outer: bool = False,
    index: str | None = None,
) -> Dataset:
    """Construct an `Unnest` node (see `Dataset.explode` for the contract)."""
    if column not in ds.columns:
        raise PlanError(f"explode(): unknown column {column!r}")
    return ds._derive(Unnest(ds._plan, column, alias or column, outer, index))


def build_unnest(ds: Dataset, columns: str | list[str]) -> Dataset:
    """Expand each struct `column` into its fields as top-level columns (Polars
    ``unnest``; Spark ``select("s.*")``). Composes ``struct.field`` extraction — no
    new IR. See `Dataset.unnest` for the contract."""
    import pyarrow as pa

    from batcher.plan.expr_ir import col

    names = [columns] if isinstance(columns, str) else list(columns)
    schema = ds.schema
    fields_of: dict[str, list[str]] = {}
    for name in names:
        if name not in ds.columns:
            raise PlanError(f"unnest(): unknown column {name!r}")
        ftype = schema.field(name).type
        if not pa.types.is_struct(ftype):
            raise PlanError(f"unnest(): column {name!r} is not a struct (got {ftype})")
        fields_of[name] = [ftype.field(i).name for i in range(ftype.num_fields)]

    # Output column order: each struct expands in place to its fields; others stay.
    final: list[str] = []
    for c in ds.columns:
        final.extend(fields_of[c]) if c in fields_of else final.append(c)
    if len(final) != len(set(final)):
        # One counting pass rather than a `list.count()` per column: a struct-heavy
        # relation can expand to thousands of output names, and the error message is
        # not the place to spend quadratic time.
        dup = sorted(n for n, k in Counter(final).items() if k > 1)
        raise PlanError(f"unnest(): output columns collide: {dup} (rename before unnesting)")

    derived = {
        fname: col(sname).struct.field(fname)
        for sname, fnames in fields_of.items()
        for fname in fnames
    }
    return ds.with_columns(**derived).select(*final)


def build_pivot(
    ds: Dataset,
    index: list[str],
    on: str,
    values: str,
    aggregate: str,
    columns: list[Any] | None,
) -> Dataset:
    """Reshape long → wide (SQL ``PIVOT`` / pandas ``pivot_table``).

    Lowers to ``group_by(index).agg(...)`` with one conditional aggregate per pivot
    value: ``<agg>(values) WHERE on == v``, expressed as
    ``when(on == v).then(values).otherwise(<typed null>).<agg>()`` — so it reuses the
    tested grouping/aggregation engine with no new operator. The else-branch uses
    ``nullif(values, values)`` (a value-typed null) so non-matching rows are ignored
    by the aggregate. With `columns` omitted, the distinct pivot values are discovered
    by an eager pre-pass over `on` (like DuckDB's auto-`PIVOT`).
    """
    if aggregate not in RUNNING_AGGREGATES:
        raise PlanError(
            f"pivot(): aggregate must be one of {RUNNING_AGGREGATES}, got {aggregate!r}"
        )
    for c in (*index, on, values):
        if c not in ds.columns:
            raise PlanError(f"pivot(): unknown column {c!r}")
    if columns is None:
        seen = ds.select(on).distinct().to_pydict()[on]
        cols = sorted(v for v in seen if v is not None)
    else:
        cols = list(columns)
    if not cols:
        raise PlanError("pivot(): no pivot column values to spread")
    typed_null = nullif(Col(values), Col(values))
    aggs: dict[str, Any] = {}
    for v in cols:
        masked = when(Col(on) == v).then(Col(values)).otherwise(typed_null)
        aggs[str(v)] = getattr(masked, aggregate)()
    return ds.group_by(*index).agg(**aggs)


def build_sample(
    ds: Dataset, fraction: float | None, seed: int | None, n: int | None = None
) -> Dataset:
    """Construct a `Sample` node — a fraction sample (`fraction`) or a fixed-count
    sample (`n`). Exactly one of `fraction`/`n` is set. `seed=None` bakes a fresh
    random seed at plan-build so the sample is reproducible within a run and
    consistent across workers."""
    if (fraction is None) == (n is None):
        raise PlanError("sample() takes exactly one of `fraction` or `n`")
    if seed is None:
        seed = random.randrange(2**63)
    # The fraction field is required by the node; for count mode it is unused (1.0).
    return ds._derive(Sample(ds._plan, 1.0 if n is not None else float(fraction), int(seed), n))


def build_unpivot(
    ds: Dataset,
    index: list[str] | None,
    on: list[str] | None,
    variable_name: str,
    value_name: str,
) -> Dataset:
    """Construct an `Unpivot` node (see `Dataset.unpivot` for the contract).

    With `on` omitted, every column not in `index` is melted; with `index` omitted,
    every column not in `on` becomes an identifier.
    """
    cols = ds.columns
    if index is None and on is None:
        raise PlanError("unpivot() requires `index` or `on`")
    idx = list(index) if index is not None else [c for c in cols if c not in set(on or ())]
    vals = list(on) if on is not None else [c for c in cols if c not in set(idx)]
    return ds._derive(Unpivot(ds._plan, tuple(idx), tuple(vals), variable_name, value_name))


def _bounded_interval_join(
    left: Dataset,
    right: Dataset,
    left_keys: list[str],
    right_keys: list[str],
    left_time: str,
    right_time: str,
    within_us: int,
    how: str,
) -> Dataset:
    """`join_stream` over two *bounded* inputs — the same answer, with no watermark.

    The interval is part of the join *condition*, not a filter above it. That distinction
    is invisible for an inner join and decides the answer for an outer one: a left row
    whose key matches but whose event time is outside the interval has not matched, so a
    left outer join must emit it null-padded. Filtering above the join deleted it instead,
    while the streaming driver — which never counts such a pair as a match — emitted it.
    The same query over bounded and unbounded inputs returned two different answers.

    Expressed with the relational surface rather than a non-equi join: the matched pairs
    are an inner join plus the interval filter, and each preserved side's unmatched rows
    are what an anti-join against those pairs leaves behind. A synthetic row index gives
    the anti-join a row identity, so duplicates on either side are handled exactly. The
    unmatched rows are then joined against an *empty* copy of the opposite side, which is
    what types their null columns — cheaper to reason about than constructing typed null
    literals, and it cannot drift from the matched rows' column order.

    Args:
        left: The left input.
        right: The right input.
        left_keys: The left equality keys.
        right_keys: The right equality keys.
        left_time: The left event-time column.
        right_time: The right event-time column.
        within_us: The interval half-width, in microseconds.
        how: ``"inner"``, ``"left"``, ``"right"``, or ``"full"``.

    Returns:
        A `Dataset` of the interval-joined rows in the join's output shape.
    """
    diff = Col(left_time).cast("int64") - Col(right_time).cast("int64")
    matched = (diff <= within_us) & (diff >= -within_us)
    if how == "inner":
        return left.join(right, left_on=left_keys, right_on=right_keys, how="inner").filter(matched)

    lid, rid = "__ij_left_id", "__ij_right_id"
    tagged_left = left.with_row_index(lid)
    tagged_right = right.with_row_index(rid)
    pairs = tagged_left.join(
        tagged_right, left_on=left_keys, right_on=right_keys, how="inner"
    ).filter(matched)

    parts = [pairs]
    if how in ("left", "full"):
        unmatched = tagged_left.join(pairs.select(lid), on=[lid], how="anti")
        parts.append(
            unmatched.join(
                tagged_right.limit(0), left_on=left_keys, right_on=right_keys, how="left"
            )
        )
    if how in ("right", "full"):
        unmatched = tagged_right.join(pairs.select(rid), on=[rid], how="anti")
        parts.append(
            tagged_left.limit(0).join(
                unmatched, left_on=left_keys, right_on=right_keys, how="right"
            )
        )
    combined = parts[0]
    for part in parts[1:]:
        combined = combined.union(part)
    return combined.select(*[c for c in pairs.columns if c not in (lid, rid)])
