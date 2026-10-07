"""Bodies of `Dataset.sample` and `Dataset.upsample`: choosing rows, and adding the missing ones.

Every per-row decision here is an expression the engine evaluates, never a Python loop. A
sample's random draw is ``hash_rows`` of the row's own values under the seed, the same
deterministic, partition-independent draw the `Sample` node uses, so a weighted or keyed
sample selects the same rows on one node or many. `upsample` builds its grid with
``sequence`` and ``explode`` per group and stacks it against the observed rows.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pyarrow as pa

from batcher._internal.errors import PlanError, require_float, require_int
from batcher.api.dataset._build.combine import unused_name
from batcher.plan.expr_ir import Col, col, count, hash_rows, lit
from batcher.plan.functions.collection import sequence
from batcher.plan.functions.temporal import _duration_micros
from batcher.plan.logical import Sample
from batcher.plan.types import dtype_name

if TYPE_CHECKING:
    from batcher.api.dataset.frame import Dataset
    from batcher.plan.expr_ir import Expr

__all__ = ["build_sample", "build_upsample"]

#: 2^52: the draw keeps 52 bits of the row hash, so ``(bits + 0.5) / 2^52`` is exact in a
#: float64 and lies strictly inside (0, 1) -- `ln` of it is finite, and no row draws 1.
_DRAW_RANGE = 1 << 52


def _uniform(columns: list[str], seed: int) -> Expr:
    """A uniform draw in (0, 1) per row, from a seeded hash of `columns`' values."""
    digest = hash_rows(*(Col(c) for c in columns), seed=seed)
    bits = ((digest % _DRAW_RANGE) + _DRAW_RANGE) % _DRAW_RANGE
    return (bits.cast("float64") + 0.5) / float(_DRAW_RANGE)


def build_sample(
    ds: Dataset,
    fraction: float | None,
    seed: int | None,
    n: int | None = None,
    *,
    weights: str | None = None,
    key: str | list[str] | None = None,
) -> Dataset:
    """Sample by `fraction` or by count `n`, optionally weighted or keyed.

    Plain fraction and count samples are the `Sample` node. `weights` is Efraimidis-Spirakis
    weighted sampling without replacement: each row draws ``u`` in (0, 1) and the `n` rows
    with the smallest ``-ln(u) / weight`` are kept, an expression plus a top-n, so it is
    mergeable. `key` keeps a row iff the draw over its key columns is below `fraction`, so
    rows sharing a key are kept or dropped together.

    Args:
        ds: The dataset to sample.
        fraction: The fraction to keep, or ``None`` for a count sample.
        seed: The draw's seed; ``None`` bakes a fresh one.
        n: The number of rows to keep, or ``None`` for a fraction sample.
        weights: A numeric column weighting each row (count samples only).
        key: The column(s) whose values decide a row (fraction samples only).

    Returns:
        The sampled dataset.

    Raises:
        PlanError: For a bad argument combination, an unknown or non-numeric column, a
            negative weight, or weights that total zero.
    """
    if (fraction is None) == (n is None):
        raise PlanError("sample() takes exactly one of `fraction` or `n`")
    if seed is None:
        seed = random.randrange(2**63)
    seed = int(seed)
    if weights is not None:
        if n is None:
            raise PlanError(
                "sample(weights=...) needs a row count n: a weighted fraction has no defined "
                "size without the total weight. Pass n=<rows>."
            )
        return _weighted_sample(ds, require_int(n, func="sample", arg="n"), weights, seed)
    if key is not None:
        if fraction is None:
            raise PlanError(
                "sample(key=...) takes a fraction, not n: a keyed sample keeps whole keys, "
                "so it cannot promise an exact row count."
            )
        return _keyed_sample(ds, require_float(fraction, func="sample", arg="fraction"), key, seed)
    # The fraction field is required by the node; for count mode it is unused (1.0).
    rate = 1.0 if n is not None else require_float(fraction, func="sample", arg="fraction")
    return ds._derive(Sample(ds._plan, rate, seed, n))


def _keyed_sample(ds: Dataset, fraction: float, key: str | list[str], seed: int) -> Dataset:
    """Keep each row whose key columns draw below `fraction` -- whole keys in or out."""
    keys = [key] if isinstance(key, str) else list(key)
    if not keys:
        raise PlanError("sample(key=...) needs at least one column")
    unknown = [k for k in keys if k not in ds.columns]
    if unknown:
        raise PlanError(f"sample(key=...): unknown column(s) {unknown}; columns: {ds.columns}")
    if not 0.0 <= fraction <= 1.0:
        raise PlanError(f"sample fraction must be in [0, 1], got {fraction}")
    return ds.filter(_uniform(keys, seed) < fraction)


def _weighted_sample(ds: Dataset, n: int, weights: str, seed: int) -> Dataset:
    """Efraimidis-Spirakis: the `n` rows with the smallest ``-ln(u) / w``.

    The weights are validated with one eager aggregate first, because no expression can
    raise at execution: a negative weight has no meaning as a probability, and weights that
    total zero leave nothing to choose from. A null or zero weight is never selected.
    """
    if n < 0:
        raise PlanError(f"sample n must be non-negative, got {n}")
    if weights not in ds.columns:
        raise PlanError(f"sample(weights=...): unknown column {weights!r}; columns: {ds.columns}")
    wtype = ds.schema.field(weights).type
    if not (
        pa.types.is_integer(wtype) or pa.types.is_floating(wtype) or pa.types.is_decimal(wtype)
    ):
        raise PlanError(f"sample(weights=...): column {weights!r} must be numeric, got {wtype}")
    if ds.is_streaming:
        raise PlanError(
            "sample(weights=...) cannot run over an unbounded input: its weights are "
            "validated by a pass over the whole relation."
        )
    w = col(weights).cast("float64")
    stats = ds.agg(rows=count(), negative=(w < 0).cast("int64").sum(), total=w.sum()).to_pydict()
    negative, total = stats["negative"][0] or 0, stats["total"][0] or 0.0
    if negative:
        raise PlanError(
            f"sample(weights={weights!r}): {negative} row(s) have a negative weight; weights "
            "must be >= 0 (a null or zero weight is never selected)"
        )
    if stats["rows"][0] and total <= 0:
        raise PlanError(
            f"sample(weights={weights!r}): the weights total zero (null counts as zero), so "
            "no row can be selected"
        )
    draw = unused_name("__bc_sample_key", ds)
    keyed = ds.filter(w > 0).with_columns(**{draw: -(_uniform(ds.columns, seed).ln()) / w})
    return keyed.bottom_k(n, draw).drop(draw)


def build_upsample(
    ds: Dataset,
    time_col: str,
    every: str,
    by: list[str],
    fill: str | None,
    indicator: str | None,
) -> Dataset:
    """Insert a row at every missing step of a regular time grid (Polars ``upsample``).

    Each group's grid runs from its earliest time to its latest in steps of `every`, built
    in the column's own integer unit with ``sequence`` and ``explode``. The observed rows are
    all kept, including any that fall between grid points or have a null time; a grid point
    is added only where no observed row of its group has that exact time. That test is a
    window count over ``(by, time)``, which treats a null key as a group of its own.

    Args:
        ds: The dataset to upsample.
        time_col: The date or timestamp column the grid runs over.
        every: The fixed step, such as ``"1h"`` or ``"15m"``.
        by: The columns whose groups each get their own grid.
        fill: ``"forward"`` or ``"backward"`` to carry values into inserted rows, or ``None``.
        indicator: A boolean column to add, true on inserted rows; ``None`` adds none.

    Returns:
        The dataset with the inserted rows.

    Raises:
        PlanError: For an unknown or non-temporal column, a step that is not a whole number
            of the column's unit, an unknown `fill`, or a clashing `indicator` name.
    """
    for c in (time_col, *by):
        if c not in ds.columns:
            raise PlanError(f"upsample(): unknown column {c!r}; columns: {ds.columns}")
    if time_col in by:
        raise PlanError(f"upsample(): the time column {time_col!r} cannot also be a `by` key")
    if fill not in (None, "forward", "backward"):
        raise PlanError(f"upsample(fill=...) must be 'forward', 'backward' or None, got {fill!r}")
    if indicator is not None and indicator in ds.columns:
        raise PlanError(f"upsample(indicator={indicator!r}) names an existing column")
    ttype = ds.schema.field(time_col).type
    step = _step_in_unit(ttype, _duration_micros(every, arg="upsample(every=...)"), every)

    ticks, lo, hi, observed = (
        unused_name(s, ds) for s in ("__up_t", "__up_lo", "__up_hi", "__up_n")
    )
    target = dtype_name(ttype)
    t_int = col(time_col).cast("int64")
    grid = (
        ds.group_by(*by)
        .agg(**{lo: t_int.min(), hi: t_int.max()})
        .select(*by, **{ticks: sequence(col(lo), col(hi), step)})
        .explode(ticks)
        .select(*by, **{time_col: col(ticks).cast(target)})
    )
    flag = indicator or unused_name("__up_inserted", ds)
    from batcher.api.session.combine import concat

    stacked = concat(
        [ds.with_columns(**{flag: lit(False)}), grid.with_columns(**{flag: lit(True)})],
        how="diagonal",
    )
    # How many *observed* rows share this (group, time); a grid point is kept only at zero.
    seen = (~col(flag)).cast("int64").sum().over(partition_by=[*by, time_col])
    out = (
        stacked.with_columns(**{observed: seen})
        .filter(~col(flag) | (col(observed) == 0))
        .drop(observed)
    )
    if fill is not None:
        values = [c for c in ds.columns if c not in (time_col, *by)]
        if values:
            out = out.fill_null(
                strategy=fill, subset=values, order_by=[time_col], partition_by=by or None
            )
    return out.select(*ds.columns, *([indicator] if indicator else []))


def _step_in_unit(ttype: pa.DataType, micros: int, every: str) -> int:
    """`every`, in microseconds, as a whole count of `ttype`'s own integer unit."""
    if pa.types.is_date32(ttype):
        per_unit = 86_400_000_000
    elif pa.types.is_timestamp(ttype):
        per_unit = {"s": 1_000_000, "ms": 1_000, "us": 1, "ns": 0}[ttype.unit]
        if per_unit == 0:
            return micros * 1_000
    else:
        raise PlanError(f"upsample(): the time column must be a date or timestamp, got {ttype}")
    if micros % per_unit:
        raise PlanError(
            f"upsample(every={every!r}) is not a whole number of the time column's unit "
            f"({ttype}); use a step that is"
        )
    return micros // per_unit
