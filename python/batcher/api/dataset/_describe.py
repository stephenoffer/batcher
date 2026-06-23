"""Descriptive-statistics helpers behind `Dataset.describe` / `Dataset.null_count`.

These compose the already-tested aggregates (``count``/``mean``/``std``/``min``/
``max``/``quantile``) rather than adding any new IR. `null_count` stays lazy — it
lowers to one aggregate plus a `select`, so nothing executes until a terminal op.
`describe` is a summary view: it must *transpose* per-column aggregates into
statistic *rows*, which has no relational spelling, so it materializes the tiny
one-row aggregate (a control-plane reshape over the columns-by-statistics values,
never a per-row tuple touch) and returns it as a new `Dataset`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import col, count

if TYPE_CHECKING:
    from batcher.api.dataset.frame import Dataset

# Internal aggregate-alias suffixes, kept unlikely to collide with user columns.
_TOTAL = "__bt_n"


def _is_numeric(dtype: pa.DataType) -> bool:
    return pa.types.is_integer(dtype) or pa.types.is_floating(dtype)


def null_count(ds: Dataset) -> Dataset:
    """A one-row dataset of the null count of every column (lazy).

    Lowers to a single global aggregate (``count(*)`` and a per-column non-null
    ``count``) plus a `select` that subtracts them, so it stays lazy and mergeable
    — identical single-node and distributed. Mirrors pandas ``df.isnull().sum()``.
    """
    cols = ds.columns
    aggs = {_TOTAL: count()}
    for c in cols:
        aggs[f"{c}__cnt"] = col(c).count()
    counted = ds.agg(**aggs)
    return counted.select(**{c: (col(_TOTAL) - col(f"{c}__cnt")) for c in cols})


def describe(ds: Dataset, percentiles: tuple[float, ...] = (0.25, 0.5, 0.75)) -> Dataset:
    """Summary statistics per column as a `Dataset` (executes the query).

    See `Dataset.describe`. Numeric columns get count / null_count / mean / std /
    min / the requested `percentiles` / max; non-numeric columns get count and
    null_count only (numeric cells are null). The result has a ``statistic`` label
    column and one Float64 column per input column.
    """
    for p in percentiles:
        if not 0.0 <= p <= 1.0:
            raise PlanError(f"describe(): percentile {p} is not in [0, 1]")

    cols = ds.columns
    types = list(ds.schema.types)
    numeric = {c for c, t in zip(cols, types, strict=True) if _is_numeric(t)}

    aggs = {_TOTAL: count()}
    for c in cols:
        aggs[f"{c}__cnt"] = col(c).count()
    for c in cols:
        if c in numeric:
            aggs[f"{c}__mean"] = col(c).mean()
            aggs[f"{c}__std"] = col(c).std()
            aggs[f"{c}__min"] = col(c).min()
            aggs[f"{c}__max"] = col(c).max()
            for p in percentiles:
                aggs[f"{c}__q{p}"] = col(c).quantile(p)

    row = ds.agg(**aggs).collect()
    cell = {name: row.column(name)[0].as_py() for name in row.column_names}

    pct_labels = [f"{p * 100:g}%" for p in percentiles]
    stat_labels = ["count", "null_count", "mean", "std", "min", *pct_labels, "max"]

    out: dict[str, pa.Array] = {"statistic": pa.array(stat_labels, type=pa.string())}
    total = cell[_TOTAL]
    for c in cols:
        cnt = cell[f"{c}__cnt"]
        values: list[float | None] = [float(cnt), float(total - cnt)]
        if c in numeric:
            values.append(_as_float(cell[f"{c}__mean"]))
            values.append(_as_float(cell[f"{c}__std"]))
            values.append(_as_float(cell[f"{c}__min"]))
            values.extend(_as_float(cell[f"{c}__q{p}"]) for p in percentiles)
            values.append(_as_float(cell[f"{c}__max"]))
        else:
            # mean, std, min, max + one per percentile are undefined for non-numerics.
            values.extend([None] * (4 + len(percentiles)))
        out[c] = pa.array(values, type=pa.float64())

    from batcher.api.session import from_arrow

    return from_arrow(pa.table(out))


def profile(ds: Dataset) -> Dataset:
    """A per-column data-quality profile as a `Dataset` (executes the query).

    One row per input column with: ``count`` (non-null), ``null_count``,
    ``null_fraction``, and ``approx_distinct`` (HyperLogLog cardinality). Reuses the
    sketch-backed `approx_n_unique`, so it is one aggregate pass regardless of width
    — the quick "what does this column look like" check before a load.
    """
    cols = ds.columns
    aggs = {_TOTAL: count()}
    for c in cols:
        aggs[f"{c}__cnt"] = col(c).count()
        aggs[f"{c}__nd"] = col(c).approx_n_unique()
    cell_row = ds.agg(**aggs).collect()
    cell = {name: cell_row.column(name)[0].as_py() for name in cell_row.column_names}
    total = cell[_TOTAL]

    names: list[str] = []
    counts: list[int] = []
    null_counts: list[int] = []
    null_fracs: list[float] = []
    distincts: list[int] = []
    for c in cols:
        cnt = cell[f"{c}__cnt"]
        names.append(c)
        counts.append(cnt)
        null_counts.append(total - cnt)
        null_fracs.append((total - cnt) / total if total else 0.0)
        distincts.append(cell[f"{c}__nd"])

    from batcher.api.session import from_arrow

    return from_arrow(
        pa.table(
            {
                "column": pa.array(names, pa.string()),
                "count": pa.array(counts, pa.int64()),
                "null_count": pa.array(null_counts, pa.int64()),
                "null_fraction": pa.array(null_fracs, pa.float64()),
                "approx_distinct": pa.array(distincts, pa.int64()),
            }
        )
    )


def _as_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]
