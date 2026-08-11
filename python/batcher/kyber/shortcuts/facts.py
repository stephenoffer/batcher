"""The one place a plan's statistics become *facts* — the substrate every shortcut reads.

A metadata shortcut is only ever an optimisation: the value it returns must equal the value
executing the query would produce, or the optimizer is a wrong-answer generator and the
faster it runs the worse it is. Getting that right a hundred times over is not a hundred
arguments — it is **one** argument, made here, and then reused.

`Facts` is that argument. It reads a plan's estimated `RelStats` once and keeps only what is
*provably true without execution*:

  - a row count only when its provenance is `EXACT`;
  - a column's `min`/`null_count`/`ndv`/`sum`/`mean` only from an `EXACT` bundle (a filter
    downgrades a carried-through bound to a mere bound, and that is not an answer);
  - a column's `max` only when the bound also ranks NaN the way SQL does (see `nan_safe`).

Anything not provable is `None`, and `None` means "the caller must execute". The families
in this package (`rows`, `nulls`, `bounds`, `distinct`, `moments`, `checks`, `ordering`,
`approx`, `storage`) are then *pure derivations over `Facts`* — no plan, no statistics, no
provenance reasoning of their own. They cannot leak an approximate value into an exact
answer, because by the time they run there are no approximate values left to leak.

Layer: `kyber`, the decide layer. It decides *whether* an answer exists and what it is; it
never executes and never measures.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from batcher.config import Config
from batcher.kyber.metadata_answer import _root_stats, nan_aware_bounds
from batcher.metadata.hub import MetadataHub
from batcher.plan.logical import LogicalPlan
from batcher.plan.stats import ColumnStat, Provenance, RelStats, SortOrder

__all__ = ["ColumnFacts", "Facts", "facts_for", "facts_from_relstats"]


@dataclass(frozen=True, slots=True)
class ColumnFacts:
    """What is provably true about one column without reading a row.

    Every *exact* field is `None` unless it is provably correct: they are populated only
    from a `Provenance.EXACT` bundle, with the extra NaN gate on `max`/`total_sum`/`mean`
    (see `Facts.nan_safe`). The four *approximate* fields (`approx_ndv`, `quantiles`,
    `mcv`, `avg_bytes`) carry any provenance and may only back an explicitly-named
    `approx_*` answer — never an exact one.
    """

    name: str
    dtype: pa.DataType | None = None
    # --- exact facets (None ⇒ not provable ⇒ the caller must execute) ---
    min: Any | None = None
    max: Any | None = None
    null_count: int | None = None
    ndv: int | None = None
    total_sum: float | None = None
    mean: float | None = None
    # --- approximate facets (never answer an exact question) ---
    approx_ndv: int | None = None
    quantiles: Mapping[str, list[float]] | None = None
    mcv: Mapping[str, float] | None = None
    avg_bytes: float | None = None
    # A membership bloom over the column's values. Consulted only to prove *absence*
    # (a value the bloom rejects is provably not present, in this relation or any subset
    # of it), never to claim presence — so it is sound at any provenance.
    bloom: bytes | None = None

    @property
    def has_bounds(self) -> bool:
        """True iff both an exact lower and upper bound are known."""
        return self.min is not None and self.max is not None

    @property
    def is_float(self) -> bool:
        """True iff this is a floating-point column (the type NaN lives in)."""
        return self.dtype is not None and pa.types.is_floating(self.dtype)

    @property
    def is_numeric(self) -> bool:
        """True iff this column is numeric (integer, float, or decimal)."""
        if self.dtype is None:
            return False
        return (
            pa.types.is_integer(self.dtype)
            or pa.types.is_floating(self.dtype)
            or pa.types.is_decimal(self.dtype)
        )


@dataclass(frozen=True, slots=True)
class Facts:
    """Everything a shortcut may read about a relation, already gated for exactness.

    `rows` is the exact row count or `None`; `estimated_rows` always has a value but is
    only ever an estimate. `nan_safe` records whether float bounds may be trusted as
    answers (see the `bounds_include_nan` source declaration): when it is False, no float
    `max` / `sum` / `mean` fact is populated at all, so a derivation downstream cannot
    accidentally use one.
    """

    rows: int | None = None
    estimated_rows: float = 0.0
    columns: Mapping[str, ColumnFacts] = field(default_factory=dict)
    sorted_by: tuple[SortOrder, ...] = ()
    nan_safe: bool = False

    def col(self, name: str) -> ColumnFacts:
        """This relation's facts for column `name` — all-unknown if nothing is known."""
        known = self.columns.get(name)
        return known if known is not None else ColumnFacts(name=name)

    @property
    def rows_known(self) -> bool:
        """True iff the exact row count is known without execution."""
        return self.rows is not None


def facts_for(
    plan: LogicalPlan,
    sources: list,
    source_stats: list | None = None,
    hub: MetadataHub | None = None,
    config: Config | None = None,
) -> Facts:
    """Estimate `plan`'s root and distil it into the provable `Facts` a shortcut may use.

    One optimizer rewrite + one EXACT-first estimate, shared by every shortcut the caller
    then asks — which is the point: a namespace that answers thirty questions about a
    dataset pays for the plan analysis once, and each answer after that is a field read.
    """
    _, stats = _root_stats(plan, sources, source_stats, hub, config)
    schema = plan.available_schema()
    nan_safe = nan_aware_bounds(sources, source_stats)
    columns = {
        name: _column_facts(name, stat, _dtype_of(schema, name), nan_safe)
        for name, stat in stats.columns.items()
    }
    # A column the estimator says nothing about still has a *type*, and a type alone
    # answers a question ("is this numeric?"). Carry every output column, not just the
    # ones with statistics.
    for name in schema.names if schema is not None else ():
        if name not in columns:
            columns[name] = ColumnFacts(name=name, dtype=_dtype_of(schema, name))
    return Facts(
        rows=int(stats.rows) if stats.rows_exact else None,
        estimated_rows=float(stats.rows),
        columns=columns,
        sorted_by=stats.sorted_by,
        nan_safe=nan_safe,
    )


def _dtype_of(schema: Any, name: str) -> pa.DataType | None:
    """`name`'s Arrow type in `schema`, or None when the schema can't say."""
    if schema is None or not schema.has(name):
        return None
    return schema.field(name).type


def _column_facts(
    name: str, stat: ColumnStat, dtype: pa.DataType | None, nan_safe: bool
) -> ColumnFacts:
    """Distil one `ColumnStat` into facts, dropping everything not provably exact.

    The whole provenance firewall for the shortcut layer, in one function:

    * A bundle that is not `Provenance.EXACT` contributes **no** exact facet. A column
      carried through a filter keeps its min/max as valid *bounds*, but a bound is not an
      extreme, and an optimization that confuses the two returns a wrong answer.
    * `ndv` carries its own tag (`ndv_is_exact`), because a Parquet column holds a measured
      (HLL) distinct count alongside exact bounds. Only an exact ndv becomes the exact
      `ndv`; a sketch one becomes `approx_ndv`, which no exact answer may read.
    * `max`, `total_sum`, and `mean` of a **float** column additionally require `nan_safe`.
      NaN is the greatest value in SQL's total order, but the Parquet spec omits it from
      statistics and a KLL sketch drops it — so an unaware bound reports the largest
      *non-NaN* value, and a sum/mean computed without it is likewise not the sum/mean the
      engine would compute. `min` is exempt: a dropped NaN can never have been the minimum.
    """
    exact = stat.provenance is Provenance.EXACT
    is_float = dtype is not None and pa.types.is_floating(dtype)
    float_gated = is_float and not nan_safe
    ndv_exact = stat.ndv is not None and stat.ndv_is_exact
    return ColumnFacts(
        name=name,
        dtype=dtype,
        min=stat.min if exact else None,
        max=stat.max if (exact and not float_gated) else None,
        # Not gated on `exact` (the bundle): a Parquet string column has a byte-truncated
        # min/max and therefore a DEFAULT bundle, but its null count is exactly recorded.
        null_count=(
            int(stat.null_count)
            if (stat.null_count is not None and stat.null_count_is_exact)
            else None
        ),
        ndv=int(stat.ndv) if ndv_exact else None,
        total_sum=stat.total_sum if (exact and not float_gated) else None,
        mean=stat.mean if (exact and not float_gated) else None,
        approx_ndv=int(stat.ndv) if stat.ndv is not None else None,
        quantiles=stat.quantiles,
        mcv=stat.mcv,
        avg_bytes=stat.avg_bytes,
        bloom=stat.bloom,
    )


def facts_from_relstats(stats: RelStats, schema: Any = None, *, nan_safe: bool = False) -> Facts:
    """Distil an already-estimated `RelStats` into `Facts` — the plan-free entry point.

    Used by tests and by callers that hold a `RelStats` directly (a measured relation, a
    hand-built estimate). `facts_for` is the normal way in; this shares its distillation so
    the two cannot disagree about what counts as provable.
    """
    columns = {
        name: _column_facts(name, stat, _dtype_of(schema, name), nan_safe)
        for name, stat in stats.columns.items()
    }
    return Facts(
        rows=int(stats.rows) if stats.rows_exact else None,
        estimated_rows=float(stats.rows),
        columns=columns,
        sorted_by=stats.sorted_by,
        nan_safe=nan_safe,
    )
