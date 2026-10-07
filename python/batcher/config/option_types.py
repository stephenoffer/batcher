"""`Literal` aliases for the closed option vocabularies on the public signatures.

A keyword such as ``join(how=...)`` accepts a fixed set of strings. Typing it ``str`` lets
an editor offer nothing and a type checker accept ``how="lefft"``, which then fails only
when the plan is built. These aliases give the signature the vocabulary, so completion
lists the accepted spellings and a checker flags a misspelling before the code runs.

They are annotations, not validation. Each runtime check keeps its own accepted set, and
that set stays the single source of truth: `tests/unit/test_option_types.py` holds every
alias here equal to the set its validator accepts, so the two cannot drift. Only a
vocabulary the runtime treats as closed belongs here. A keyword that also accepts aliases
or format-specific spellings, such as a writer's ``mode=``, which folds case and takes
Spark's ``"ErrorIfExists"`` and a sink's own DML verbs, stays ``str``: a ``Literal`` there
would reject calls the engine accepts.

Neutral layer 0: it imports nothing from Batcher, so `plan` and `api` both annotate with it.
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "AsofDirection",
    "Backend",
    "BatchFormat",
    "ConcatHow",
    "DaskMaterialize",
    "DistinctKeep",
    "DropNullsHow",
    "DtypeBackend",
    "ExtraColumns",
    "FillStrategy",
    "HuggingFaceMode",
    "JoinHow",
    "JoinValidate",
    "MappingStrategy",
    "NanPolicy",
    "NumpyNulls",
    "OutputModeName",
    "QuantileInterpolation",
    "RowBatchFormat",
    "TableWriteMode",
    "UpdateHow",
]

#: ``Dataset.join(how=...)``: the keyed joins, ``"outer"`` (read as ``"full"``), and ``"cross"``.
JoinHow = Literal["inner", "left", "right", "full", "outer", "semi", "anti", "cross"]

#: ``Dataset.join(validate=...)``: pandas' key-cardinality spellings.
JoinValidate = Literal["1:1", "1:m", "m:1", "m:m"]

#: ``Dataset.join_asof(direction=...)``: which side of the left key a match may fall on.
AsofDirection = Literal["backward", "forward", "nearest"]

#: ``batch_format=`` on ``map_batches``/``iter_batches``/``filter``: the batch handed over.
BatchFormat = Literal["pyarrow", "numpy", "pandas", "torch", "polars", "jax"]

#: ``batch_format=`` on the per-row ``map``/``flat_map``: the formats a row adapter can split.
RowBatchFormat = Literal["pyarrow", "numpy"]

#: ``quantile(interpolation=...)``: how a quantile between two values is resolved.
QuantileInterpolation = Literal["linear", "lower", "higher", "nearest", "midpoint"]

#: ``max(nan_policy=...)``: whether a NaN wins the maximum or is skipped.
NanPolicy = Literal["propagate", "ignore"]

#: ``Dataset.fill_null(strategy=...)``: a constant zero, a whole-relation aggregate, or a carry.
FillStrategy = Literal["zero", "mean", "min", "max", "forward", "backward"]

#: ``Dataset.distinct(keep=...)``: which row of a duplicate set survives.
DistinctKeep = Literal["first", "last", "any"]

#: ``Dataset.drop_nulls(how=...)``: drop on any null, or only when every column is null.
DropNullsHow = Literal["any", "all"]

#: ``Dataset.update(how=...)``: which rows of the left side an update keeps.
UpdateHow = Literal["left", "inner", "full"]

#: ``Dataset.match_to_schema(extra_columns=...)``: what a column the schema lacks does.
ExtraColumns = Literal["raise", "ignore"]

#: ``Dataset.to_numpy(nulls=...)``: how a null reaches a NumPy array.
NumpyNulls = Literal["nan", "raise", "mask"]

#: ``Dataset.to_pandas(dtype_backend=...)``: the pandas column representation.
DtypeBackend = Literal["numpy", "numpy_nullable", "pyarrow"]

#: ``Dataset.to_dask(materialize=...)``: when and where the query runs.
DaskMaterialize = Literal["arrow", "deferred", "parquet"]

#: ``Dataset.to_huggingface(mode=...)``: an in-memory ``Dataset`` or an ``IterableDataset``.
HuggingFaceMode = Literal["materialized", "iterable"]

#: ``collect``/``explain(backend=...)``: the CPU engine, the device tier, or Kyber's pick.
Backend = Literal["cpu", "gpu", "auto"]

#: ``output_mode=`` on a streaming write: the rows each trigger emits.
OutputModeName = Literal["append", "complete", "update"]

#: ``Dataset.write.table(mode=...)``: how a catalog write treats an existing table.
TableWriteMode = Literal[
    "error", "ignore", "append", "overwrite", "overwrite_partitions", "replace"
]

#: ``bt.concat(how=...)``: how frames with different columns are stacked.
ConcatHow = Literal["vertical", "vertical_relaxed", "diagonal", "horizontal"]

#: ``Expr.over(mapping_strategy=...)``: how a window result maps back onto rows.
MappingStrategy = Literal["group_to_rows", "join", "explode"]
