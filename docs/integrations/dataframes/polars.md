# Polars

This page covers moving data between Polars and Batcher, and how to decide which of the two should run a given step.

The two libraries agree on more than they disagree on. Both are lazy, both are expression-first, and both hold Arrow underneath, so a `LazyFrame` and a {py:obj}`Dataset <batcher.Dataset>` are the same idea. The conversion is a buffer handoff.

| | |
| --- | --- |
| **Read** | {py:obj}`bt.from_polars(df) <batcher.from_polars>` |
| **Write** | {py:obj}`ds.to_polars() <batcher.Dataset.to_polars>` |
| **Extra** | `polars` |
| **Cost** | Zero-copy through Arrow, both directions |

## Round trip

```python
import batcher as bt
import polars as pl

spend = pl.DataFrame({"user": ["a", "b", "a", "c"], "spend": [10.0, 2.0, 7.0, 4.0]})

big = (
    bt.from_polars(spend)
    .group_by("user")
    .agg(total=bt.col("spend").sum())
    .filter(bt.col("total") > 5)
    .sort("user")
)
print(big.to_pydict())
# {'user': ['a'], 'total': [17.0]}
```

`to_polars()` hands the result back as a Polars `DataFrame`, so a Batcher step drops into the middle of a Polars script without either side knowing:

```python
print(type(big.to_polars()).__name__)
# DataFrame
```

`from_polars` takes an eager `DataFrame`. A `LazyFrame` has not computed anything yet, so collect it first and hand over the result. There is no way to splice one engine's unevaluated plan into the other's, and pretending otherwise would mean re-planning Polars expressions in Batcher's IR.

## Which engine should run the step

Reach for Batcher when the job outgrows one machine, needs a source Polars has no reader for, or mixes relational work with models. Stay in Polars when it already does the job and the data fits.

| Reach for Batcher when | Because |
| --- | --- |
| The data outgrows memory, or one machine | Every stateful operator is mergeable, so the same code spills and distributes |
| The pipeline ends in a model | `ds.ml` scores, embeds, and loads for training in the same plan |
| The source is Kafka, a warehouse, or a lakehouse table | Those readers split and push down; a `pl.read_*` does not reach them |
| Rows are images, audio, or video | Decode is an expression the optimizer can skip |
| The query runs often enough to be worth learning | Measured cardinalities carry across runs |

Stay in Polars for a one-shot transform on data that fits, for its plotting and I/O conveniences, and for anything already written and working. The conversion is cheap enough that mixing the two per step is a reasonable design rather than a compromise.

## What does not carry over

Column names and Arrow types carry over exactly. Three Polars concepts have no Batcher counterpart, and each is a deliberate absence rather than a gap:

- **Eager evaluation.** `pl.DataFrame` computes as you type. A `Dataset` is always lazy, even for `from_polars`, so nothing runs until a terminal call.
- **`pl.Series` as a standalone value.** Batcher has no one-column type outside a dataset. Take the column with `to_polars()` and index it there.
- **The `.map_elements` row callback.** Batcher's UDF contract is per Arrow batch, never per row, because a per-row Python call in the data plane is the cost the engine exists to avoid. {doc}`/user-guide/transform/columns/udfs` covers the batch form.

The verb-by-verb mapping, including every Polars name whose Batcher spelling differs and every place the two engines compute different answers, is generated from the parity registry: {doc}`/getting-started/migration/polars/index`.

## See also

- {doc}`/getting-started/migration/polars/index`: all 265 Polars names, with the status of each.
- {doc}`index`: the other in-process libraries and the zero-copy contract they share.
- {doc}`/benchmarks/comparisons/vs-polars`: the measured comparison, suite by suite.
- {doc}`/user-guide/transform/columns/expressions`: the expression language, which reads much like Polars'.
