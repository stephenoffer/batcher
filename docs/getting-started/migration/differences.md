# Differences and verification

This page covers the concepts that change whichever engine you come from, the familiar APIs Batcher replaces, and how to prove a port returns the same rows. Name-by-name differences are in the generated reference: {doc}`spark/index`, {doc}`polars/index`, {doc}`daft/index`, and {doc}`ray-data/index`.

## A dataset is lazy

A {py:class}`Dataset <batcher.Dataset>` is a plan, not data, as in a Polars `LazyFrame` or a Spark `DataFrame`. Coming from pandas, a line that used to compute a value now only extends a plan, and a terminal call such as `count`, `to_pydict`, or a write runs it.

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3]})
plan = ds.filter(bt.col("x") > 1)  # nothing has run yet
print(plan.count())
# 2
```

## One spelling per capability

Batcher keeps one name per capability. Another engine's spelling, or a second spelling Batcher used to accept, raises `AttributeError` naming the replacement, and the codemod rewrites a script that still uses one. It prints a diff unless you pass `--write`:

```bash
python -m batcher.migrate --from batcher --to batcher <paths>
```

Replace `<paths>` with the files or directories to rewrite.

## Write modes have different defaults

Default save modes differ between engines. Batcher's `ds.write(path)` overwrites by default, and `ds.write.delta(uri)` appends. The other engines default as follows:

| Engine | Writer | Default when the target exists |
|---|---|---|
| PySpark | `DataFrame.write` | raise an error (`errorifexists`) |
| Polars | `DataFrame.write_delta` | raise an error |
| Daft | `write_parquet`, `write_csv` | append |
| Ray Data | `write_parquet`, `write_csv` | append |

Pass `mode=` explicitly on every ported write. `mode="error"` matches PySpark:

```python
ds.write.parquet("sales.parquet", mode="overwrite")
try:
    ds.write.parquet("sales.parquet", mode="error")
except bt.PlanError:
    print("refused: output exists")
# refused: output exists
```

File sinks don't take `append`; a Parquet or CSV write that appended in Daft or Ray Data becomes a lakehouse table here.

## Integer overflow wraps

Integer arithmetic in Batcher wraps on overflow instead of raising or returning null, so adding 1 to the largest 64-bit integer returns the smallest one:

```python
big = bt.from_pydict({"x": [2**63 - 1]})
print(big.select(y=bt.col("x") + 1).to_pydict())
# {'y': [-9223372036854775808]}
```

An integer `sum` that overflows raises an `ExecutionError` asking you to cast to a wider type first. Unsigned 64-bit integers are stored as signed 64-bit, so a `UInt64` above `2**63 - 1` overflows on the way in.

## APIs with a relational replacement

Some familiar APIs are absent by design, because a relation has no row index or row order. Each one raises an `AttributeError` naming the replacement.

:::{dropdown} The replacement table
:open:


| Absent | Why | Instead |
|---|---|---|
| `df.set_index`, `df.reset_index`, `df.loc`, `df.iloc` | A relation is an unordered multiset with no row index, as in SQL. | {py:meth}`ds.filter(...) <batcher.Dataset.filter>`, {py:meth}`ds.select(...) <batcher.Dataset.select>`, {py:meth}`ds.sort(...) <batcher.Dataset.sort>`, {py:meth}`ds.with_row_index() <batcher.Dataset.with_row_index>` |
| `df.iterrows`, `df.itertuples`, `df.applymap` | Per-row Python never runs on the hot path. | {py:meth}`ds.iter_rows(named=True) <batcher.Dataset.iter_rows>` at the end of a pipeline; expressions or {py:meth}`ds.map_batches() <batcher.Dataset.map_batches>` inside one |
| `df.apply` | Its per-row and per-column meanings don't survive a columnar engine. | {py:meth}`ds.with_columns(y=expr) <batcher.Dataset.with_columns>` or {py:meth}`ds.map_batches(fn) <batcher.Dataset.map_batches>` |
| `df.T`, `df.transpose` | Transposing needs a materialized, single-typed frame. | `ds.to_pandas().T`, or {py:meth}`ds.unpivot() <batcher.Dataset.unpivot>` / {py:meth}`ds.pivot() <batcher.Dataset.pivot>` |
| `df.shift`, `df.diff`, `df.cumsum`, `df.rolling` | Each needs a row order the relation doesn't carry. | {py:meth}`ds.window(order_by=[...], functions={...}) <batcher.Dataset.window>` |
| `df.resample` | Time bucketing is a grouping. | {py:meth}`ds.group_by(bucket=bt.window(bt.col("t"), "1h")).agg(...) <batcher.Dataset.group_by>` |
| Looping over a {py:class}`GroupBy <batcher.GroupBy>` | It materializes one frame per key in Python and caps the job at one machine. | `.agg(...)`, or `.window(partition_by=[...])` to keep every row |

:::

The two most common replacements, a row index and a running total:

```python
nums = bt.from_pydict({"x": [1, 2, 3]})
print(nums.with_row_index("i").to_pydict())
# {'i': [0, 1, 2], 'x': [1, 2, 3]}
print(nums.window(order_by=["x"], functions={"running": ("sum", "x")}).to_pydict())
# {'x': [1, 2, 3], 'running': [1, 3, 6]}
```

Column attribute access such as `df.amount` is absent too, because a column named `filter` would shadow a method. Use `ds["amount"]` or {py:func}`bt.col("amount") <batcher.col>`.

## The error messages teach you the mapping

Type the method you already know, and the traceback names the Batcher spelling. That holds on a {py:class}`Dataset <batcher.Dataset>`, on an expression, on a `GroupBy`, and on the `bt` package itself.

```python
import batcher as bt

demo = bt.from_pydict({"x": [1, 2, 3], "k": ["a", "b", "a"]})

# A pandas reshape on a Dataset:
try:
    demo.pivot_table
except AttributeError as exc:
    assert "ds.pivot" in str(exc)

# A Polars per-element UDF on an expression:
try:
    bt.col("x").map_elements
except AttributeError as exc:
    assert "map_batches" in str(exc)

# A pandas GroupBy transform:
try:
    demo.group_by("k").transform
except AttributeError as exc:
    assert "ds.window" in str(exc)

# A Polars top-level constructor:
try:
    bt.LazyFrame
except AttributeError as exc:
    assert "already lazy" in str(exc)

print("every wrong spelling names its Batcher replacement")
# every wrong spelling names its Batcher replacement
```

A near miss on a real method gets a `Did you mean ...?` suggestion instead, so a typo such as `ds.filtr` or `bt.col("x").meen` points straight at `filter` and `mean`.

## Checking a port

{py:meth}`ds.equals(other) <batcher.Dataset.equals>` executes both sides and compares their rows, so two queries built from different verbs are equal when their results agree. Column names and types must match, and row order is ignored unless you pass `ordered=True`.

```python
ds = bt.from_pydict({"status": ["paid", "open", "paid"], "amount": [10, 20, 30]})

ported = ds.filter(status="paid")
expected = ds.filter(bt.col("status") == "paid")
print(ported.equals(expected))
# True
print(ds.sort("amount").equals(ds.sort("amount", descending=True), ordered=True))
# False
```

## Requirements and limitations

- {py:func}`from_pandas <batcher.from_pandas>`, {py:func}`from_polars <batcher.from_polars>`, {py:func}`from_spark <batcher.from_spark>`, {py:func}`from_daft <batcher.from_daft>`, {py:func}`from_dask <batcher.from_dask>`, {py:func}`from_ray_dataset <batcher.from_ray_dataset>`, {py:func}`from_huggingface <batcher.from_huggingface>`, {py:func}`from_torch <batcher.from_torch>`, and {py:func}`from_tf <batcher.from_tf>` each need the source framework installed.
- To hand a result to Dask or HuggingFace, go through {py:meth}`to_arrow <batcher.Dataset.to_arrow>` or {py:meth}`to_pandas <batcher.Dataset.to_pandas>`.
- `append` mode is accepted by lakehouse sinks only.
- `merge_on` is a `write.delta` parameter. It has no equivalent on a plain Parquet write.
- Distributed execution and the GPU actor pools need the optional `[ray]` extra.
- LLM generation needs a text-generation engine you install separately, such as `batcher-engine[vllm]`.

## See also

- {doc}`/getting-started/migration/transforming`: the verb-by-verb table.
- {doc}`/getting-started/migration/spark/index`, {doc}`/getting-started/migration/polars/index`, {doc}`/getting-started/migration/daft/index`, {doc}`/getting-started/migration/ray-data/index`: every name in each engine, with its Batcher spelling and what differs.
- {doc}`/agents`: the migration skills, each ending in this verification step.
- {doc}`/user-guide/operate/running/troubleshooting`: diagnosing a ported query that runs but misbehaves.
