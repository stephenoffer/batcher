---
name: migrate-from-polars-or-pandas
description: Port an existing pandas or Polars script to Batcher's public Python API — the eager-to-lazy shift, the verb-by-verb translation table, expression namespaces, window semantics, and how to prove the port returns identical results. Invoke when converting a pandas/Polars (or DataFrame-style) workload to Batcher, or when asked "what is the Batcher equivalent of X".
---

# Migrate from Polars or pandas

`docs/migration/index.md` is the user-facing mapping table and the source of truth.
Read it first; this skill is the agent-side procedure around it, and must never
contradict it. Everything below is verified against the live surface — if you need a
name this skill doesn't list, check it with
`python -c "import batcher as bt; print(bt.<name>)"` rather than guessing.

## When to use

- A user hands you a pandas or Polars script and wants it running on Batcher.
- You are asked for the Batcher spelling of a specific DataFrame operation.
- A port already exists and you need to prove it produces the same answer.

Not for: adding new engine capability (that is `add-expression-or-function` /
`add-relational-operator`).

## The one shift: everything is lazy

A `Dataset` is a handle to a plan. `select`, `filter`, `with_columns`,
`group_by().agg()`, `join`, `sort` all return a **new** `Dataset` and run nothing.
Work happens only at a terminal op: `collect()`, `to_arrow()`, `to_pydict()`,
`to_pandas()`, `to_polars()`, `count()`, `show()`, `iter_batches()`, `write.*`.

- **From pandas** this is the whole migration. Every `df = df.something()` line that
  silently materialized now just extends a plan. Print statements that relied on
  eager values need an explicit terminal call.
- **From Polars** the `LazyFrame` model maps 1:1 — with one simplification: Batcher
  has **no eager/`scan_*` split**. There is no `bt.scan_parquet`; every `bt.read.*`
  is already lazy and already does projection/predicate pushdown. One spelling per
  format, and it is the lazy one.

## Translation table

Extends the tables in `docs/migration/index.md`; consult that page for the full
pandas/Polars/PySpark grid, including IO, terminal ops, and the `from_*`/`to_*`
round-trips.

| pandas | Polars | Batcher |
|---|---|---|
| `pd.read_parquet(p)` | `pl.scan_parquet(p)` | `bt.read.parquet(p)` (lazy) |
| `pd.DataFrame(d)` | `pl.DataFrame(d)` | `bt.from_pydict(d)` |
| `df[["a","b"]]` | `df.select("a","b")` | `ds.select("a", "b")` |
| `df.assign(c=...)` | `df.with_columns(c=...)` | `ds.with_columns(c=...)` |
| `df[df.a > 1]` | `df.filter(pl.col("a") > 1)` | `ds.filter(col("a") > 1)` |
| `df.groupby("k").agg(...)` | `df.group_by("k").agg(...)` | `ds.group_by("k").agg(total=col("v").sum())` |
| `df.merge(o, on="k")` | `df.join(o, on="k")` | `ds.join(o, on="k", how="inner")` |
| `df.sort_values("a", ascending=False)` | `df.sort("a", descending=True)` | `ds.sort("a", descending=True)` |
| `df.drop_duplicates()` | `df.unique()` | `ds.distinct()` |
| `df.head(n)` | `df.head(n)` | `ds.limit(n)` / `ds.head(n)` |
| `df.explode("c")` | `df.explode("c")` | `ds.explode("c")` |
| `df.melt(...)` | `df.unpivot(...)` | `ds.unpivot(index=..., on=...)` |
| `df.pivot_table(...)` | `df.pivot(...)` | `ds.pivot(index=..., on=..., values=...)` |
| `df.fillna(0)` | `df.fill_null(0)` | `ds.fill_null(0)` |
| `np.where(c, a, b)` | `pl.when(c).then(a).otherwise(b)` | `bt.when(c).then(a).otherwise(b)` |
| `s.str.contains(p)` | `pl.col("s").str.contains(p)` | `col("s").str.contains(p)` |
| `s.dt.year` | `pl.col("t").dt.year()` | `col("t").dt.year()` |
| `df.select_dtypes("number")` | `pl.col(pl.NUMERIC_DTYPES)` | `ds.select(bt.numeric())` |
| `g.transform("sum")` | `pl.col("v").sum().over("k")` | `col("v").sum().over(partition_by=["k"])` |
| (eager) | `df.collect()` | `ds.collect()` (pyarrow `Table`) |

pandas-familiar aliases exist and are real (`ds.assign`, `ds.merge`, `ds.groupby`,
`ds.sort_values`, `ds.astype`, `ds.fillna`, `ds.dropna`, `ds.nlargest`), but prefer the
canonical spelling above — one obvious way per operation.

## Conceptual shifts that bite

- **`with_columns` vs `select`.** `select` defines the *entire* output; `with_columns`
  adds/replaces and keeps everything else. A pandas `df["c"] = ...` is
  `with_columns`, never `select`.
- **Namespace accessors carry the breadth.** Per-type methods live on `.str`, `.dt`,
  `.list`, `.struct`, `.json`, `.map` (plus `.image`/`.audio`/`.video`). `Expr` itself
  stays a thin fluent builder — if a method is missing on `col(...)`, look in a
  namespace before concluding it doesn't exist.
- **`over(...)` is keyword-only.** `col("v").sum().over(partition_by=["k"], order_by=["t"])`,
  with an optional `frame=(start, end)`. Polars' positional `.over("k")` does not port
  verbatim. `ds.window(partition_by=..., order_by=..., functions={...})` is the
  table-shaped form.
- **Null semantics are SQL's, not pandas'.** Null and `NaN` are distinct: `.is_null()`
  / `.fill_null()` for missing, `.is_nan()` / `.fill_nan()` / `bt.nanvl` for `NaN`.
  There is no `NaN`-as-missing conflation, and aggregates skip nulls.
- **No index.** There is no pandas index or `reset_index`. Row position comes from an
  explicit `ds.with_row_index()`.
- **Round-trips are cheap and symmetric.** `bt.from_pandas` / `bt.from_polars` /
  `bt.from_arrow` in, `ds.to_pandas()` / `ds.to_polars()` / `ds.to_arrow()` out — so a
  port can be incremental: move one stage at a time and hand the rest back.

## What Batcher gives you that Polars cannot

Worth saying out loud when a user asks "why port at all":

- **Adaptive re-optimization** — the plan is re-optimized at pipeline breakers on
  *measured* cardinalities, not just estimates (`docs/deep-dives/adaptive-reoptimization.md`).
- **The same code runs distributed** — `ds.collect(distributed=True)` uses the same
  mergeable `partial → combine → finalize` operators, so single-node and multi-node
  results are identical by construction.
- **Out-of-core spill** — aggregation, join, and sort spill under a memory bound
  (`ds.collect(spill=True)`), so a query bigger than RAM finishes instead of OOMing.
- **`ds.stats()`** — measured per-operator rows, time, bytes, spill, and which
  operator was the bottleneck. `ds.explain()` shows the optimized plan.

## Porting recipe

1. **Inventory the terminal ops.** Find every place the original script reads a value
   (prints, `len()`, `.iloc`, an `if` on a cell). Each becomes an explicit Batcher
   terminal call — or, better, gets deleted because it was only there to inspect.
2. **Replace the readers.** `pd.read_*` / `pl.read_*` / `pl.scan_*` → `bt.read.<fmt>`.
   Drop any manual `scan_` vs `read_` decision.
3. **Translate verbs top-down** using the table. Keep the chain in one expression; do
   not insert `.collect()` between stages — that defeats the optimizer.
4. **Move per-row Python into expressions.** `df.apply(...)` / a Python loop becomes an
   `Expr`. Only if the expression language genuinely has no answer, fall back to
   `ds.map_batches(fn)` over Arrow batches (see `docs/user-guide/udfs.md`), declaring
   `input_columns` and `output_columns`.
5. **Fix the tail.** `ds.write.parquet(path)` / `ds.to_pandas()` at the boundary where
   downstream code still expects a DataFrame.
6. **Verify** (below), then `ds.explain()` and `ds.stats()` to confirm the plan is what
   you expect and nothing degenerated into a per-batch Python stage.

### Verifying equivalence

Run both implementations and compare **order-independently** unless the query has an
explicit `sort` — a plan without an `ORDER BY` has no defined row order in either
engine, and asserting on order will produce spurious failures (and, worse, an
order-*independent* comparison cannot see a real sort bug, so sorted queries must be
compared *in order*).

The in-repo pattern is `tests/differential/conftest.py::assert_same`: coerce to
pyarrow, normalize rows to tuples, sort by a total order, compare multisets with
int↔float and float-rounding tolerance. Reuse it rather than hand-rolling:

```python
import batcher as bt
from batcher import col

old = original_polars_pipeline()            # -> pl.DataFrame
new = (
    bt.from_polars(source_df)
    .filter(col("amount") > 10)
    .group_by("city")
    .agg(total=col("amount").sum(), n=bt.count())
)
# Order-independent multiset comparison, the assert_same way.
assert sorted(map(tuple, old.rows())) == sorted(
    tuple(r.values()) for r in new.to_pylist()
)
```

For a sorted query, compare `new.to_pylist()` against the oracle **in order**.

## Gotchas / do-not

- **Do not touch a tuple in Python.** No `for row in ...`, no `.apply`, no
  element-wise math outside an `Expr`. That is the single largest performance loss in
  a naive port, and `map` / `flat_map` (per-row dict callbacks) are the trap door.
- **Do not assume ordering without `sort`.** Neither `group_by` nor a parallel scan
  preserves input order. If the original relied on pandas' incidental ordering, add an
  explicit `ds.sort(...)`.
- **Do not `collect()` in the middle of a chain.** Each `collect` is a materialization
  barrier that hides the rest of the plan from the optimizer.
- **Do not `collect()` a huge result.** Use `ds.iter_batches()` (streams pyarrow
  `RecordBatch`es in bounded memory), `ds.write.*`, or `ds.limit(n)` for a peek.
- **Do not port eager assumptions.** A `Dataset` has no `len()`; use `ds.count()`.
  Mutation in place does not exist — every op returns a new `Dataset`.
- **Do not translate `.over("k")` positionally** or assume `descending` defaults to
  pandas' `ascending=True` inverse; check the signature.

## See also

- `docs/migration/index.md` — the full pandas/Polars/PySpark mapping tables.
- `docs/user-guide/expressions.md`, `docs/api/expressions.md` — the expression surface
  and every accessor namespace.
- `docs/user-guide/udfs.md` — when a UDF is justified and what it costs.
- `docs/user-guide/window-functions.md`, `docs/user-guide/transformations.md`.
- `docs/benchmarks/vs-polars.md` — where Batcher wins and where it does not.
- `/migrate-from-daft` — the multimodal/ML-first sibling of this skill.
