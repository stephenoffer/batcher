# Sorting

Sorting is a pipeline breaker: the engine has to see every row before it can emit the
first one. That also makes it one of the few operators where a bug hides in plain sight.
A result that is *almost* ordered still looks right in a `head(5)`, and an
order-independent assertion in your test cannot see the difference at all. This page is
the contract: what `sort` guarantees, where nulls and NaN land, and when not to sort at
all.

## Setup

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "name": ["ann", "bob", "cy", "dan", "eve"],
        "team": ["red", "blue", "red", "blue", "red"],
        "score": [30, 25, 30, 12, 41],
    }
)
```

## sort

`sort(*by, descending=False, nulls_first=False)` takes column names or expressions.

```python
print(ds.sort("score").select("name", "score").to_pydict())
# {'name': ['dan', 'bob', 'ann', 'cy', 'eve'], 'score': [12, 25, 30, 30, 41]}
```

`descending` and `nulls_first` are either one bool for every key or a list aligned with
the keys. A multi-key sort with mixed directions is the common reporting shape: group
label ascending, metric descending. The SQL spelling lowers to the same plan, so pick
whichever reads better.

::::{tab-set}
:::{tab-item} DataFrame

```python
by_team = ds.sort("team", "score", descending=[False, True])
print(by_team.select("team", "score", "name").to_pydict())
# {'team': ['blue', 'blue', 'red', 'red', 'red'], 'score': [25, 12, 41, 30, 30],
#  'name': ['bob', 'dan', 'eve', 'ann', 'cy']}
```

:::

:::{tab-item} SQL

```python
print(bt.sql("SELECT team, score, name FROM t ORDER BY team ASC, score DESC", t=ds).to_pydict())
# {'team': ['blue', 'blue', 'red', 'red', 'red'], 'score': [25, 12, 41, 30, 30],
#  'name': ['bob', 'dan', 'eve', 'ann', 'cy']}
```

:::
::::

A sort key can be an expression, which is how you sort by something you never want in
the output.

```python
print(ds.sort(bt.col("name").str.len(), descending=True).select("name").to_pydict())
# {'name': ['ann', 'bob', 'dan', 'eve', 'cy']}
```

## Ties are not stable, so break them yourself

Two rows with the same key can come back in either order, and the order can change
between a sequential run, a multi-core run, and a distributed one. `ann` and `cy` both
score 30 above; nothing promises which comes first. If the order of tied rows matters
(and it usually does, the moment you `head(n)` or write the result), add a tiebreaker
key that is unique.

```python
print(ds.sort("score", "name", descending=[True, False]).select("score", "name").to_pydict())
# {'score': [41, 30, 30, 25, 12], 'name': ['eve', 'ann', 'cy', 'bob', 'dan']}
```

This is the same discipline SQL demands: `ORDER BY score DESC` with ties is a
non-deterministic result, whatever engine you run it on.

## Nulls sort last by default, in both directions

`nulls_first=False` (the default) puts nulls at the end whether you sort ascending or
descending. Nulls are not "smallest", they are "absent". Flip `nulls_first=True` to get
them up front.

```python
scores = bt.from_pydict({"name": ["a", "b", "c", "d"], "score": [3, None, 1, 2]})

print(scores.sort("score").to_pydict())
# {'name': ['c', 'd', 'a', 'b'], 'score': [1, 2, 3, None]}

print(scores.sort("score", descending=True).to_pydict())
# {'name': ['a', 'd', 'c', 'b'], 'score': [3, 2, 1, None]}

print(scores.sort("score", nulls_first=True).to_pydict())
# {'name': ['b', 'c', 'd', 'a'], 'score': [None, 1, 2, 3]}
```

That matters for `head`: `sort("score", descending=True).head(1)` gives you the
maximum, never a null row. If you want nulls out of the result entirely, say so with
`drop_nulls`. Do not rely on where the sort happens to put them.

## Floats: NaN is a value, null is absence

NaN is not null and it is not "unordered". Batcher sorts it as larger than every
number, so ascending puts it after `+inf` and before the nulls; descending puts it
first. `-0.0` and `0.0` compare equal, so their relative order is a tie, the same as any
other tie.

```python
floats = bt.from_pydict({"x": [1.0, float("nan"), -0.0, 0.0, None, -1.0]})

print(floats.sort("x").to_pydict())
# {'x': [-1.0, -0.0, 0.0, 1.0, nan, None]}

print(floats.sort("x", descending=True).to_pydict())
# {'x': [nan, 1.0, 0.0, -0.0, -1.0, None]}
```

Put the two flags together and the whole ordering is this:

| Call | Where the values land |
| --- | --- |
| `sort("x")` | numbers ascending, then NaN, then nulls |
| `sort("x", descending=True)` | NaN, then numbers descending, then nulls |
| `sort("x", nulls_first=True)` | nulls, then numbers ascending, then NaN |
| `sort("x", descending=True, nulls_first=True)` | nulls, then NaN, then numbers descending |

:::{warning}
NaN is *larger than every number*, so it sorts last in ascending order and first in
descending order, immediately before the nulls in both. A `head(10)` over a descending
float sort therefore returns ten NaN rows the moment your data has ten of them, and the
result looks exactly like a working query. If NaN in your data means "no measurement"
rather than a real value, convert it before you sort: {py:meth}`.fill_nan(...) <batcher.plan.expr_ir.core.Expr.fill_nan>` replaces IEEE
NaN, which `.fill_null(...)` never touches.
:::

## Don't sort when you want the top n

`sort(...).head(k)` asks the engine for a total order and then throws almost all of it
away. `top_k(k, by=...)` keeps a bounded heap instead, so memory is O(k) rather than
O(rows) and there is nothing to spill.

:::{tip}
Reach for the narrowest operator that answers the question you actually asked.

| What you want | Use | What it costs |
| --- | --- | --- |
| A total order over the whole relation | `sort(...)` | a breaker; spills when it does not fit |
| The largest or smallest `k` rows | `top_k(k, by=...)` | a bounded heap, O(k) |
| The top n *within* each group | `row_number()` / `rank()` over a window | one partitioned pass |
:::

```python
print(ds.top_k(2, by="score").select("name", "score").to_pydict())
# {'name': ['eve', 'ann'], 'score': [41, 30]}
```

`descending=True` is the default there, since top means largest. Pass `descending=False`
for the bottom `k`. The same rule applies to `limit`: the optimizer can push a limit into a
sort, but it cannot push one into a sort you wrote as a separate materialized step.

Sorting to make a *later* operator cheaper is usually wasted too. A {py:meth}`group_by <batcher.Dataset.group_by>` hashes,
it does not need sorted input, and a `join` builds a hash table. Sort at the end, once,
for presentation or for the file layout you are writing.

## What makes a sort fast

The key's *type* decides which algorithm runs, and the difference is large enough to be worth
knowing when you have a choice of key.

A fixed-width key, such as an integer, a date or a timestamp, sorts by a counting sort over an
order-preserving integer, which is linear in the rows. A string key has to compare bytes. Where
a column is available in both forms, ordering by the fixed-width one is materially cheaper, and
ordering by an `id` and rendering the label afterwards is cheaper still.

Sorting by **several** fixed-width keys is not more expensive than sorting by one, as long as
their combined value ranges are narrow. The engine measures each key's live range and packs the
whole tuple into a single integer, so `ORDER BY <date>, <priority>` costs about what
`ORDER BY <date>` costs. Mixed directions are free, and so is a key that turns out to be
constant.

| Leading key | What runs |
| --- | --- |
| One integer, date, or timestamp | counting sort over an integer transform |
| Several integer/temporal keys, ranges narrow enough to pack | the same counting sort, over one packed integer |
| A string or a binary column, up to 8 bytes wide | the same counting sort, over the key's own bytes |
| A wider string or binary column | byte comparisons, with the first 8 bytes carried inline |
| A float, or keys too wide to pack | a comparison sort over Arrow's row encoding |

```python
import datetime as dt

events = bt.from_pydict(
    {
        "day": [dt.date(2026, 3, 2), dt.date(2026, 3, 1), dt.date(2026, 3, 2)],
        "priority": [2, 1, 1],
        "label": ["c", "a", "b"],
    }
)
print(events.sort("day", "priority", descending=[False, True]).to_pydict()["label"])
# ['a', 'c', 'b']
```

None of this changes the answer, only the time. If the key you have is a string, sort on it.
The point is to reach for a fixed-width key when one is genuinely available, rather than to
reshape data around the sort.

## Sorting large results: spill

A sort that does not fit in the memory budget spills sorted runs to disk and merges
them. This is on by default under {py:meth}`collect(spill=True) <batcher.Dataset.collect>` and the out-of-core path, and
the merged result is exactly the in-memory result: same order, same rows.

```python
big = bt.range(0, 50_000).with_columns(k=(bt.col("value") * 7919) % 1000)
out = big.sort("k", "value", descending=[True, False]).collect(spill=True)
first = out.column("k")[0].as_py()
last = out.column("k")[-1].as_py()
print(first, last, out.num_rows)
# 999 0 50000
```

:::{warning}
Verify a spilled sort with an order-*dependent* check, as above. Comparing the sorted
result to the expected result as a multiset passes even when the rows come back in an
arbitrary order. That is exactly how a real `descending=True` spill bug once shipped
green: every gate was passing, and the test could not see the one thing that was wrong.
:::

## Sorting and writing

Sort order survives into the files you write, so a sorted write gives readers cheap
range pruning on the sort key later. Pair it with `repartition` when you care about
file layout.

```python
# docs: skip
ds.sort("team", "score").repartition(by="team").write("out/")
```

## See also

- {doc}`Filtering </user-guide/transform/rows/filtering>`: cut rows before you order them.
- {doc}`Window functions </user-guide/analyze/window-functions>`: `row_number`/`rank` over an ordered
  partition, which is the right tool for "top n per group".
- {doc}`Performance </user-guide/operate/tuning/performance>`: the spill path and the memory budget.
- {doc}`Sort internals </architecture/deep-dives/operators/sort-internals>`: the run generation and k-way merge
  that make the spilled result identical to the in-memory one.
- {doc}`Dataset API </api/relational/dataset>`: the `sort`, `top_k`, and `limit` reference.
- {doc}`Top k per group </cookbook/analytics/aggregates/top-k-per-group>`: the window recipe, worked
  end to end.
- {doc}`DuckDB comparison </benchmarks/comparisons/vs-duckdb>`: where sort stands against the
  single-node bar.
- {doc}`/cookbook/expressions/scalar/sorting_and_ranking`: the sort edge cases that hide bugs, as a runnable script.
