# Joins

A join combines rows from two datasets on matching key values. Batcher supports the
standard relational join types, plus the set operations: union, intersect, except.
Joins are mergeable, so the same operator runs on one core or across a cluster with an
identical result.

## Setup

```python
import batcher as bt

orders = bt.from_pydict(
    {
        "id": [1, 2, 3, 4, 5],
        "category": ["a", "b", "a", "b", "a"],
        "amount": [10, 20, 30, 40, 50],
    }
)
dim = bt.from_pydict({"category": ["a", "b"], "region": ["west", "east"]})
```

## join

`join` defaults to an inner join on the column named by `on`. Inner joins keep
only rows with a match in both inputs.

```python
out = orders.join(dim, on="category").select("id", "category", "region").sort("id")
print(out.to_pydict())
# {'id': [1, 2, 3, 4, 5], 'category': ['a', 'b', 'a', 'b', 'a'],
#  'region': ['west', 'east', 'west', 'east', 'west']}
```

## Join types

The `how` argument selects the join type: `"inner"`, `"left"`, `"right"`,
`"full"` (also `"outer"`), `"semi"`, and `"anti"`.

### Left, right, and full

A left join keeps every left row, filling right columns with null where there is
no match. Right and full are the mirror and the union of both sides.

```python
left = bt.from_pydict({"id": [1, 2, 3], "category": ["a", "b", "c"]})
out = left.join(dim, on="category", how="left").sort("id")
print(out.to_pydict())
# {'category': ['a', 'b', 'c'], 'id': [1, 2, 3], 'region': ['west', 'east', None]}
```

### Semi and anti

A semi join keeps left rows that have a match, and an anti join keeps left rows
that do not. Neither adds columns from the right input. They filter by existence.

```python
print(orders.join(dim, on="category", how="semi").select("id").sort("id").to_pydict())
# {'id': [1, 2, 3, 4, 5]}

print(left.join(dim, on="category", how="anti").sort("id").to_pydict())
# {'id': [3], 'category': ['c']}
```

## Join keys

Use `on` when both sides share the key name. Use `left_on` and `right_on` when the
key columns are named differently.

```python
a = bt.from_pydict({"k": [1, 2], "v": [10, 20]})
b = bt.from_pydict({"kk": [1, 2], "w": [100, 200]})
print(a.join(b, left_on="k", right_on="kk").sort("k").to_pydict())
# {'k': [1, 2], 'v': [10, 20], 'w': [100, 200]}
```

When both inputs carry a non-key column of the same name, the right side's column
gets the `suffix` (default `"_right"`).

## Set operations

Set operations combine two datasets with matching schemas.

`union` concatenates rows. Pass `distinct=True` to drop duplicates.

```python
s1 = bt.from_pydict({"x": [1, 2, 3]})
s2 = bt.from_pydict({"x": [2, 3, 4]})
print(s1.union(s2).sort("x").to_pydict())
# {'x': [1, 2, 2, 3, 3, 4]}

print(s1.union(s2, distinct=True).sort("x").to_pydict())
# {'x': [1, 2, 3, 4]}
```

`intersect` keeps rows present in both inputs. `except_` keeps rows in the first but not
the second.

```python
print(s1.intersect(s2).sort("x").to_pydict())
# {'x': [2, 3]}

print(s1.except_(s2).sort("x").to_pydict())
# {'x': [1]}
```

## Enrichment pattern

A common use is a left join that attaches lookup columns to a fact table while
keeping every fact row.

```python
enriched = orders.join(dim, on="category", how="left").sort("id")
print(enriched.to_pydict())
# {'category': ['a', 'b', 'a', 'b', 'a'], 'id': [1, 2, 3, 4, 5],
#  'amount': [10, 20, 30, 40, 50], 'region': ['west', 'east', 'west', 'east', 'west']}
```

## As-of joins

Two time series rarely share a clock. A trade lands at 10:31:07.412 and the quote it should
be priced against arrived at 10:31:07.198, so an equi-join on the timestamp finds nothing.
{py:meth}`join_asof <batcher.Dataset.join_asof>` matches each left row to the *nearest* right
row instead, which is the join every market-data, sensor-fusion, and slowly-changing-dimension
pipeline is built on.

It is left-style: every left row survives, with null right columns when nothing matched. Pass
`by=` for columns that must match exactly, so one instrument's quotes never price another's
trades.

```python
trades = bt.from_pydict({"sym": ["A", "A", "B"], "t": [10, 40, 10], "size": [100, 200, 50]})
quotes = bt.from_pydict({"sym": ["A", "A", "B"], "t": [8, 38, 1], "price": [1.0, 1.1, 9.0]})

print(trades.join_asof(quotes, on="t", by="sym").sort("sym", "t").to_pydict())
# {'sym': ['A', 'A', 'B'], 't': [10, 40, 10], 'size': [100, 200, 50],
#  'price': [1.0, 1.1, 9.0]}
```

The `B` trade at `t=10` matched a quote from `t=1`. That is the correct nearest earlier
quote, and it may also be badly stale. `tolerance` is how you say so: beyond it, the row is
left unmatched rather than carrying a value nobody would stand behind.

```python
print(trades.join_asof(quotes, on="t", by="sym", tolerance=5).sort("sym", "t").to_pydict())
# {'sym': ['A', 'A', 'B'], 't': [10, 40, 10], 'size': [100, 200, 50],
#  'price': [1.0, 1.1, None]}
```

Give `tolerance` a number for a numeric key, and a duration such as `"5m"` (or a
`datetime.timedelta`) for a timestamp or date key. Reach for it whenever a missing match is
more useful than a stale one, which in practice is most of the time.

`direction` chooses which way to look. The default `"backward"` takes the last value at or
before the left row, which is the causal reading and the one you almost always want.
`"forward"` looks the other way, for questions like "what happened next". `"nearest"` takes
whichever is closer and is right when the two clocks drift either side of each other, as with
two sensors sampling the same physical event.

```python
print(trades.join_asof(quotes, on="t", by="sym", direction="nearest").sort("sym", "t").to_pydict())
# {'sym': ['A', 'A', 'B'], 't': [10, 40, 10], 'size': [100, 200, 50],
#  'price': [1.0, 1.1, 9.0]}
```

Both `tolerance` and `"nearest"` have to subtract two keys, so they need a numeric or
temporal `on` column. A string key still orders fine for a plain backward or forward search.

## Joins on predicates

An equi-join matches equal keys. {py:meth}`join_where <batcher.Dataset.join_where>` matches on any predicates over both sides instead, such as an event falling inside an interval. It keeps every pair of rows for which all the predicates are true, which is an inner join.

```python
events = bt.from_pydict({"t": [3, 12, 25], "reading": [0.4, 0.9, 0.7]})
shifts = bt.from_pydict({"start": [0, 10, 20], "end": [10, 20, 30], "crew": ["x", "y", "z"]})
inside = events.join_where(shifts, bt.col("t") >= bt.col("start"), bt.col("t") < bt.col("end"))
print(inside.select("t", "crew").sort("t").to_pydict())
# {'t': [3, 12, 25], 'crew': ['x', 'y', 'z']}
```

A predicate names right columns by name. A right column whose name the left side already has takes the `suffix`, `_right` by default, so `bt.col("v_right")` is the right side's `v`. One or two inequalities between the sides run as a range join rather than as a filtered cartesian product, and an equality runs as a hash join.

## Update values from another dataset

{py:meth}`update <batcher.Dataset.update>` overwrites values with another dataset's where the keys match. Every column the two share, other than the key, takes the other side's value on a matched row, and a null there leaves the value alone.

```python
fixes = bt.from_pydict({"id": [2, 4], "amount": [25, None]})
print(orders.update(fixes, on="id").sort("id").to_pydict())
# {'id': [1, 2, 3, 4, 5], 'category': ['a', 'b', 'a', 'b', 'a'], 'amount': [10, 25, 30, 40, 50]}
```

Pass `include_nulls=True` when a null is itself the new value, and `how="inner"` or `how="full"` to keep only the matched rows or to add the other side's unmatched ones.

```python
print(orders.update(fixes, on="id", include_nulls=True).sort("id").select("id", "amount").to_pydict())
# {'id': [1, 2, 3, 4, 5], 'amount': [10, 25, 30, None, 50]}
```

## Pair rows by position

{py:meth}`zip <batcher.Dataset.zip>` puts datasets side by side, pairing the first row of each with the first row of the others. A dataset has no row order of its own, so `order_by` names the column that defines position. Number the rows where you read them with `with_row_index` when the data carries no such column.

```python
features = bt.from_pydict({"row": [0, 1, 2], "x": [0.1, 0.2, 0.3]})
labels = bt.from_pydict({"row": [0, 1, 2], "label": [1, 0, 1]})
print(features.zip(labels, order_by="row").to_pydict())
# {'row': [0, 1, 2], 'x': [0.1, 0.2, 0.3], 'row_1': [0, 1, 2], 'label': [1, 0, 1]}
```

The datasets must hold the same number of rows. `zip` counts them before it builds the plan and raises when they differ. When the rows share a key, `join` on that key says the same thing without depending on position.

## Lookup joins against a key-value store

Every join above reads its right side as a dataset, which means reading all of it. That is
the right thing when the dimension is small enough to broadcast or when you need a
consistent snapshot of it. It is the wrong thing when the dimension is a hundred million
rows in Redis and the data touches ten thousand of them.

{py:meth}`lookup_join() <batcher.Dataset.lookup_join>` asks the store for the keys each
batch actually contains, instead of reading the store:

```python
# docs: skip
enriched = orders.lookup_join(
    "redis://localhost:6379/0",
    on="customer_id",
    schema={"name": "string", "tier": "int64"},
    prefix="cust_",
)
```

The cost scales with the distinct keys in your data rather than with the size of the store,
which is the only reason a store far larger than memory can be joined at all. It works
unchanged single-node, distributed, and over an unbounded source, because the enrichment
happens per batch. `rocksdb:///path/to/db` reads an embedded database instead of a server.

`how="left"` keeps every row and null-fills the misses; `how="inner"` drops them. A right or
full outer join is not offered, because producing one would mean enumerating the store,
which is the scan this exists to avoid.

### Why it is fast, and what it costs

Repeated keys are the whole mechanism. Each worker keeps an LRU of what it has looked up,
so a fact stream that hits the same few thousand customers over and over pays for a few
thousand lookups rather than a few million. It also caches **absences**, which is what stops
an unmatched key from costing a round trip on every batch. On a dirty join key that is the
larger of the two wins.

| Option | Meaning |
| --- | --- |
| `cache_size` | Entries each worker holds, hits and absences together. `0` disables the cache, which is how you measure what it is buying. |
| `cache_ttl` | How long an entry stays usable (`"30s"`, `"5m"`). `None` keeps it for the life of the worker. |
| `batch_size` | Rows per lookup batch. `None`, the default, is right unless you have measured otherwise. Larger batches mean fewer round trips and cheaper assembly; see the warning below. |
| `num_workers` | How many workers issue lookups at once. This is what hides the store's latency, at the cost of one cache per worker. |

The cache is **per worker**, not shared between them, because sharing one would put a lock
in front of the thing the workers exist to do concurrently. So `num_workers=8` warms eight
caches and issues up to eight times the round trips for the same distinct keys. That is the
right trade against a store whose latency you are hiding, and the wrong one against a store
you are close to rate-limiting. Flink's lookup join has the same property.

:::{warning}
Setting `batch_size` small is the one way to make a lookup join slow. The per-batch cost is
one unit of work per *distinct key in the batch*, so a batch smaller than the distinct-key
count pays for the same keys over and over. On a two-million-row probe over five thousand
distinct keys, `batch_size=16384` runs seven times slower than the engine's own batching.
Leave it alone unless you have measured a reason not to.
:::

You give up a consistent snapshot. The store is read as it stands when each batch arrives, and `cache_ttl` bounds how stale a cached row may be. Where a point-in-time answer
is what you meant, read the dimension as a dataset and use `join`.

On a 96-core box, against an in-process store, a lookup join runs about twice as slow as the
hash join it replaces while reading 0.25% of the dimension. That ratio is the mechanism's
overhead, not its benefit: the case it is for is a store the hash join cannot read at all
without pulling every row of it over the network. Reach for it when the dimension lives
somewhere else, not to beat a join over data you already have. See
`benchmarks/internals/cache_bench.py`.

`schema` is required and cannot be inferred. A join's output columns cannot depend on which
keys the first batch happened to contain, or a batch that matched nothing would have a
different shape from the batch before it, and two workers would disagree about the shape of
the same result.

## See also

- {doc}`Aggregations </user-guide/analyze/aggregations>`: summarize joined results.
- {doc}`Window functions </user-guide/analyze/window-functions>`: per-row computations over partitions.
- {doc}`Dataset API </api/relational/dataset>`: the `join`, `join_asof`, `join_where`, `update`, `zip` and `lookup_join` reference.
- {doc}`Caching results </user-guide/operate/tuning/caching>`: the other place a key-value store speeds a query up, by holding whole results.
- {doc}`/cookbook/dataset/verbs/joins`: join types, key spellings, and the as-of join, as a script.
