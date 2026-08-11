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

## See also

- {doc}`Aggregations </user-guide/analyze/aggregations>`: summarize joined results.
- {doc}`Window functions </user-guide/analyze/window-functions>`: per-row computations over partitions.
- {doc}`Dataset API </api/relational/dataset>`: the `join` and `join_asof` reference.
- {doc}`/cookbook/dataset/verbs/joins`: join types, key spellings, and the as-of join, as a script.
