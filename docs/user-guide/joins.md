# Joins

A join combines rows from two datasets on matching key values. Batcher supports
the standard relational join types plus the set operations union, intersect, and
except. Joins are mergeable, so the same operator runs on one core or across a
cluster with an identical result.

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
that do not. Neither adds columns from the right input; they filter by existence.

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

`intersect` keeps rows present in both; `except_` keeps rows in the first but not
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

## Next steps

- [Aggregations](aggregations.md): summarize joined results.
- [Window functions](window-functions.md): per-row computations over partitions.
