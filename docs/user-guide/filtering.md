# Filtering and selection

Filtering keeps the rows that satisfy a predicate. A predicate is an `Expr` that
evaluates to a boolean column, built with comparisons and combined with boolean
operators. This page also covers null tests, set membership, ranges,
deduplication, and limiting.

## Setup

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "name": ["ann", "bob", "cy", "dan", "eve"],
        "age": [30, 25, 40, None, 22],
        "city": ["nyc", "sf", "nyc", "la", "sf"],
    }
)
```

## filter

Pass a boolean expression to `filter`. Only rows where it is true are kept.

```python
print(ds.filter(bt.col("age") > 25).to_pydict())
# {'age': [30, 40], 'city': ['nyc', 'nyc'], 'name': ['ann', 'cy']}
```

## Comparison and boolean operators

Comparisons (`==`, `!=`, `>`, `>=`, `<`, `<=`) produce boolean columns. Combine
them with `&` (and), `|` (or), and `~` (not). Parenthesize each comparison;
the operators bind tighter than you may expect.

```python
print(ds.filter((bt.col("age") > 20) & (bt.col("city") == "sf")).to_pydict())
# {'age': [25, 22], 'city': ['sf', 'sf'], 'name': ['bob', 'eve']}
```

```python
print(ds.filter(~(bt.col("city") == "nyc")).to_pydict())
# {'age': [25, None, 22], 'city': ['sf', 'la', 'sf'], 'name': ['bob', 'dan', 'eve']}
```

## is_in

`is_in` keeps rows whose value is in a given collection.

```python
print(ds.filter(bt.col("city").is_in(["nyc", "la"])).to_pydict())
# {'age': [30, 40, None], 'city': ['nyc', 'nyc', 'la'], 'name': ['ann', 'cy', 'dan']}
```

## between

`between` is an inclusive range test on both bounds.

```python
print(ds.filter(bt.col("age").between(23, 35)).to_pydict())
# {'age': [30, 25], 'city': ['nyc', 'sf'], 'name': ['ann', 'bob']}
```

## Null tests

`is_null` keeps rows where a column is null; `is_not_null` keeps the rest.

```python
print(ds.filter(bt.col("age").is_null()).to_pydict())
# {'age': [None], 'city': ['la'], 'name': ['dan']}

print(ds.filter(bt.col("age").is_not_null()).to_pydict())
# {'age': [30, 25, 40, 22], 'city': ['nyc', 'sf', 'nyc', 'sf'], 'name': ['ann', 'bob', 'cy', 'eve']}
```

## distinct

`distinct` removes duplicate rows across all columns.

```python
cities = bt.from_pydict({"city": ["nyc", "sf", "nyc", "la", "sf"]})
print(cities.distinct().sort("city").to_pydict())
# {'city': ['la', 'nyc', 'sf']}
```

## limit and head

`limit(n, offset=0)` keeps `n` rows starting after `offset`. `head(n)` is the
common case of the first `n` rows.

```python
print(ds.sort("name").limit(2).to_pydict())
# {'age': [30, 25], 'city': ['nyc', 'sf'], 'name': ['ann', 'bob']}

print(ds.sort("name").limit(2, offset=1).to_pydict())
# {'age': [25, 40], 'city': ['sf', 'nyc'], 'name': ['bob', 'cy']}

print(ds.sort("name").head(3).to_pydict())
# {'age': [30, 25, 40], 'city': ['nyc', 'sf', 'nyc'], 'name': ['ann', 'bob', 'cy']}
```

## Chaining

Filters and the operators above compose into a single lazy plan. The optimizer
pushes predicates toward the source where it can.

```python
result = (
    ds.filter(bt.col("age").is_not_null())
    .filter(bt.col("city").is_in(["nyc", "sf"]))
    .sort("age", descending=True)
    .head(2)
)
print(result.to_pydict())
# {'age': [40, 30], 'city': ['nyc', 'nyc'], 'name': ['cy', 'ann']}
```

## Next steps

- [Aggregations](aggregations.md): group and summarize the rows you kept.
- [Joins](joins.md): combine datasets and use semi/anti joins to filter by
  existence.
