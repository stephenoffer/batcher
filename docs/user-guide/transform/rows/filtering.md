# Filtering and selection

Filtering keeps the rows that satisfy a predicate. A predicate is an {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` that
evaluates to a boolean column, built with comparisons and combined with boolean
operators. Null tests, set membership, ranges, deduplication and limiting all follow
from the same idea.

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

## Writing a predicate

Pass a boolean expression to `filter`. Only rows where it is true are kept.

```python
print(ds.filter(bt.col("age") > 25).to_pydict())
# {'name': ['ann', 'cy'], 'age': [30, 40], 'city': ['nyc', 'nyc']}
```

Comparisons (`==`, `!=`, `>`, `>=`, `<`, `<=`) produce boolean columns. Combine
them with `&` (and), `|` (or), and `~` (not). Parenthesize each comparison. The
operators bind tighter than you may expect.

```python
print(ds.filter((bt.col("age") > 20) & (bt.col("city") == "sf")).to_pydict())
# {'name': ['bob', 'eve'], 'age': [25, 22], 'city': ['sf', 'sf']}
```

`~` negates a whole predicate, so it takes parentheses too.

```python
print(ds.filter(~(bt.col("city") == "nyc")).to_pydict())
# {'name': ['bob', 'dan', 'eve'], 'age': [25, None, 22], 'city': ['sf', 'la', 'sf']}
```

## Membership, ranges, and nulls

Three predicates cover most of what a chain of comparisons would otherwise spell out.
{py:meth}`is_in <batcher.plan.expr_ir.core.Expr.is_in>` keeps rows whose value is in a given collection.

```python
print(ds.filter(bt.col("city").is_in(["nyc", "la"])).to_pydict())
# {'name': ['ann', 'cy', 'dan'], 'age': [30, 40, None], 'city': ['nyc', 'nyc', 'la']}
```

`between` is an inclusive range test on both bounds.

```python
print(ds.filter(bt.col("age").between(23, 35)).to_pydict())
# {'name': ['ann', 'bob'], 'age': [30, 25], 'city': ['nyc', 'sf']}
```

Null gets its own pair of methods rather than a comparison, because a comparison
against null answers null rather than true, and a filter keeps only the rows that are
true. {py:meth}`is_null <batcher.plan.expr_ir.core.Expr.is_null>` keeps rows where a column is null, and {py:meth}`is_not_null <batcher.plan.expr_ir.core.Expr.is_not_null>` keeps the rest.

```python
print(ds.filter(bt.col("age").is_null()).to_pydict())
# {'name': ['dan'], 'age': [None], 'city': ['la']}

print(ds.filter(bt.col("age").is_not_null()).to_pydict())
# {'name': ['ann', 'bob', 'cy', 'eve'], 'age': [30, 25, 40, 22], 'city': ['nyc', 'sf', 'nyc', 'sf']}
```

That is also why `age > 25` dropped `dan` at the top of the page. The comparison never
said false. It said nothing.

A NaN is a different thing from a null: it is a float that is not a number. {py:meth}`drop_nans <batcher.Dataset.drop_nans>` drops the rows holding one in any floating-point column, or in the columns you name, and keeps the rows whose value is null.

```python
readings = bt.from_pydict({"sensor": ["a", "b", "c"], "value": [1.5, float("nan"), None]})
print(readings.drop_nans().to_pydict())
# {'sensor': ['a', 'c'], 'value': [1.5, None]}
```

## One dataset per key

{py:meth}`partition_by <batcher.Dataset.partition_by>` splits a dataset into one dataset per distinct key value, returned as a dict keyed by tuples. It finds the keys with an eager `distinct`, and each part is a lazy filter that reads the input again, so it suits a handful of keys, such as one output per city.

```python
by_city = ds.partition_by("city")
print({key: part.count() for key, part in by_city.items()})
# {('la',): 1, ('nyc',): 2, ('sf',): 2}
```

To compute a result per group, `group_by` does the work in one pass instead.

## Trimming the result

`distinct` removes duplicate rows across all columns.

```python
cities = bt.from_pydict({"city": ["nyc", "sf", "nyc", "la", "sf"]})
print(cities.distinct().sort("city").to_pydict())
# {'city': ['la', 'nyc', 'sf']}
```

`limit(n, offset=0)` keeps `n` rows starting after `offset`, and `head(n)` is the
common case of the first `n` rows. Both examples below sort first on purpose. Over a
relation with no order a limit keeps some `n` rows rather than a defined `n`, and which
ones you get is a property of the schedule rather than of the query.

```python
print(ds.sort("name").limit(2).to_pydict())
# {'name': ['ann', 'bob'], 'age': [30, 25], 'city': ['nyc', 'sf']}

print(ds.sort("name").limit(2, offset=1).to_pydict())
# {'name': ['bob', 'cy'], 'age': [25, 40], 'city': ['sf', 'nyc']}

print(ds.sort("name").head(3).to_pydict())
# {'name': ['ann', 'bob', 'cy'], 'age': [30, 25, 40], 'city': ['nyc', 'sf', 'nyc']}
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
# {'name': ['cy', 'ann'], 'age': [40, 30], 'city': ['nyc', 'nyc']}
```

## See also

- {doc}`Aggregations </user-guide/analyze/aggregations>`: group and summarize the rows you kept.
- {doc}`Joins </user-guide/analyze/joins>`: combine datasets and use semi/anti joins to filter by
  existence.
- {doc}`Dataset API </api/relational/dataset>`: the `filter`, `distinct`, `sample`, and `limit`
  reference.
- {doc}`/cookbook/expressions/scalar/conditionals`: branching inside an expression, as a runnable script.
