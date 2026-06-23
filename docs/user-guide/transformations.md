# Transformations

Transformations reshape the columns of a dataset: choosing which columns survive,
deriving new ones, and renaming or dropping them. Each returns a new `Dataset` and
runs nothing until a terminal operation. Column work is expressed with `Expr`
values and evaluated in the Rust data plane.

## Setup

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "name": ["alice", "bob", "carol"],
        "price": [10.0, 20.0, 30.0],
        "qty": [1, 2, 3],
    }
)
```

## select

`select` chooses the full output. Pass existing column names as positional
arguments and derived columns as keyword arguments; the result contains exactly
the columns you name.

```python
out = ds.select("name", total=bt.col("price") * bt.col("qty"))
print(out.to_pydict())
# {'name': ['alice', 'bob', 'carol'], 'total': [10.0, 40.0, 90.0]}
```

Because `select` defines the entire output, it is also how you drop down to a
subset of columns:

```python
print(ds.select("name", "price").to_pydict())
# {'name': ['alice', 'bob', 'carol'], 'price': [10.0, 20.0, 30.0]}
```

## with_columns

`with_columns` adds or replaces columns and keeps every other column. New columns
are passed as keyword arguments. Adding several in one call evaluates them in a
single pass.

```python
out = ds.with_columns(
    total=bt.col("price") * bt.col("qty"),
    name_upper=bt.col("name").str.upper(),
)
print(out.to_pydict())
# {'name': ['alice', 'bob', 'carol'], 'price': [10.0, 20.0, 30.0], 'qty': [1, 2, 3],
#  'total': [10.0, 40.0, 90.0], 'name_upper': ['ALICE', 'BOB', 'CAROL']}
```

When a keyword names an existing column, the new expression replaces it:

```python
out = ds.with_columns(price=bt.col("price") * 1.1)
print(out.to_pydict())
# {'name': ['alice', 'bob', 'carol'], 'price': [11.0, 22.0, 33.0], 'qty': [1, 2, 3]}
```

## with_column

`with_column` adds or replaces a single column by name. It is the one-column form
of `with_columns`.

```python
out = ds.with_column("subtotal", bt.col("price") * bt.col("qty"))
print(out.to_pydict())
# {'name': ['alice', 'bob', 'carol'], 'price': [10.0, 20.0, 30.0], 'qty': [1, 2, 3],
#  'subtotal': [10.0, 40.0, 90.0]}
```

## drop

`drop` removes the named columns and keeps the rest.

```python
print(ds.drop("qty").to_pydict())
# {'name': ['alice', 'bob', 'carol'], 'price': [10.0, 20.0, 30.0]}
```

## rename

`rename` takes a mapping of old name to new name. Columns not in the mapping are
unchanged.

```python
print(ds.rename({"qty": "quantity"}).to_pydict())
# {'name': ['alice', 'bob', 'carol'], 'price': [10.0, 20.0, 30.0], 'quantity': [1, 2, 3]}
```

## Choosing between select and with_columns

There is one obvious tool for each intent. Use `select` when you want to define
the complete set of output columns, and `with_columns` (or `with_column`) when you
want to add to or replace columns in the existing set. Casting is an expression
method that takes an Arrow type name, applied inside either one:

```python
print(ds.with_columns(qty=bt.col("qty").cast("float64")).to_pydict())
# {'name': ['alice', 'bob', 'carol'], 'price': [10.0, 20.0, 30.0], 'qty': [1.0, 2.0, 3.0]}
```

## Next steps

- [Filtering](filtering.md): row selection, deduplication, and limits.
- [Aggregations](aggregations.md): grouped and global summaries.
