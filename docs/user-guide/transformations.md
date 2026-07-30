# Transformations

Transformations reshape the columns of a dataset. You choose which columns survive,
derive new ones, rename or drop what is left. Each call returns a new `Dataset` and
runs nothing until a terminal operation. Column work is expressed with `Expr` values
and evaluated in the Rust data plane.

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
arguments and derived columns as keyword arguments. The result contains exactly
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

## Column selectors

The transforms above name columns one at a time. A **selector** stands for *every*
column matching a rule: a name, a name pattern, an Arrow dtype. One written
expression then becomes as many computed columns as match. Because a selector is an
`Expr` leaf (`Selector`), the whole scalar algebra composes onto it, and it works
anywhere a projection is built (`select`, `with_columns`, `drop`).

`bt.exclude(...)` selects every column except the named ones, the mirror image of
listing the ones you want to keep:

```python
print(ds.select(bt.exclude("qty")).columns)
# ['name', 'price']
```

The dtype selectors pick columns by kind. `bt.numeric()` covers integer, float, and
decimal, and `bt.integer()`, `bt.floating()`, `bt.string()`, and `bt.boolean()` narrow
that to one kind each. `bt.temporal()` covers date, time, timestamp, and duration, and
`bt.by_dtype(pa.int32(), ...)` matches exact Arrow types.

The name selectors match column *names*. `bt.matches(regex)` matches by regular
expression, and `bt.starts_with(...)`, `bt.ends_with(...)`, and `bt.contains(...)` match
by literal prefix, suffix, and substring. Each of those three accepts several arguments.
`bt.all()` matches every column.

```python
import datetime

events = bt.from_pydict(
    {
        "user": ["u1", "u2"],
        "amount": [10.0, 20.0],
        "day": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
    }
)
print(events.select(bt.floating()).columns)  # ['amount']
print(events.select(bt.temporal()).columns)  # ['day']
```

Because a selector is an expression, composing scalar work onto it computes over
every matched column at once, and the `.name` accessor renames the expanded
outputs:

```python
print(ds.select(bt.numeric().name.prefix("n_")).to_pydict())
# {'n_price': [10.0, 20.0, 30.0], 'n_qty': [1, 2, 3]}
```

`alias(...)` names exactly one column, so it cannot name a selector that matched
several. The `.name` accessor derives each output name from its matched input name
instead: `.name.prefix(...)`, `.name.suffix(...)`, `.name.to_lowercase()`,
`.name.to_uppercase()`, `.name.map(fn)` for an arbitrary rule, and `.name.keep()`
to state the default explicitly. Normalizing a messy header row is a one-liner:

```python
messy = bt.from_pydict({"User ID": [1], "Signup Date": ["2024-01-01"]})
print(messy.select(bt.all().name.map(lambda c: c.lower().replace(" ", "_"))).columns)
# ['user_id', 'signup_date']

print(messy.select(bt.all().name.to_uppercase()).columns)
# ['USER ID', 'SIGNUP DATE']
```

Put `.name` *before* the scalar work. It is an accessor on the selector, not on the
computed expression, so read `bt.numeric().name.prefix("n_").round(2)` as "the
numeric columns, prefixed, rounded".

Because renaming happens per matched column, `with_columns` replaces a column in
place when the output name is unchanged, and adds a new one when it changes:

```python
print(ds.with_columns(bt.floating() * 2).to_pydict()["price"])
# [20.0, 40.0, 60.0] — price replaced in place

print(ds.with_columns(bt.floating().name.suffix("_x2") * 2).columns)
# ['name', 'price', 'qty', 'price_x2'] — price kept, a new column added
```

Selectors compose with set algebra: `|` (union), `&` (intersection), `-`
(difference), `~` (complement). Name a group by describing it.

```python
print(ds.select(bt.numeric() - bt.floating()).columns)
# ['qty'] — the numeric columns that are not floats
```

## Choosing between select and with_columns

One obvious tool per intent. `select` defines the complete set of output columns.
`with_columns` and `with_column` add to or replace columns in the set you already have.
Casting is an expression method taking an Arrow type name, and works inside either one:

```python
print(ds.with_columns(qty=bt.col("qty").cast("float64")).to_pydict())
# {'name': ['alice', 'bob', 'carol'], 'price': [10.0, 20.0, 30.0], 'qty': [1.0, 2.0, 3.0]}
```

## Reusing your own transformations

`pipe` applies a function to the dataset and returns its result, so a step you
wrote yourself reads in the order it runs instead of inside-out. It adds no plan
node and stays lazy when your function does.

```python
def with_total(frame, tax=0.0):
    return frame.with_columns(total=bt.col("price") * bt.col("qty") * (1 + tax))


print(ds.pipe(with_total, tax=0.5).filter(bt.col("total") > 20).to_pydict()["total"])
# [60.0, 135.0]
```

Without `pipe` the same pipeline reads backwards. `with_total(ds).filter(...)` puts the
first step in the middle. Reach for `pipe` whenever a chain grows a step that has no
built-in method.

## Flattening nested data

Semistructured data arrives with lists and structs inside columns. Two relational
transforms flatten them, and they compose to unnest arbitrarily deep shapes.

`explode` turns a **list** column into one row per element, repeating the other
columns, the same as SQL `UNNEST`. Empty and null lists drop out.

```python
nested = bt.from_pydict({"id": [1, 2], "tags": [["a", "b"], ["c"]]})
print(nested.explode("tags").to_pydict())
# {'id': [1, 1, 2], 'tags': ['a', 'b', 'c']}
```

`unnest` promotes a **struct** column's fields to top-level columns, replacing the
struct in place.

```python
import pyarrow as pa

people = bt.from_arrow(
    pa.table({"person": pa.array([{"name": "Ann", "age": 30}, {"name": "Bo", "age": 25}])})
)
print(people.unnest("person").to_pydict())
# {'age': [30, 25], 'name': ['Ann', 'Bo']}
```

To reach a single field without flattening the whole struct, use the `.struct` and
`.json` accessors (see {doc}`Expressions <expressions>`) in a `select`: `.struct.field(name)`
projects one struct field, and `.json.extract_int(path)` (and its typed siblings)
reads a value from a JSON-text column by JSONPath without a decode step. Explode a
list of structs first, then `unnest`, to flatten a nested array of records into a
flat table.

## See also

- {doc}`Filtering <filtering>`: row selection, deduplication, limits.
- {doc}`Aggregations <aggregations>`: grouped and global summaries.
- {doc}`Dataset API <../api/dataset>`: the full method reference for every transformation.
