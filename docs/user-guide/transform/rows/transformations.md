# Transformations

This page covers the verbs that shape a dataset's columns: choosing which survive, deriving new ones, renaming and dropping, matching many columns at once with selectors, and flattening nested data. Each call returns a new {py:class}`Dataset <batcher.Dataset>` and runs nothing until a terminal operation, so a chain of them reaches the optimizer as one plan. The column work itself is written as {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` values and evaluated in the Rust data plane.

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

## Choosing and deriving columns

Two verbs cover almost everything, and the difference between them is what happens to the columns you didn't mention. `select` chooses the full output. Pass existing column names as positional arguments and derived columns as keyword arguments. The result contains exactly the columns you name.

```python
out = ds.select("name", total=bt.col("price") * bt.col("qty"))
print(out.to_pydict())
# {'name': ['alice', 'bob', 'carol'], 'total': [10.0, 40.0, 90.0]}
```

Because `select` defines the entire output, it is also how you drop down to a subset of columns:

```python
print(ds.select("name", "price").to_pydict())
# {'name': ['alice', 'bob', 'carol'], 'price': [10.0, 20.0, 30.0]}
```

{py:meth}`with_columns <batcher.Dataset.with_columns>` is the other half. It adds or replaces columns and keeps every other one. New columns are passed as keyword arguments. Adding several in one call evaluates them in a single pass.

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

## Dropping and renaming

`drop` removes the named columns and keeps the rest.

```python
print(ds.drop("qty").to_pydict())
# {'name': ['alice', 'bob', 'carol'], 'price': [10.0, 20.0, 30.0]}
```

`rename` takes a mapping of old name to new name. Columns not in the mapping are unchanged.

```python
print(ds.rename({"qty": "quantity"}).to_pydict())
# {'name': ['alice', 'bob', 'carol'], 'price': [10.0, 20.0, 30.0], 'quantity': [1, 2, 3]}
```

## Column selectors

The transforms above name columns one at a time. A *selector* stands for every column matching a rule, such as a name, a name pattern, or an Arrow dtype. One written expression then becomes as many computed columns as match. A {py:class}`Selector <batcher.plan.expr_ir.selectors.Selector>` is an `Expr` leaf, so the whole scalar algebra composes onto it, and it works anywhere a projection is built: `select`, `with_columns`, `drop`, and `group_by().agg()`.

{py:func}`bt.exclude(...) <batcher.exclude>` selects every column except the named ones, the mirror image of listing the ones you want to keep:

```python
print(ds.select(bt.exclude("qty")).columns)
# ['name', 'price']
```

The dtype selectors pick columns by kind. {py:func}`bt.numeric() <batcher.numeric>` covers integer, float, and decimal, and {py:func}`bt.integer() <batcher.integer>`, {py:func}`bt.floating() <batcher.floating>`, {py:func}`bt.string() <batcher.string>`, and {py:func}`bt.boolean() <batcher.boolean>` narrow that to one kind each. {py:func}`bt.temporal() <batcher.temporal>` covers date, time, timestamp, and duration, and {py:func}`bt.by_dtype(pa.float64(), ...) <batcher.by_dtype>` matches Arrow types as the engine stores them, taking a `pyarrow` type or its name.

The name selectors match column *names*. {py:func}`bt.matches(regex) <batcher.matches>` matches by regular expression, and {py:func}`bt.starts_with(...) <batcher.starts_with>`, {py:func}`bt.ends_with(...) <batcher.ends_with>`, and {py:func}`bt.contains(...) <batcher.contains>` match by literal prefix, suffix, and substring. Each of those three accepts several arguments. {py:func}`bt.all() <batcher.all>` matches every column.

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

Because a selector is an expression, composing scalar work onto it computes over every matched column at once, and the {py:class}`.name <batcher.plan.expr_ir.selectors.core._SelectorNameNamespace>` accessor renames the expanded outputs:

```python
print(ds.select(bt.numeric().name.prefix("n_")).to_pydict())
# {'n_price': [10.0, 20.0, 30.0], 'n_qty': [1, 2, 3]}
```

{py:meth}`alias(...) <batcher.plan.expr_ir.core.Expr.alias>` names exactly one column, so it cannot name a selector that matched several. The `.name` accessor derives each output name from its matched input name instead: {py:meth}`.name.prefix(...) <batcher.plan.expr_ir.selectors.core._SelectorNameNamespace.prefix>`, {py:meth}`.name.suffix(...) <batcher.plan.expr_ir.selectors.core._SelectorNameNamespace.suffix>`, {py:meth}`.name.to_lowercase() <batcher.plan.expr_ir.selectors.core._SelectorNameNamespace.to_lowercase>`, {py:meth}`.name.to_uppercase() <batcher.plan.expr_ir.selectors.core._SelectorNameNamespace.to_uppercase>`, {py:meth}`.name.map(fn) <batcher.plan.expr_ir.selectors.core._SelectorNameNamespace.map>` for an arbitrary rule, and {py:meth}`.name.keep() <batcher.plan.expr_ir.selectors.core._SelectorNameNamespace.keep>` to state the default explicitly. Normalizing a messy header row is a one-liner:

```python
messy = bt.from_pydict({"User ID": [1], "Signup Date": ["2024-01-01"]})
print(messy.select(bt.all().name.map(lambda c: c.lower().replace(" ", "_"))).columns)
# ['user_id', 'signup_date']

print(messy.select(bt.all().name.to_uppercase()).columns)
# ['USER ID', 'SIGNUP DATE']
```

The accessor works on either side of the scalar work. `bt.numeric().name.prefix("n_").round(2)` and `bt.numeric().round(2).name.prefix("n_")` both mean "the numeric columns, rounded, prefixed", because the rename is recorded on the selector inside the expression.

Because renaming happens per matched column, `with_columns` replaces a column in place when the output name is unchanged, and adds a new one when it changes:

```python
print(ds.with_columns(bt.floating() * 2).to_pydict()["price"])  # price replaced in place
# [20.0, 40.0, 60.0]

print(ds.with_columns(bt.floating().name.suffix("_x2") * 2).columns)  # a new column added
# ['name', 'price', 'qty', 'price_x2']

print(ds.with_columns((bt.numeric() * 2).name.suffix("_x2")).columns)  # .name after the work
# ['name', 'price', 'qty', 'price_x2', 'qty_x2']
```

Selectors compose with set algebra: `|` for union, `&` for intersection, `-` for difference, `^` for the columns in exactly one of the two, and `~` for complement. Name a group by describing it. A plain {py:obj}`bt.col("x") <batcher.col>` on the other side of a set operator is read as the selector for that one column, as Polars reads it. Any other operand is arithmetic or logic over every matched column, so `bt.numeric() - 1` subtracts one from each.

```python
print(ds.select(bt.numeric() - bt.floating()).columns)  # numeric columns that are not floats
# ['qty']

print(ds.select(bt.numeric() ^ bt.floating()).columns)  # in one of the two, not both
# ['qty']

print(ds.select(bt.numeric() - bt.col("qty")).columns)  # col("qty") as a one-column selector
# ['price']

print(ds.select(bt.numeric() - 1).to_pydict())  # a scalar operand is arithmetic
# {'price': [9.0, 19.0, 29.0], 'qty': [0, 1, 2]}
```

A set operation keeps no rename, so combining or complementing a selector that already carries `.name` raises a `PlanError`. Rename after combining: `(~bt.numeric()).name.prefix("p_")`. A selector's own `.exclude(...)` is the exception: it only narrows the selection, so it keeps the rename.

### Selectors in aggregations

An aggregate over a selector expands to one aggregate per matched column, in `group_by().agg()` and in a whole-frame `select`. The group keys are never aggregated over. Several aggregates over one selector need distinct names, which `.name` gives them. On an aggregate, `.name` is the same rename accessor, as in Polars:

```python
sales = bt.from_pydict({"region": ["n", "n", "s"], "price": [10.0, 20.0, 30.0], "qty": [1, 2, 3]})
summary = sales.group_by("region").agg(
    bt.numeric().sum().name.suffix("_sum"),
    bt.numeric().max().name.prefix("max_"),
)
print(summary.sort("region").to_pydict())
# {'region': ['n', 's'], 'price_sum': [30.0, 30.0], 'qty_sum': [3, 3],
#  'max_price': [20.0, 30.0], 'max_qty': [2, 3]}

print(sales.select(bt.numeric().mean()).to_pydict())
# {'price': [20.0], 'qty': [2.0]}
```

`alias(...)` names one column, so an aliased aggregate over a selector that matched several columns raises a `PlanError` when the query is written, and so does a keyword such as `agg(total=bt.numeric().sum())`. A window works the same way: `bt.numeric().sum().over(partition_by=["region"]).name.suffix("_region")` adds one windowed column per numeric column.

### Dtypes are the stored types

Batcher widens narrow types once, when data enters the engine: every integer width becomes `int64`, `float16` and `float32` become `float64`, and `large_string` and a dictionary-encoded string become `string`. A dtype selector matches the stored type, and {py:obj}`bt.by_dtype <batcher.by_dtype>` widens the type you ask for the same way, so {py:obj}`bt.by_dtype(pa.int32()) <batcher.by_dtype>` selects every `int64` column, including one that was `int64` in the source:

```python
import pyarrow as pa

narrow = bt.from_arrow(pa.table({"small": pa.array([1], pa.int32()), "big": pa.array([2], pa.int64())}))
print(narrow.select(bt.by_dtype(pa.int32())).columns)
# ['small', 'big']
```

The same holds for {py:obj}`bt.col(pa.int32()) <batcher.col>`, and it is why {py:obj}`bt.string() <batcher.string>` also selects a column that was dictionary-encoded, where Polars keeps its categoricals apart.

### Where selectors are accepted

Beyond the projections, `drop_nulls(subset=...)`, `drop_nans(subset=...)`, `distinct(subset=...)`, `unpivot(on=..., index=...)`, and positional `group_by(...)` keys take a selector, bare or inside a list, and resolve it to the columns it matches. A selector that matches nothing contributes no columns anywhere: `drop` and `drop_nulls` leave the data unchanged, and `select` or `with_columns` raise a `PlanError` naming the selector only when nothing at all is left to project.

```python
print(ds.drop(bt.temporal()).columns)  # nothing matched, nothing dropped
# ['name', 'price', 'qty']
```

A selector is refused with a `PlanError` in a filter predicate, as a join key, and in the other verbs that take column names. A join key is refused because the two sides would each expand it on their own. Some Polars selector constructors are not provided. There is no positional selector such as `by_index`, `first`, or `last`, and {py:obj}`bt.first <batcher.first>` and {py:obj}`bt.last <batcher.last>` are aggregates rather than selectors. There is also no finer dtype selector such as `date`, `datetime`, `duration`, `decimal`, `categorical`, `binary`, or `signed_integer`. Use {py:obj}`bt.by_dtype(...) <batcher.by_dtype>` with the Arrow type instead.

## Casting inside a projection

Casting is an expression method taking a type name, so it works inside either verb. {doc}`The type system </user-guide/transform/columns/type-system>` lists the names and what a cast can and cannot do.

```python
print(ds.with_columns(qty=bt.col("qty").cast("float64")).to_pydict())
# {'name': ['alice', 'bob', 'carol'], 'price': [10.0, 20.0, 30.0], 'qty': [1.0, 2.0, 3.0]}
```

## Conforming to a schema

{py:meth}`match_to_schema <batcher.Dataset.match_to_schema>` holds a dataset to a declared set of columns. The result has exactly the schema's columns, in its order, and a column whose type differs raises rather than being cast, so bad input stops the pipeline instead of flowing through it. `missing_columns="insert"` adds an absent column as nulls, and `extra_columns="ignore"` drops a column the schema does not name.

```python
raw = bt.from_pydict({"amount": [10, 20], "id": [1, 2], "debug": ["x", "y"]})
contract = {"id": "int64", "amount": "int64", "currency": "string"}
print(raw.match_to_schema(contract, missing_columns="insert", extra_columns="ignore").to_pydict())
# {'id': [1, 2], 'amount': [10, 20], 'currency': [None, None]}
```

The check reads only the schema, so it fails before any data is read.

## Reusing your own transformations

`pipe` applies a function to the dataset and returns its result, so a step you wrote yourself reads in the order it runs instead of inside-out. It adds no plan node and stays lazy when your function does.

```python
def with_total(frame, tax=0.0):
    return frame.with_columns(total=bt.col("price") * bt.col("qty") * (1 + tax))


print(ds.pipe(with_total, tax=0.5).filter(bt.col("total") > 20).to_pydict()["total"])
# [60.0, 135.0]
```

Without `pipe` the same pipeline reads backwards. `with_total(ds).filter(...)` puts the first step in the middle. Reach for `pipe` whenever a chain grows a step that has no built-in method.

Expressions have the same method. {py:meth}`Expr.pipe <batcher.Expr.pipe>` hands the expression to your function, so a reusable column builder chains the same way:

```python
def discounted(price, rate):
    return price * (1 - rate)


print(ds.select(net=bt.col("price").pipe(discounted, 0.1)).to_pydict()["net"])
```

## Flattening nested data

Semistructured data arrives with lists and structs inside columns. Two relational transforms flatten them, and they compose to unnest arbitrarily deep shapes.

`explode` turns a `list` column into one row per element, repeating the other columns, the same as SQL `UNNEST`. Empty and null lists drop out.

```python
nested = bt.from_pydict({"id": [1, 2], "tags": [["a", "b"], ["c"]]})
print(nested.explode("tags").to_pydict())
# {'id': [1, 1, 2], 'tags': ['a', 'b', 'c']}
```

`unnest` promotes a `struct` column's fields to top-level columns, replacing the struct in place.

```python
import pyarrow as pa

people = bt.from_arrow(
    pa.table({"person": pa.array([{"name": "Ann", "age": 30}, {"name": "Bo", "age": 25}])})
)
print(people.unnest("person").to_pydict())
# {'age': [30, 25], 'name': ['Ann', 'Bo']}
```

To reach a single field without flattening the whole struct, use the {py:class}`.struct <batcher.plan.expr_ir.namespaces.collections._StructNamespace>` and `.json` {doc}`accessors </user-guide/transform/columns/expression-accessors>` in a `select`. {py:meth}`.struct.field(name) <batcher.plan.expr_ir.namespaces.collections._StructNamespace.field>` projects one struct field, and {py:meth}`.json.extract_int(path) <batcher.plan.expr_ir.namespaces.collections._JsonNamespace.extract_int>` and its typed siblings read a value from a JSON-text column by JSONPath without a decode step. Explode a list of structs first, then `unnest`, to flatten a nested array of records into a flat table.

## See also

- {doc}`Filtering </user-guide/transform/rows/filtering>`: keeping the rows a predicate accepts.
- {doc}`Pivoting and reshaping </user-guide/analyze/pivoting>`: long to wide and back.
- {doc}`Aggregations </user-guide/analyze/aggregations>`: grouped and global summaries.
- {doc}`Dataset API </api/relational/dataset>`: the full method reference for every transformation.
- {doc}`/cookbook/dataset/verbs/reshaping`: pivot, unpivot, explode, unnest, and set operations, as a script.
