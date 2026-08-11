# Expression accessors

An accessor namespace groups the methods that only make sense for one kind of column.
{py:class}`.str <batcher.plan.expr_ir.namespaces.strings._StrNamespace>` holds the string methods, {py:class}`.dt <batcher.plan.expr_ir.namespaces.temporal._DtNamespace>` the date and time ones, and {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>`, {py:class}`.struct <batcher.plan.expr_ir.namespaces.collections._StructNamespace>`,
and {py:class}`.json <batcher.plan.expr_ir.namespaces.collections._JsonNamespace>` the nested ones. They keep {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` itself small: a hundred string functions
live behind `.str` rather than on every expression.

This page covers each namespace in turn. The core expression language they hang off is
in {doc}`/user-guide/transform/columns/expressions`. Every example runs against the engine, and blocks share one
namespace and execute in order.

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "name": ["Ann", "bob", "CARL"],
        "price": [10.0, 20.0, 30.0],
        "qty": [1, 2, 3],
    }
)
```

## String accessor: .str

The `.str` namespace is the largest of them by a wide margin -- casing, trimming, search,
slicing, padding, regular expressions, encodings and the document-quality filters -- so it
has a page of its own: {doc}`/user-guide/transform/columns/string-accessor`.

## Datetime accessor: .dt

The `.dt` namespace extracts calendar parts from timestamp columns.

```python
import datetime

events = bt.from_pydict(
    {"ts": [datetime.datetime(2024, 1, 15, 9, 30), datetime.datetime(2024, 6, 1, 18, 0)]}
)
out = events.select(
    year=bt.col("ts").dt.year(),
    month=bt.col("ts").dt.month(),
    day_name=bt.col("ts").dt.dayname(),
)
print(out.to_pydict())
# {'year': [2024, 2024], 'month': [1, 6], 'day_name': ['Monday', 'Saturday']}
```

Also available: `day`, `hour`, `minute`, `second`, `quarter`, `week`,
`dayofweek`, `dayofyear`, `epoch`, `monthname`, `isodow`, `century`, `decade`,
`millennium`, `last_day`, and `truncate(unit)`. `iso_year` gives the ISO 8601
week-numbering year, and {py:meth}`is_leap_year <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.is_leap_year>` / {py:meth}`days_in_month <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.days_in_month>` answer calendar
questions per row. `epoch` returns whole seconds since 1970; `epoch_ms()`,
{py:meth}`epoch_us() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.epoch_us>`, and {py:meth}`epoch_ns() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.epoch_ns>` give the same instant at millisecond, microsecond,
and nanosecond resolution.

```python
out = events.select(
    iso=bt.col("ts").dt.iso_year(),
    leap=bt.col("ts").dt.is_leap_year(),
    month_len=bt.col("ts").dt.days_in_month(),
)
print(out.to_pydict())
# {'iso': [2024, 2024], 'leap': [True, True], 'month_len': [31, 30]}
```

{py:meth}`strftime(format) <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.strftime>` renders a timestamp as text with a chrono/strftime pattern;
{py:meth}`offset_by(by) <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.offset_by>` shifts it by a Polars-style duration string (`"1mo"`, `"3d"`,
`"-1h"`), preserving the type; and {py:meth}`convert_timezone(from_tz, to_tz) <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.convert_timezone>` re-reads each
naive wall-clock from one zone in another, DST-aware.

```python
out = events.select(
    text=bt.col("ts").dt.strftime("%Y/%m/%d"),
    next_month=bt.col("ts").dt.offset_by("1mo"),
    in_ny=bt.col("ts").dt.convert_timezone("UTC", "America/New_York"),
)
print(out.select(text=bt.col("text"), next=bt.col("next_month").dt.month(), ny_hour=bt.col("in_ny").dt.hour()).to_pydict())
# {'text': ['2024/01/15', '2024/06/01'], 'next': [2, 7], 'ny_hour': [4, 14]}
```

### Top-level date/time functions

Some date arithmetic reads better as a function than as an accessor call.
{py:obj}`bt.date_part(part, expr) <batcher.date_part>` extracts a named field (the
SQL spelling of `.dt.<part>()`); {py:obj}`bt.date_add(expr, days) <batcher.date_add>`
and {py:obj}`bt.date_sub(expr, days) <batcher.date_sub>` shift by a whole number of
days.

```python
out = events.select(
    part=bt.date_part("month", bt.col("ts")),
    later=bt.date_add(bt.col("ts"), 7),
    earlier=bt.date_sub(bt.col("ts"), 7),
)
print(out.select(part=bt.col("part"), later=bt.col("later").dt.day(), earlier=bt.col("earlier").dt.day()).to_pydict())
# {'part': [1, 6], 'later': [22, 8], 'earlier': [8, 25]}
```

### Building dates and timestamps

The `.dt` accessors take a date apart. Four functions put one together.
{py:obj}`bt.make_date(year, month, day) <batcher.make_date>` and
{py:obj}`bt.make_timestamp(...) <batcher.make_timestamp>` assemble one from integer
columns. A date that doesn't exist, such as February 29 in a non-leap year, becomes null
rather than an error, so one bad row in a scan of dirty upstream integers doesn't abort
the query.

```python
parts = bt.from_pydict({"y": [2024, 2023], "m": [2, 2], "d": [29, 29]})
out = parts.select(day=bt.make_date(bt.col("y"), bt.col("m"), bt.col("d")))
print(out.to_pydict())
# {'day': [datetime.date(2024, 2, 29), None]}
```

{py:obj}`bt.from_epoch(expr, unit) <batcher.from_epoch>` reads an integer column of epoch
counts as a timestamp, and {py:obj}`bt.from_unix_date(expr) <batcher.from_unix_date>` reads
a column of days since 1970-01-01 as a date.

```python
logs = bt.from_pydict({"ts": [1700000000], "ms": [1700000000123]})
out = logs.select(
    from_s=bt.from_epoch(bt.col("ts"), "s"),
    from_ms=bt.from_epoch(bt.col("ms"), "ms"),
)
print(out.to_pydict())
# {'from_s': [datetime.datetime(2023, 11, 14, 22, 13, 20)], 'from_ms': [datetime.datetime(2023, 11, 14, 22, 13, 20, 123000)]}
```

:::{important}
State the unit. An `Int64` column of epoch counts looks identical whether it holds seconds
or nanoseconds, so `cast("timestamp")` has to assume one, and it assumes microseconds. A
column of epoch seconds cast that way lands in January 1970 with no error at all. The
units are `"s"`, `"ms"`, `"us"`, and `"ns"`.
:::

{py:obj}`bt.current_date() <batcher.current_date>` and
{py:obj}`bt.current_timestamp() <batcher.current_timestamp>` capture "now" as a
literal, bound once at plan-build time (so every row sees the same value). Because
the value depends on the wall clock, compare it to Python's clock rather than a
fixed constant.

```python
today = bt.from_pydict({"z": [1]}).select(d=bt.current_date()).to_pydict()["d"][0]
now = bt.from_pydict({"z": [1]}).select(t=bt.current_timestamp()).to_pydict()["t"][0]
print(today == datetime.date.today(), isinstance(now, datetime.datetime))
# True True
```

## List accessor: .list

The `.list` namespace reduces and reshapes list-typed columns.

```python
lists = bt.from_pydict({"tags": [["x", "y"], ["z"], ["a", "b", "c"]]})
out = lists.select(
    n=bt.col("tags").list.len(),
    joined=bt.col("tags").list.join("-"),
    first=bt.col("tags").list.first(),
    last=bt.col("tags").list.last(),
)
print(out.to_pydict())
# {'n': [2, 1, 3], 'joined': ['x-y', 'z', 'a-b-c'], 'first': ['x', 'z', 'a'], 'last': ['y', 'z', 'c']}
```

Numeric lists support reductions: `sum`, `min`, `max`, `mean`, `median`, `std`,
`var`, `product`, `n_unique`, `arg_min`, `arg_max`. Structural methods include
`sort`, `reverse`, `unique`, `slice`, `head(n)` (the leading `n` elements), and
`contains`. Element access is `get(i)` (negative indexes from the end), with
`first()`/`last()` as shorthands.

## Struct accessor: .struct

{py:meth}`.struct.field(name) <batcher.plan.expr_ir.namespaces.collections._StructNamespace.field>` pulls a field out of a struct column.

```python
import pyarrow as pa

points = bt.from_arrow(pa.table({"p": pa.array([{"x": 1, "y": 2}, {"x": 3, "y": 4}])}))
out = points.select(x=bt.col("p").struct.field("x"), y=bt.col("p").struct.field("y"))
print(out.to_pydict())
# {'x': [1, 3], 'y': [2, 4]}
```

Going the other way, {py:obj}`bt.named_struct(name, value, ...) <batcher.named_struct>`
packs several columns into one struct from alternating name/value arguments. Use it
to nest a group of fields before a write, or before a `.struct.field` lookup.

```python
out = ds.select(row=bt.named_struct("who", bt.col("name"), "n", bt.col("qty")))
print(out.to_pydict())
# {'row': [{'who': 'Ann', 'n': 1}, {'who': 'bob', 'n': 2}, {'who': 'CARL', 'n': 3}]}
```

## Map accessor: .map

A `Map` column holds a different set of keys per row, which is what separates it from a
struct: a struct's fields are fixed by its type, so `.struct.field("x")` is a name
resolution and asking for a field the type lacks is an error. A map's keys are data, so
`.map.get("x")` is a per-row lookup and a missing key is an ordinary null.

```python
import pyarrow as pa

tags = pa.array(
    [[("env", "prod"), ("tier", "web")], [("env", "dev")], []],
    type=pa.map_(pa.string(), pa.string()),
)
svc = bt.from_arrow(pa.table({"tags": tags}))
out = svc.select(
    env=bt.col("tags").map.get("env"),
    tier=bt.col("tags").map.get("tier"),
    n=bt.col("tags").map.len(),
    has_env=bt.col("tags").map.contains("env"),
)
print(out.to_pydict())
# {'env': ['prod', 'dev', None], 'tier': ['web', None, None], 'n': [2, 1, 0], 'has_env': [True, True, False]}
```

{py:meth}`.map.keys() <batcher.plan.expr_ir.namespaces.collections._MapNamespace.keys>`
and {py:meth}`.map.values() <batcher.plan.expr_ir.namespaces.collections._MapNamespace.values>`
each return a list column, positionally aligned with each other.

```python
out = svc.select(k=bt.col("tags").map.keys(), v=bt.col("tags").map.values())
print(out.to_pydict())
# {'k': [['env', 'tier'], ['env'], []], 'v': [['prod', 'web'], ['dev'], []]}
```

When a key has to travel with its value, reach for
{py:meth}`.map.entries() <batcher.plan.expr_ir.namespaces.collections._MapNamespace.entries>`
instead of zipping those two lists. It returns one `{key, value}` struct per entry, so the
pairing is structural and survives anything that reorders the list later.

```python
out = svc.select(e=bt.col("tags").map.entries())
print(out.to_pydict())
# {'e': [[{'key': 'env', 'value': 'prod'}, {'key': 'tier', 'value': 'web'}], [{'key': 'env', 'value': 'dev'}], []]}
```

That shape is what turns a map column into rows. Explode the entry list and each map row
becomes one row per key, which is the usual way to group or join on keys that vary by row.

```python
long = (
    svc.select(e=bt.col("tags").map.entries())
    .explode("e")
    .select(key=bt.col("e").struct.get("key"), value=bt.col("e").struct.get("value"))
)
print(long.to_pydict())
# {'key': ['env', 'tier', 'env'], 'value': ['prod', 'web', 'dev']}
```

A null map row stays null through all of these, and an empty map returns an empty list
rather than a null. The two are distinct, and a filter on `.map.len() == 0` selects only
the empty ones.

## JSON accessor: .json

{py:meth}`.json.extract_string(path) <batcher.plan.expr_ir.namespaces.collections._JsonNamespace.extract_string>` reads a string value from a JSON text column using a
JSONPath expression.

```python
docs = bt.from_pydict({"doc": ['{"a": {"b": "hi"}}', '{"a": {"b": "bye"}}']})
out = docs.select(value=bt.col("doc").json.extract_string("$.a.b"))
print(out.to_pydict())
# {'value': ['hi', 'bye']}
```

The typed variants read a JSON value directly as a scalar instead of text.
{py:meth}`extract_bool <batcher.plan.expr_ir.namespaces.collections._JsonNamespace.extract_bool>`, {py:meth}`extract_int <batcher.plan.expr_ir.namespaces.collections._JsonNamespace.extract_int>`, and {py:meth}`extract_float <batcher.plan.expr_ir.namespaces.collections._JsonNamespace.extract_float>` return `Boolean`, `Int64`, and
`Float64` columns, yielding null when the path is absent or the value has the wrong
type. No separate cast step.

```python
records = bt.from_pydict(
    {"r": ['{"ok": true, "n": 3, "score": 4.5}', '{"ok": false, "n": 7, "score": 9.0}']}
)
out = records.select(
    ok=bt.col("r").json.extract_bool("$.ok"),
    n=bt.col("r").json.extract_int("$.n"),
    score=bt.col("r").json.extract_float("$.score"),
)
print(out.to_pydict())
# {'ok': [True, False], 'n': [3, 7], 'score': [4.5, 9.0]}
```

### Inspecting a document's shape

Extraction assumes you already know the document. When you don't, these methods answer questions about its structure instead. `structure()` renders the shape with each leaf replaced by its type name, which is the thing to group by when finding out what shapes a column holds. `value(path)` returns a scalar's JSON token and null for a container (DuckDB draws that line between `json_value` and `json_extract_string`); `contains(value)` tests membership; `pretty()` re-renders for a human. `array_length(path)` counts elements without
parsing any of them, `keys(path)` lists an object's keys in source order, `type_of(path)`
names the JSON type so you can route a field whose type varies row to row, and
{py:meth}`exists(path) <batcher.plan.expr_ir.namespaces.collections._JsonNamespace.exists>` reports presence.

```python
payloads = bt.from_pydict(
    {"p": ['{"id": 1, "tags": ["a", "b"], "note": null}', '{"id": 2, "tags": []}']}
)
out = payloads.select(
    fields=bt.col("p").json.keys(),
    n_tags=bt.col("p").json.array_length("$.tags"),
    kind=bt.col("p").json.type_of("$.id"),
    has_note=bt.col("p").json.exists("$.note"),
)
print(out.to_pydict())
# {'fields': [['id', 'tags', 'note'], ['id', 'tags']], 'n_tags': [2, 0], 'kind': ['number', 'number'], 'has_note': [True, False]}
```

`exists` is the only one of these that separates an absent key from a key whose value is
JSON `null`. The `extract_*` methods can't: both come back as SQL NULL, and in an
ingestion pipeline those mean different things.

`values(path)` turns a JSON array into a list column, which hands the array to
{doc}`explode </user-guide/transform/rows/transformations>` and the whole `.list` namespace.

```python
orders = bt.from_pydict({"id": [1, 2], "j": ['{"items": ["pen", "ink"]}', '{"items": ["pad"]}']})
out = orders.with_columns(item=bt.col("j").json.values("$.items")).explode("item")
print(out.select("id", "item").to_pydict())
# {'id': [1, 1, 2], 'item': ['pen', 'ink', 'pad']}
```

## See also

- {doc}`/user-guide/transform/columns/expressions`: the core expression language these namespaces extend.
- {doc}`/user-guide/transform/columns/expression-recipes`: the same methods assembled into feature and text pipelines.
- {doc}`/api/relational/expressions`: the exhaustive reference for every accessor method.
- {doc}`/user-guide/transform/columns/type-system`: the Arrow types each accessor requires.
- {doc}`/cookbook/expressions/index`: 39 runnable recipes across the accessor namespaces.
- {doc}`/user-guide/transform/columns/sequence-accessor`: the {py:class}`.seq <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace>` accessor, for DNA, RNA, protein, and FASTQ-quality columns.
