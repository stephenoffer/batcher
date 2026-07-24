# Expression accessors

An accessor namespace groups the methods that only make sense for one kind of column.
`.str` holds the string methods, `.dt` the date and time ones, and `.list`, `.struct`,
and `.json` the nested ones. They keep `Expr` itself small: a hundred string functions
live behind `.str` rather than on every expression.

This page covers each namespace in turn. The core expression language they hang off is
in {doc}`expressions`. Every example runs against the engine, and blocks share one
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

The `.str` namespace covers casing and trimming, search, slicing, padding,
encoding. Search methods such as `contains`, `starts_with`, and `like` are
case-sensitive; use `ilike` for case-insensitive matching.

```python
out = ds.select(
    upper=bt.col("name").str.upper(),
    length=bt.col("name").str.len(),
    has_a=bt.col("name").str.ilike("%a%"),
    first_two=bt.col("name").str.left(2),
)
print(out.to_pydict())
# {'upper': ['ANN', 'BOB', 'CARL'], 'length': [3, 3, 4], 'has_a': [True, False, True], 'first_two': ['An', 'bo', 'CA']}
```

More predicates and slicers round out the namespace: `ends_with` mirrors
`starts_with` for a literal suffix, `split_part(delimiter, n)` returns the `n`-th
1-based field of a split, `substring_index(delimiter, count)` keeps everything up
to the `count`-th delimiter, and `normalize_whitespace` collapses each run of
whitespace to one space and trims the ends. `zfill(width)` zero-pads fixed-width
codes, and `contains_any([...])` tests a row against several literal substrings at
once (an OR of `contains`).

```python
codes = bt.from_pydict({"id": ["7", "42"], "tag": ["cat-a", "dog-b"]})
out = codes.select(
    padded=bt.col("id").str.zfill(4),
    flagged=bt.col("tag").str.contains_any(["cat", "fish"]),
)
print(out.to_pydict())
# {'padded': ['0007', '0042'], 'flagged': [True, False]}
```

```python
paths = bt.from_pydict({"path": ["etc/app/conf", "usr/local/bin", "  a   b  "]})
out = paths.select(
    is_conf=bt.col("path").str.ends_with("conf"),
    second=bt.col("path").str.split_part("/", 2),
    head=bt.col("path").str.substring_index("/", 2),
    tidy=bt.col("path").str.normalize_whitespace(),
)
print(out.to_pydict())
# {'is_conf': [True, False, False], 'second': ['app', 'local', ''], 'head': ['etc/app', 'usr/local', '  a   b  '], 'tidy': ['etc/app/conf', 'usr/local/bin', 'a b']}
```

`ascii` returns the codepoint of the first character. `bit_length` and
`octet_length` measure the encoded size in bits and UTF-8 bytes, not characters.
`levenshtein(target)` gives the edit distance to a constant string and `soundex` its
phonetic key; both earn their keep in fuzzy matching and deduplication.

```python
words = bt.from_pydict({"w": ["Robert", "Rupert", "café"]})
out = words.select(
    code=bt.col("w").str.ascii(),
    bytes=bt.col("w").str.octet_length(),
    dist=bt.col("w").str.levenshtein("Ruperts"),
    phonetic=bt.col("w").str.soundex(),
)
print(out.to_pydict())
# {'code': [82, 82, 99], 'bytes': [6, 6, 5], 'dist': [3, 1, 7], 'phonetic': ['R163', 'R163', 'C100']}
```

Other `.str` methods include `lower`, `trim`, `lstrip`, `rstrip`, `reverse`,
`substr`, `right`, `repeat`, `lpad`, `rpad`, `position`, `split`, `replace`,
`initcap`, `hex`, `base64`, `from_base64`, `unhex`, and `translate`.

### Recasing identifiers

Column names, event names, and enum values arrive from upstream systems in whatever
convention that system used. `to_case(style)` normalizes them. One splitter finds the
words, so every style agrees on where the boundaries are: separators split, a
lower-to-upper transition splits, and a run of capitals splits before its last capital,
which keeps an acronym whole. Digits stay with the word they touch, so `sha256` survives.

```python
events = bt.from_pydict({"name": ["parseHTTPResponse", "user signed-up", "ORDER_PLACED"]})
out = events.select(
    snake=bt.col("name").str.to_case("snake"),
    camel=bt.col("name").str.to_case("camel"),
    title=bt.col("name").str.to_case("title"),
)
print(out.to_pydict())
# {'snake': ['parse_http_response', 'user_signed_up', 'order_placed'], 'camel': ['parseHttpResponse', 'userSignedUp', 'orderPlaced'], 'title': ['Parse Http Response', 'User Signed Up', 'Order Placed']}
```

The styles are `snake`, `upper_snake`, `camel`, `pascal`, `kebab`, `upper_kebab`,
`title`, `sentence`, `dot`, and `train`.

:::{note}
Recasing is idempotent in every style that joins with a separator. `camel` and `pascal`
join with nothing, so an input with consecutive single-letter words can't survive a round
trip: `a_b_c` becomes `aBC`, which reads back as two words. Prefer a separator style when
the result will be parsed again.
:::

### Regex

Alongside the single-match `regexp_matches`, `regexp_replace`, and `regexp_extract`,
three methods work over *every* match in a string: `regexp_count` tallies the
matches, `regexp_extract_all` gathers them into a list, and
`regexp_replace_all(pattern, replacement)` substitutes them all.

```python
codes = bt.from_pydict({"s": ["a1b2c3", "xyz", "p4q5"]})
out = codes.select(
    digits=bt.col("s").str.regexp_count("[0-9]"),
    found=bt.col("s").str.regexp_extract_all("[0-9]"),
    masked=bt.col("s").str.regexp_replace_all("[0-9]", "#"),
)
print(out.to_pydict())
# {'digits': [3, 0, 2], 'found': [['1', '2', '3'], [], ['4', '5']], 'masked': ['a#b#c#', 'xyz', 'p#q#']}
```

### Building strings from several columns

Two top-level helpers assemble one string from many expressions.
{py:obj}`bt.format_string(template, *exprs) <batcher.format_string>` interpolates
values into a template at each `{}` placeholder (Polars' `format`), while
{py:obj}`bt.concat_ws(separator, *exprs) <batcher.concat_ws>` joins values with a
separator between them (DuckDB/Spark `concat_ws`).

```python
out = ds.select(
    label=bt.format_string("{} x{}", bt.col("name"), bt.col("qty")),
    key=bt.concat_ws("-", bt.col("name"), bt.col("qty").cast("utf8")),
)
print(out.to_pydict())
# {'label': ['Ann x1', 'bob x2', 'CARL x3'], 'key': ['Ann-1', 'bob-2', 'CARL-3']}
```

### Parsing text into dates

Parsing string columns into temporal types also lives on `.str`:
`to_date(format)` yields a `Date` and `to_datetime(format)` a `Timestamp`, each
reading a chrono/strftime pattern. Once parsed, reach for the `.dt` accessor below.

```python
day_strs = bt.from_pydict({"d": ["2024-01-15", "2024-06-01"]})
print(day_strs.select(day=bt.col("d").str.to_date("%Y-%m-%d")).to_pydict())
# {'day': [datetime.date(2024, 1, 15), datetime.date(2024, 6, 1)]}

stamp_strs = bt.from_pydict({"t": ["2024-01-15 09:30", "2024-06-01 18:00"]})
print(stamp_strs.select(stamp=bt.col("t").str.to_datetime("%Y-%m-%d %H:%M")).to_pydict())
# {'stamp': [datetime.datetime(2024, 1, 15, 9, 30), datetime.datetime(2024, 6, 1, 18, 0)]}
```

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
week-numbering year, and `is_leap_year` / `days_in_month` answer calendar
questions per row. `epoch` returns whole seconds since 1970; `epoch_ms()`,
`epoch_us()`, and `epoch_ns()` give the same instant at millisecond, microsecond,
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

`strftime(format)` renders a timestamp as text with a chrono/strftime pattern;
`offset_by(by)` shifts it by a Polars-style duration string (`"1mo"`, `"3d"`,
`"-1h"`), preserving the type; and `convert_timezone(from_tz, to_tz)` re-reads each
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

`.struct.field(name)` pulls a field out of a struct column.

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

## JSON accessor: .json

`.json.extract_string(path)` reads a string value from a JSON text column using a
JSONPath expression.

```python
docs = bt.from_pydict({"doc": ['{"a": {"b": "hi"}}', '{"a": {"b": "bye"}}']})
out = docs.select(value=bt.col("doc").json.extract_string("$.a.b"))
print(out.to_pydict())
# {'value': ['hi', 'bye']}
```

The typed variants read a JSON value directly as a scalar instead of text.
`extract_bool`, `extract_int`, and `extract_float` return `Boolean`, `Int64`, and
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

Extraction assumes you already know the document. When you don't, five methods answer
questions about its structure instead. `array_length(path)` counts elements without
parsing any of them, `keys(path)` lists an object's keys in source order, `type_of(path)`
names the JSON type so you can route a field whose type varies row to row, and
`exists(path)` reports presence.

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
{doc}`explode <transformations>` and the whole `.list` namespace.

```python
orders = bt.from_pydict({"id": [1, 2], "j": ['{"items": ["pen", "ink"]}', '{"items": ["pad"]}']})
out = orders.with_columns(item=bt.col("j").json.values("$.items")).explode("item")
print(out.select("id", "item").to_pydict())
# {'id': [1, 1, 2], 'item': ['pen', 'ink', 'pad']}
```

## See also

:::{seealso}
- {doc}`expressions`: the core expression language these namespaces extend.
- {doc}`expression-recipes`: the same methods assembled into feature and text pipelines.
- {doc}`../api/expressions`: the exhaustive reference for every accessor method.
- {doc}`type-system`: the Arrow types each accessor requires.
:::
