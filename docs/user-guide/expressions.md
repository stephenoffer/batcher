# Expressions

Column work in Batcher is expressed with the `Expr` API, never with Python loops.
An expression is a small, typed description of a computation. It lowers to the
Rust data plane and runs over Arrow batches, so the same code is fast on three
rows or three billion.

Every example on this page runs against the engine. Blocks share one namespace
and execute in order.

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

## Columns and literals

{py:obj}`bt.col(name) <batcher.col>` refers to an input column. {py:obj}`bt.lit(value) <batcher.lit>` is a constant. Both are
expressions, so they compose with operators and methods.

```python
out = ds.select(
    "name",
    revenue=bt.col("price") * bt.col("qty"),
    discounted=bt.col("price") * bt.lit(0.9),
)
print(out.to_pydict())
# {'name': ['Ann', 'bob', 'CARL'], 'revenue': [10.0, 40.0, 90.0], 'discounted': [9.0, 18.0, 27.0]}
```

## Arithmetic, comparison, and boolean operators

Arithmetic uses `+ - * / %` and `**` (power). Reflected forms work, so a literal
may lead: `2 * bt.col("x")`. Comparison uses `== != > >= < <=`. Boolean logic uses
`&` (and), `|` (or), and `~` (not); parenthesize each side because `&` binds
tighter than comparison.

```python
out = ds.select(
    "name",
    cheap=(bt.col("price") < 25),
    cheap_and_small=((bt.col("price") < 25) & (bt.col("qty") <= 1)),
    not_cheap=~(bt.col("price") < 25),
)
print(out.to_pydict())
# {'name': ['Ann', 'bob', 'CARL'], 'cheap': [True, True, False], 'cheap_and_small': [True, False, False], 'not_cheap': [False, False, True]}
```

## Conditionals: when / then / otherwise

{py:obj}`bt.when(cond).then(value) <batcher.when>` builds a SQL `CASE`. Chain more `.when(...).then(...)`
clauses and close with `.otherwise(default)`.

```python
out = ds.select(
    "name",
    tier=bt.when(bt.col("price") >= 30)
    .then(bt.lit("high"))
    .when(bt.col("price") >= 15)
    .then(bt.lit("mid"))
    .otherwise(bt.lit("low")),
)
print(out.to_pydict())
# {'name': ['Ann', 'bob', 'CARL'], 'tier': ['low', 'mid', 'high']}
```

With exactly two branches, {py:obj}`bt.iff(cond, if_true, if_false) <batcher.iff>`
is the terse form of a single `when/then/otherwise`. It is the SQL `IF`/`IFF`.

```python
out = ds.select(
    "name",
    size=bt.iff(bt.col("price") >= 20, bt.lit("big"), bt.lit("small")),
)
print(out.to_pydict())
# {'name': ['Ann', 'bob', 'CARL'], 'size': ['small', 'big', 'big']}
```

## Null handling

{py:obj}`bt.coalesce <batcher.coalesce>` returns the first non-null argument. {py:obj}`bt.nullif(a, b) <batcher.nullif>` returns null
when `a == b`. {py:obj}`bt.greatest <batcher.greatest>` and {py:obj}`bt.least <batcher.least>` pick the extreme across columns. On a
single expression, `.fill_null(value)`, `.is_null()`, and `.is_not_null()` apply.

```python
nulls = bt.from_pydict({"a": [1, None, 3], "b": [9, 8, 7]})
out = nulls.select(
    first_present=bt.coalesce(bt.col("a"), bt.col("b")),
    filled=bt.col("a").fill_null(0),
    bigger=bt.greatest(bt.col("a").fill_null(0), bt.col("b")),
)
print(out.to_pydict())
# {'first_present': [1, 8, 3], 'filled': [1, 0, 3], 'bigger': [9, 8, 7]}
```

A floating-point `NaN` is distinct from null. {py:obj}`bt.nanvl(value, fallback) <batcher.nanvl>`
(Spark's `nanvl`) substitutes `fallback` only where `value` is `NaN`. Real numbers
are left alone, and so are nulls.

```python
import math

floats = bt.from_pydict({"v": [1.0, math.nan, 3.0]})
out = floats.select(clean=bt.nanvl(bt.col("v"), bt.lit(0.0)))
print(out.to_pydict())
# {'clean': [1.0, 0.0, 3.0]}
```

## Row-wise (horizontal) reductions

Aggregates fold a column *down* to one value; the `*_horizontal` functions fold
*across* columns within each row. `sum_horizontal`/`mean_horizontal` combine numeric
columns (nulls treated as 0 / skipped), and `min_horizontal`/`max_horizontal` are the
Polars-named row-wise `least`/`greatest`. `all_horizontal`/`any_horizontal` reduce
many boolean columns into one, which is how you combine validation flags.

```python
checks = bt.from_pydict({"a": [1, 2, 3], "b": [4, 6, 6], "c": [7, 8, 9]})
out = checks.select(
    total=bt.sum_horizontal(bt.col("a"), bt.col("b"), bt.col("c")),
    smallest=bt.min_horizontal(bt.col("a"), bt.col("b"), bt.col("c")),
    all_even=bt.all_horizontal(bt.col("a") % 2 == 0, bt.col("b") % 2 == 0),
)
print(out.to_pydict())
# {'total': [12, 16, 18], 'smallest': [1, 2, 3], 'all_even': [False, True, False]}
```

## Membership, ranges, and casts

```python
out = ds.select(
    "name",
    in_set=bt.col("qty").is_in([1, 3]),
    in_range=bt.col("price").between(15.0, 30.0),
    qty_f=bt.col("qty").cast("float64"),
)
print(out.to_pydict())
# {'name': ['Ann', 'bob', 'CARL'], 'in_set': [True, False, True], 'in_range': [False, True, True], 'qty_f': [1.0, 2.0, 3.0]}
```

`.cast` takes an Arrow type name as a string (for example `"int64"`, `"float64"`,
`"utf8"`).

## Math methods

Numeric expressions carry a full set of math methods, including `.abs()`,
`.round(digits)`, `.sqrt()`, `.pow(e)`, `.floor()`, `.ceil()`, `.ln()`,
`.log10()`, `.log2()`, `.exp()`, the trig family (`.sin()`, `.cos()`, `.tan()`,
`.asin()`, `.acos()`, `.atan()`, `.sinh()`, `.cosh()`, `.tanh()`, `.cot()`),
`.sign()`, `.trunc()`, `.cbrt()`, `.degrees()`, and `.radians()`.
{py:obj}`bt.atan2(y, x) <batcher.atan2>` is a top-level two-argument form.

```python
nums = bt.from_pydict({"x": [1.0, 4.0, 9.0]})
out = nums.select(
    root=bt.col("x").sqrt(),
    third=(bt.col("x") / 3).round(2),
    squared=bt.col("x").pow(2),
)
print(out.to_pydict())
# {'root': [1.0, 2.0, 3.0], 'third': [0.33, 1.33, 3.0], 'squared': [1.0, 16.0, 81.0]}
```

A few math functions take two columns. {py:obj}`bt.gcd <batcher.gcd>` and
{py:obj}`bt.lcm <batcher.lcm>` are integer number-theory helpers;
{py:obj}`bt.hypot(a, b) <batcher.hypot>` is the Euclidean norm `sqrt(a² + b²)`
(like `atan2`, a top-level two-argument form).

```python
pairs = bt.from_pydict({"a": [12.0, 15.0], "b": [18.0, 20.0], "x": [3.0, 5.0], "y": [4.0, 12.0]})
out = pairs.select(
    g=bt.gcd(bt.col("a"), bt.col("b")),
    l=bt.lcm(bt.col("a"), bt.col("b")),
    dist=bt.hypot(bt.col("x"), bt.col("y")),
)
print(out.to_pydict())
# {'g': [6.0, 5.0], 'l': [36.0, 60.0], 'dist': [5.0, 13.0]}
```

{py:obj}`bt.width_bucket(value, low, high, count) <batcher.width_bucket>` assigns each
value to one of `count` equal-width histogram buckets spanning `[low, high)`. The
result is `1..count`, with `0` for values below the range and `count + 1` above it.
Reach for it to bin a continuous column without a chain of `when`s.

```python
scores = bt.from_pydict({"score": [5.0, 55.0, 95.0, 120.0]})
out = scores.select(bucket=bt.width_bucket(bt.col("score"), bt.lit(0.0), bt.lit(100.0), 4))
print(out.to_pydict())
# {'bucket': [1.0, 3.0, 4.0, 5.0]}
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
# {'upper': ['ANN', 'BOB', 'CARL'], 'length': [3, 3, 4], 'has_a': [True, False, True], 'first_two': ['An', 'bo']}
```

More predicates and slicers round out the namespace: `ends_with` mirrors
`starts_with` for a literal suffix, `split_part(delimiter, n)` returns the `n`-th
1-based field of a split, `substring_index(delimiter, count)` keeps everything up
to the `count`-th delimiter, and `normalize_whitespace` collapses each run of
whitespace to one space and trims the ends.

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
questions per row.

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
`sort`, `reverse`, `unique`, `slice`, and `contains`. Element access is
`get(i)` (negative indexes from the end), with `first()`/`last()` as shorthands.

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

## Aggregate expressions

Aggregate methods such as `.sum()`, `.mean()`, `.min()`, `.max()`, `.median()`,
`.std()`, `.var()`, `.quantile(q)`, `.count()`, and `.n_unique()` are used inside
`group_by(...).agg(...)`. {py:obj}`bt.count() <batcher.count>` is the top-level `COUNT(*)`.

```python
out = ds.group_by().agg(
    total=bt.col("price").sum(),
    avg_qty=bt.col("qty").mean(),
    rows=bt.count(),
)
print(out.to_pydict())
# {'total': [60.0], 'avg_qty': [2.0], 'rows': [3]}
```

## Next steps

- [Expressions API](../api/expressions.md): every `Expr` method and accessor
  (`.str`/`.dt`/`.list`/`.struct`/`.json`/`.map`/`.image`/`.audio`/`.video`) in one
  exhaustive reference.
- [Aggregations](aggregations.md) and [Window functions](window-functions.md): where
  aggregate and windowed expressions are used.
- [SQL](sql.md): the same column language, spelled as SQL.
