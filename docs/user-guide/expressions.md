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
`count_horizontal` counts the non-null values in each row and `product_horizontal`
multiplies them (nulls treated as 1).

```python
checks = bt.from_pydict({"a": [1, 2, 3], "b": [4, 6, 6], "c": [7, 8, 9]})
out = checks.select(
    total=bt.sum_horizontal(bt.col("a"), bt.col("b"), bt.col("c")),
    smallest=bt.min_horizontal(bt.col("a"), bt.col("b"), bt.col("c")),
    filled=bt.count_horizontal(bt.col("a"), bt.col("b"), bt.col("c")),
    prod=bt.product_horizontal(bt.col("a"), bt.col("b"), bt.col("c")),
    all_even=bt.all_horizontal(bt.col("a") % 2 == 0, bt.col("b") % 2 == 0),
)
print(out.to_pydict())
# {'total': [12, 16, 18], 'smallest': [1, 2, 3], 'filled': [3, 3, 3], 'prod': [28, 96, 162], 'all_even': [False, True, False]}
```

When no named `*_horizontal` helper fits, `reduce_horizontal(fn, *exprs)` folds the
columns left-to-right with your own binary combiner, and `fold_horizontal(acc, fn,
*exprs)` does the same from an explicit seed. The combiner runs once at plan-build
time on `Expr` operands, never on a row, so the fold still lowers to pure Rust:

```python
cols = [bt.col("a"), bt.col("b"), bt.col("c")]
out = checks.select(
    manual_sum=bt.reduce_horizontal(lambda x, y: x + y, *cols),
    sum_sq=bt.fold_horizontal(bt.lit(0), lambda s, x: s + x * x, *cols),
)
print(out.to_pydict())
# {'manual_sum': [12, 16, 18], 'sum_sq': [66, 104, 126]}
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
{py:obj}`bt.hypot(a, b) <batcher.hypot>` is the Euclidean norm `sqrt(a^2 + b^2)`, a
top-level two-argument form as `atan2` is.

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

## Migrating from Polars or pandas

Coming from another DataFrame library, the operation you know by its Polars or pandas
name is usually available under that name too, delegating to Batcher's SQL-style
primary. On `.str`: `to_lowercase`, `to_uppercase`, `to_titlecase`, `pad_start`,
`pad_end`, `ljust`, `rjust`, `count_matches`, `extract`, `extract_all`, `replace_all`,
`len_chars`, `len_bytes`, `strip_chars`, `strip_chars_start`, `strip_chars_end`, `head`,
`tail`, and `slice`. On `.dt`: `weekday`, `ordinal_day`, `to_string`, `date`,
`month_start`, `month_end`, and the sub-second `millisecond` / `microsecond` /
`nanosecond`. On `.list`: `set_union`, `set_intersection`, `set_difference`. On an
expression: `arcsin`/`arccos`/`arctan`/`arcsinh`/`arccosh`/`arctanh`, `clip_min` /
`clip_max`, and `is_between`; plus top-level `bt.arctan2(y, x)`.

The pandas spellings are there too. On `.str`: `strip`, `startswith`, `endswith`,
`match`, `title`, and Python's `removeprefix` / `removesuffix`; on `.dt`: `day_name`,
`month_name`, `daysinmonth`, `weekofyear`, `normalize`, and `floor(unit)`. On the
`Dataset` itself: `fillna`, `dropna`, `isna`, `notna`, `astype`, `assign`, `groupby`,
`merge`, `sort_values`, `nlargest`, `nsmallest`, `round`, `abs`, `clip`, `shape`,
`size`, plus `nunique`, `select_dtypes`, `sample_frac`, and `drop_constant_columns`.

```python
migrate = bt.from_pydict({"name": ["  Ann  "], "code": ["7"]})
out = migrate.select(
    clean=bt.col("name").str.strip_chars().str.to_uppercase(),
    padded=bt.col("code").str.rjust(4, "0"),
)
print(out.to_pydict())
# {'clean': ['ANN'], 'padded': ['0007']}
```

An expression carries the pandas names as well, so a ported column computation runs
without a find-and-replace pass: `astype`, `isna`, `isnull`, `notna`, `notnull`,
`fillna`, `isin`, `nunique`, `rename`, `skew`, `kurt`, `prod`, `any`, `all`, `log`
(numpy's natural logarithm), and the cumulative `cumsum`, `cummax`, `cummin`,
`cumcount`. Each operator has its pandas method form too (`add`,
`sub`, `mul`, `truediv`, `div`, `floordiv`, `mod`, `eq`, `ne`, `lt`, `le`, `gt`, `ge`),
along with `and_`, `or_`, `not_`, and `xor` for the boolean operators that Python's
keywords can't express.

```python
sales = bt.from_pydict({"units": [2, None, 5], "price": [10, 20, 30]})
print(sales.select(
    revenue=bt.col("units").fillna(0).mul(bt.col("price")),
    missing=bt.col("units").isna(),
).to_pydict())
# {'revenue': [20, 0, 150], 'missing': [False, True, False]}
```

The typed accessors follow the same rule. `.str` answers to Python's own string
predicates `isdigit`, `isalpha`, `isalnum`, and `isspace`, and to Polars'
`strip_prefix` / `strip_suffix`. `.dt` takes the snake_case `day_of_week`,
`day_of_year`, and `week_of_year`. `.list` takes `lengths`, PySpark's `element_at`, and
numpy's `argmin` / `argmax`.

```python
records = bt.from_pydict({"code": ["123", "a1"], "tags": [[3, 1, 2], [5]]})
print(records.select(
    numeric=bt.col("code").str.isdigit(),
    n=bt.col("tags").list.lengths(),
    smallest=bt.col("tags").list.argmin(),
    second=bt.col("tags").list.element_at(1),
).to_pydict())
# {'numeric': [True, False], 'n': [3, 1], 'smallest': [1, 0], 'second': [1, None]}
```

A few ecosystem names are deliberately missing, because they don't mean the same thing
here. `str.find` and `str.index` are absent because `position` is 1-based and returns 0
when the substring is absent, where pandas returns a 0-based index and -1. `str.islower`
and `str.isupper` are absent because Batcher's `is_lower` / `is_upper` are true for a
string with no cased characters, where Python's are false. Use `str.slice` rather than a
`substring` alias, and `str.regexp_count` for pandas' regex `count`.

## Feature engineering for data science

The expression layer carries the transforms a model pipeline needs, so feature
engineering runs in the engine rather than in pandas. The scaling and encoding functions
each accept `partition_by=` to fit per group: `zscore`, `minmax_scale`, `maxabs_scale`,
`mean_center`, `label_encode`, and `hash_bucket` for a reproducible split key. Activations (`sigmoid`, `logit`, `relu`, `softplus`), share/ratio features
(`pct_of_total`, `cumulative_pct`, `normalize_l1`, `rank_pct`, `safe_divide`), and the
expanding statistics (`expanding_mean`, `expanding_var`, `expanding_std`) round it out.
Value predicates `is_positive`, `is_negative`, `is_zero`, `is_even`, `is_odd`, and
`is_outlier` read as filters.

```python
model = bt.from_pydict({"g": ["a", "a", "b", "b"], "v": [1.0, 3.0, 10.0, 20.0]})
out = model.select(
    z=bt.col("v").zscore(["g"]).round(4),
    scaled=bt.col("v").minmax_scale(["g"]),
    activated=bt.col("v").sigmoid().round(4),
)
print(out.to_pydict())
# {'z': [-0.7071, 0.7071, -0.7071, 0.7071], 'scaled': [0.0, 1.0, 0.0, 1.0], 'activated': [0.7311, 0.9526, 1.0, 1.0]}
```

Calendar features come off `.dt`: `is_weekend` / `is_weekday`, `is_month_start` /
`is_month_end`, `is_quarter_start` / `is_quarter_end`, `is_year_start` /
`is_year_end`, plus `quarter_start`, `year_start`, `days_in_year`, and
`week_of_month`, the period closes `quarter_end` and `year_end`, and the elapsed-time
features `seconds_between`, `minutes_between`, `hours_between`, `days_between`, and
`weeks_between`. Text features come off `.str`: `word_count`, `digit_count`,
`contains_all`, `count_char`,
`capitalize`, `remove_punctuation`, and the character-class checks `is_alpha`,
`is_numeric`, `is_alnum`, `is_space`, `is_upper`, `is_lower`.

```python
import datetime as dt

events = bt.from_pydict({"d": [dt.datetime(2024, 2, 3)], "note": ["Hi, there!"]})
out = events.select(
    weekend=bt.col("d").dt.is_weekend(),
    week=bt.col("d").dt.week_of_month(),
    words=bt.col("note").str.word_count(),
    clean=bt.col("note").str.remove_punctuation(),
)
print(out.to_pydict())
# {'weekend': [True], 'week': [1], 'words': [2], 'clean': ['Hi there']}
```

For column profiling, `bt.q1` / `bt.q3` / `bt.iqr` give the robust spread,
`bt.value_range` the full spread, `bt.null_rate` / `bt.non_null_rate` completeness, and
`bt.nunique_ratio` the cardinality ratio that separates identifiers from categoricals.

```python
prof = bt.from_pydict({"x": [1.0, None, 3.0, 4.0]})
out = prof.agg(
    spread=bt.iqr("x"),
    rng=bt.value_range("x"),
    missing=bt.null_rate("x"),
    card=bt.nunique_ratio("x"),
)
print(out.to_pydict())
# {'spread': [1.5], 'rng': [3.0], 'missing': [0.25], 'card': [0.75]}
```

## Curating an AI training corpus

Filtering a pretraining corpus is a per-row scan, so it runs in the engine. The `.str`
namespace carries the Gopher / C4-style quality heuristics: the character-class ratios
`alpha_ratio`, `digit_ratio`, `uppercase_ratio`, `lowercase_ratio`,
`punctuation_ratio`, `whitespace_ratio`, `non_ascii_ratio`, and `alnum_ratio`, plus the
shape statistics `line_count`, `mean_line_length`, `avg_word_length`, `sentence_count`,
`non_ascii_count`, `url_count`, and `email_count`. Thresholding a couple of these
removes most boilerplate, link dumps, and machine-generated text.

```python
corpus = bt.from_pydict(
    {"text": ["Real prose, with sentences and words.", "AAA 111 &&& ||| ###"]}
)
kept = corpus.filter(
    (bt.col("text").str.alpha_ratio() > 0.6)
    & (bt.col("text").str.avg_word_length().is_between(3, 10))
)
print(kept.to_pydict())
# {'text': ['Real prose, with sentences and words.']}
```

Document shape adds `paragraph_count`, `is_single_line`, `ends_with_punctuation`,
`has_repeated_punctuation`, `quote_count`, `paren_count`, `digit_to_word_ratio`, and the
code detectors `code_fence_count` and `looks_like_code`.
Further signals include `uppercase_word_count`, `long_word_count`,
`symbol_to_word_ratio`, `hashtag_count`, `mention_count`, `phone_count`, and
`has_phone`. Cleaning and PII scrubbing use `remove_urls`, `remove_emails`,
`remove_phones`, the shape-preserving `mask_emails` / `mask_urls`, `remove_non_ascii`,
`remove_digits`, and `remove_html_tags`; `truncate_chars` and `truncate_words` cap a row
to a budget without cutting mid-word. The detection predicates `has_url`, `has_email`,
`has_non_ascii`, `has_digits`, `has_html`, `is_ascii_only`, `is_blank`,
`starts_with_bullet`, and `looks_like_json` read as filters. For context windows,
`estimate_tokens` and `fits_token_budget` give a tokenizer-free size estimate.

```python
raw = bt.from_pydict({"text": ["Mail bob@x.com or see http://y.io for more"]})
print(raw.select(clean=bt.col("text").str.remove_emails().str.remove_urls()).to_pydict())
# {'clean': ['Mail  or see  for more']}
```

Counts and predicates round it out: `newline_count`, `tab_count`, `space_count`,
`word_char_ratio`, `avg_sentence_length`, `is_short` / `is_long`, `is_question`,
`is_exclamation`, `starts_with_capital`, `is_all_caps`, `has_currency`, and the
whole-string `is_url` / `is_email`. Extraction gives `extract_urls`, `extract_emails`,
`extract_numbers`, `extract_hashtags`, `extract_mentions`, `first_sentence`,
`first_word`, and `last_word`; normalization gives `slugify`, `remove_bullets`,
`remove_repeated_punctuation`, `remove_markdown_links`, `remove_code_blocks`,
`remove_stopwords`, and `truncate_sentences`.

An embedding is a list column, so its vector methods live on `.list` alongside the
reductions above: `dim`, `is_zero_vector`, `sum_squares`, `mean_pool`, `max_pool`,
`magnitude`, `is_unit_norm` (assert normalization before a cosine search),
`euclidean_distance`, and `angular_distance`. Preparing the training set itself
uses `ds.shuffle(seed=)`, `ds.stratified_split(label, test_size)`,
`ds.sample_per_group(by, n)`, `ds.class_balance(label)`, and `ds.class_weights(label)`.

```python
labelled = bt.from_pydict({"y": ["a"] * 6 + ["b"] * 2, "x": list(range(8))})
train, test = labelled.stratified_split("y", 0.25, seed=5)
print(labelled.class_weights("y").sort("y").to_pydict())
# {'y': ['a', 'b'], 'weight': [0.6666666666666666, 2.0]}
```

## Next steps

- [Expressions API](../api/expressions.md): every `Expr` method and accessor
  (`.str`/`.dt`/`.list`/`.struct`/`.json`/`.map`/`.image`/`.audio`/`.video`) in one
  exhaustive reference.
- [Aggregations](aggregations.md) and [Window functions](window-functions.md): where
  aggregate and windowed expressions are used.
- [SQL](sql.md): the same column language, spelled as SQL.
- [Transformations](transformations.md): where expressions are applied to a Dataset.
