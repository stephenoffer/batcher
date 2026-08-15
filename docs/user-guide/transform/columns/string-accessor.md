# String accessor: .str

This page covers the {py:class}`.str <batcher.plan.expr_ir.namespaces.strings._StrNamespace>`
namespace, which holds every method that only makes sense for a string column. The other
namespaces are in {doc}`/user-guide/transform/columns/expression-accessors`.

Every example runs against the engine, and the blocks below share one namespace and
execute in order.

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
1-based field of a split, {py:meth}`substring_index(delimiter, count) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.substring_index>` keeps everything up
to the `count`-th delimiter, and `normalize_whitespace` collapses each run of
whitespace to one space and trims the ends. {py:meth}`zfill(width) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.zfill>` zero-pads fixed-width
codes, and {py:meth}`contains_any([...]) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.contains_any>` tests a row against several literal substrings at
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

`ascii` returns the codepoint of the first character; `bit_length` and `octet_length`
measure the encoded size in bits and UTF-8 bytes, not characters. {py:meth}`levenshtein(target) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.levenshtein>`
gives the edit distance to a constant string and `soundex` its phonetic key.

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

{py:meth}`hamming(target) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.hamming>` counts the positions at which two equal-length strings differ, which is the
right distance for fixed-width codes. `jaccard(target)` scores the overlap of two
values' character sets. `hamming` raises on unequal lengths rather than comparing a
prefix, because a prefix comparison answers a caller's mistake with a plausible number.

## Paths and URLs

Text columns often hold a file path or a URL component rather than prose. {py:meth}`parse_path <batcher.plan.expr_ir.namespaces.strings._StrNamespace.parse_path>`
splits a path into its parts, and {py:meth}`parse_filename <batcher.plan.expr_ir.namespaces.strings._StrNamespace.parse_filename>`, {py:meth}`parse_dirname <batcher.plan.expr_ir.namespaces.strings._StrNamespace.parse_dirname>` and {py:meth}`parse_dirpath <batcher.plan.expr_ir.namespaces.strings._StrNamespace.parse_dirpath>`
pick one out. The last two are easy to confuse and are genuinely different:
`parse_dirname` is the *first* component (`/` for an absolute path) while `parse_dirpath`
is the directory holding the file.

{py:meth}`url_encode <batcher.plan.expr_ir.namespaces.strings._StrNamespace.url_encode>` percent-encodes a URL *component* (`/` and `+` included) and {py:meth}`url_decode <batcher.plan.expr_ir.namespaces.strings._StrNamespace.url_decode>`
reverses it; a malformed escape decodes to itself. {py:meth}`regexp_escape <batcher.plan.expr_ir.namespaces.strings._StrNamespace.regexp_escape>` ({py:meth}`escape_regex <batcher.plan.expr_ir.namespaces.strings._StrNamespace.escape_regex>` in Polars) neutralizes a value's
regex metacharacters so it matches itself.

```python
files = bt.from_pydict({"p": ["/data/2024/events.parquet", "raw/in.csv"]})
out = files.select(
    name=bt.col("p").str.parse_filename(),
    first=bt.col("p").str.parse_dirname(),
    folder=bt.col("p").str.parse_dirpath(),
    parts=bt.col("p").str.parse_path(),
    quoted=bt.col("p").str.url_encode(),
)
print(out.to_pydict())
# {'name': ['events.parquet', 'in.csv'], 'first': ['/', 'raw'], 'folder': ['/data/2024', 'raw'], 'parts': [['/', 'data', '2024', 'events.parquet'], ['raw', 'in.csv']], 'quoted': ['%2Fdata%2F2024%2Fevents.parquet', 'raw%2Fin.csv']}
```

On `.list`, `concat(other)` appends and is deliberately not `union`: it keeps duplicates
and order, and a null list counts as *empty*, so `concat` of `[1,2]` and `[2,3]` is
`[1,2,2,3]` where `union` is `[1,2,3]`. `has_all(other)` and `has_any(other)` test
containment and, unlike `concat`, are null when either side is null.

{py:meth}`to_binary <batcher.plan.expr_ir.namespaces.strings._StrNamespace.to_binary>` renders a value's UTF-8 bytes as `0`/`1` text and {py:meth}`from_binary <batcher.plan.expr_ir.namespaces.strings._StrNamespace.from_binary>` reads it back
(undecodable input is null, the rule `unhex` also follows).

Other `.str` methods include `lower`, `trim`, `lstrip`, `rstrip`, `reverse`,
`substr`, `right`, `repeat`, `lpad`, `rpad`, `position`, `split`, `replace`,
`initcap`, `hex`, `base64`, `from_base64`, `unhex`, and `translate`.

## Recasing identifiers

Column names, event names, and enum values arrive from upstream systems in whatever
convention that system used. {py:meth}`to_case(style) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.to_case>` normalizes them. One splitter finds the
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

## Compressed payloads inside a column

Compressed bytes arrive inside columns, not only inside files: a gzipped JSON body in a
Kafka record, a zstd-framed blob in a warehouse table. {py:meth}`compress(codec) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.compress>` and
{py:meth}`decompress(codec) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.decompress>` handle those without leaving the engine for a Python UDF.

```python
blobs = bt.from_pydict({"body": ["a payload worth compressing " * 10]})
out = blobs.select(
    packed=bt.col("body").str.compress("gzip").str.len_bytes(),
    raw=bt.col("body").str.len_bytes(),
)
print(out.to_pydict())
# {'packed': [51], 'raw': [280]}
```

The codecs are `gzip`, `zlib`, `deflate`, `zstd`, `brotli`, and `lz4`. The frames are the
real thing, so anything else that reads gzip reads what Batcher writes, and the reverse.

```python
records = bt.from_pydict({"body": ["round trip"]})
out = records.select(
    back=bt.col("body").str.compress("zstd").str.decompress("zstd").cast("string")
)
print(out.to_pydict())
# {'back': ['round trip']}
```

A frame that isn't valid for the codec you named decompresses to null rather than failing
the query, so one corrupt blob in a scan of a billion rows costs you that row and nothing
else.

:::{note}
`deflate` is the one codec that can't tell a corrupt frame from a valid one: raw deflate
carries no header and no checksum. Use `zlib` or `gzip` where detection matters. They wrap
the same algorithm in a frame that can be validated.
:::

:::{note}
Recasing is idempotent in every style that joins with a separator. `camel` and `pascal`
join with nothing, so an input with consecutive single-letter words can't survive a round
trip: `a_b_c` becomes `aBC`, which reads back as two words. Prefer a separator style when
the result will be parsed again.
:::

## Regex

Alongside the single-match {py:meth}`regexp_matches <batcher.plan.expr_ir.namespaces.strings._StrNamespace.regexp_matches>`, {py:meth}`regexp_replace <batcher.plan.expr_ir.namespaces.strings._StrNamespace.regexp_replace>`, and {py:meth}`regexp_extract <batcher.plan.expr_ir.namespaces.strings._StrNamespace.regexp_extract>`,
three methods work over *every* match in a string: {py:meth}`regexp_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.regexp_count>` tallies the
matches, {py:meth}`regexp_extract_all <batcher.plan.expr_ir.namespaces.strings._StrNamespace.regexp_extract_all>` gathers them into a list, and
{py:meth}`regexp_replace_all(pattern, replacement) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.regexp_replace_all>` substitutes them all.

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

`regexp_split(pattern)` is the regex counterpart of `split`, for a separator that varies:
a run of whitespace, one of several punctuation marks, a digit boundary. Empty pieces
created by a leading or trailing separator are kept, so the piece count stays one more
than the separator count.

```python
lines = bt.from_pydict({"s": ["alpha, beta;gamma", "one two   three"]})
out = lines.select(
    parts=bt.col("s").str.regexp_split("[,;]\\s*"),
    words=bt.col("s").str.regexp_split(r"\s+"),
)
print(out.to_pydict())
# {'parts': [['alpha', 'beta', 'gamma'], ['one two   three']], 'words': [['alpha,', 'beta;gamma'], ['one', 'two', 'three']]}
```

## Building strings from several columns

Two top-level helpers assemble one string from many expressions.
{py:obj}`bt.format_string(template, *exprs) <batcher.format_string>` interpolates
values into a template at each `{}` placeholder (Polars' `format`), while
{py:obj}`bt.concat_ws(separator, *exprs) <batcher.concat_ws>` joins values with a
separator between them (DuckDB/Spark {py:func}`concat_ws <batcher.concat_ws>`).

```python
out = ds.select(
    label=bt.format_string("{} x{}", bt.col("name"), bt.col("qty")),
    key=bt.concat_ws("-", bt.col("name"), bt.col("qty").cast("utf8")),
)
print(out.to_pydict())
# {'label': ['Ann x1', 'bob x2', 'CARL x3'], 'key': ['Ann-1', 'bob-2', 'CARL-3']}
```

## Parsing text into dates

Parsing string columns into temporal types also lives on `.str`:
{py:meth}`to_date(format) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.to_date>` yields a `Date` and {py:meth}`to_datetime(format) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.to_datetime>` a `Timestamp`, each
reading a chrono/strftime pattern. Once parsed, reach for the `.dt` accessor below.

```python
day_strs = bt.from_pydict({"d": ["2024-01-15", "2024-06-01"]})
print(day_strs.select(day=bt.col("d").str.to_date("%Y-%m-%d")).to_pydict())
# {'day': [datetime.date(2024, 1, 15), datetime.date(2024, 6, 1)]}

stamp_strs = bt.from_pydict({"t": ["2024-01-15 09:30", "2024-06-01 18:00"]})
print(stamp_strs.select(stamp=bt.col("t").str.to_datetime("%Y-%m-%d %H:%M")).to_pydict())
# {'stamp': [datetime.datetime(2024, 1, 15, 9, 30), datetime.datetime(2024, 6, 1, 18, 0)]}
```

## See also

- {doc}`/user-guide/transform/columns/expression-accessors`: the `.dt`, `.list`, `.struct`, `.map` and `.json` namespaces.
- {doc}`/user-guide/transform/columns/expression-recipes`: task-shaped recipes built from these methods.
