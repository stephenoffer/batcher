# The type system

This page covers the types a Batcher column can hold: how they cross into the engine, how to cast between them, how nulls and NaN differ, and what happens when two types meet in one column.

Start with the one surprise. Read a Parquet file with an `Int32` id column, print the schema, and you get `int64`. That is not a bug and not a lossy cast. Batcher normalizes narrow numeric types once, at the FFI edge, so every operator, the interpreter, and the JIT work on two numeric paths, `Int64` and `Float64`, instead of twelve. Knowing that up front saves you an afternoon the first time a schema assertion fails.

## Setup

```python
import batcher as bt
import pyarrow as pa

table = pa.table(
    {
        "i8": pa.array([1, 2], pa.int8()),
        "i32": pa.array([10, 20], pa.int32()),
        "f32": pa.array([1.5, 2.5], pa.float32()),
        "name": pa.array(["a", "b"]),
        "day": pa.array([19000, 19001], pa.date32()),
    }
)
ds = bt.from_arrow(table)
```

## Narrow numerics widen at the boundary

A column crosses the FFI edge once on the way in, and its type can change there. What comes back out depends on one setting, covered in the next section.

![Narrow types widen at the FFI edge. In the source, Int8, Int16 and Int32 widen to Int64 in the engine, UInt8 through UInt64 widen to Int64, Float16 and Float32 widen to Float64, and a dictionary-encoded column is decoded to its value type. Dataset.schema already reports the widened types before anything runs. By default, what comes back is exactly the widened type that schema promised. With shrink_output_dtypes=True, a pass-through column narrows back to its source width. A UInt64 value above the Int64 maximum raises at the edge rather than wrapping. Only a lossless pass-through narrows, and a derived column stays wide, so cast it explicitly. A cast inside a query does produce the narrow type, and a bare scan skips the re-narrowing.](/_static/diagrams/type_widening.svg)

| Source type | Becomes | Why |
| --- | --- | --- |
| `Int8`, `Int16`, `Int32` | `Int64` | one signed integer path, not four |
| `UInt8`, `UInt16`, `UInt32`, `UInt64` | `Int64` | the same, and unsigned arithmetic stops being a special case |
| `Float16`, `Float32` | `Float64` | one floating-point path |
| Dictionary-encoded | its value type, decoded | no operator has to special-case an encoding |
| `LargeUtf8`, `Utf8View` | `Utf8` | the string kernels accept one string layout |
| `BinaryView` | `Binary` | the same, for bytes |
| `ListView`, `LargeListView` | `List` | the list kernels read one offset layout |
| Everything else | itself | `Utf8`, `Binary`, dates, timestamps, decimals, lists, structs, tensors, with narrow types inside a list or struct widened the same way |

```python
print(ds.schema)
# i8: int64
# i32: int64
# f32: double
# name: string
# day: date32[day]
```

The widening is value-preserving, since `Int32` fits in `Int64` and `Float32` in `Float64`. A `UInt64` value above the `Int64` maximum is the one case that does not fit, and it raises at the boundary rather than wrapping. The widening happens on the way in, so {py:obj}`ds.schema <batcher.Dataset.schema>` tells you the truth without executing anything.

Two consequences follow. An `Int32` overflow that would have wrapped in another engine does not wrap here, because the arithmetic runs in 64 bits. A `Float32` sum accumulates in double precision, so it differs slightly from a `Float32` engine's answer, and it is the more accurate of the two.

Widening moves the overflow boundary. It does not remove it. Scalar integer arithmetic *wraps* at the edge of `Int64`, silently, the way Rust and Polars do:

```python
big = bt.from_pydict({"x": [2**63 - 1]})
print(big.select(r=bt.col("x") + 1).to_pydict()["r"])
```

That is deliberate rather than an oversight. The Cranelift JIT compiles `+` to a machine `iadd`, which wraps, and the interpreter is required to be bit-for-bit identical to the compiled tier on every expression it supports. An interpreter that raised where the JIT wrapped would make the same query answer differently depending on whether it compiled.

Reductions do not inherit the convention, because nothing forces them to match a compiled kernel. `sum` over an `Int64` column raises rather than wrapping, and `cum_prod` returns `Float64` for an integer input for the same reason:

```python
try:
    big.agg(total=bt.col("x").sum()).to_pydict()
except Exception as exc:
    print(type(exc).__name__)
```

Whether it raises depends on the total, not on the order the rows arrive in. A column whose large values cancel sums cleanly, even though adding them left to right passes outside `Int64` on the way:

```python
cancels = bt.from_pydict({"x": [2**62, 2**62, -(2**62), -(2**62)]})
print(cancels.agg(total=bt.col("x").sum()).to_pydict()["total"])
```

That distinction matters more than it looks. How a table is split into batches, and across how many machines, is a scheduling decision. If a running total decided the outcome, the same query would succeed on one node and fail across several, on identical data.

One limit remains, and it is worth knowing before you rely on the guarantee. Each partition is summed into its own `Int64` before the partitions are merged, so a partition whose *own* total exceeds `Int64` still raises, even when the totals across partitions would cancel. Row order inside a partition never decides anything. How the rows are divided between partitions still can, in that one case.

So the rule to carry is that an integer *expression* can wrap and an integer *aggregate* cannot. If a column's values approach `2**63` and the arithmetic matters, cast before computing. Use `Float64` for magnitude and `decimal(38, s)` when the digits have to be exact.

```python
print(big.select(r=bt.col("x").cast("float64") + 1).to_pydict()["r"])
```

:::{warning}
A schema assertion copied from a pandas or Spark test fails on the type *name*, and it reads as a data bug when it is not one. `int32` in the file is `int64` in the {py:class}`Dataset <batcher.Dataset>`, every time. Assert on the value, or assert on `int64`.
:::

## Get narrow types back on output

By default, output types match {py:obj}`Dataset.schema <batcher.Dataset.schema>` exactly: what you see before running is what you get after. If a narrow output column matters, because you are writing Parquet and want the smaller footprint, turn on `shrink_output_dtypes`. A pass-through of a narrow *source* column is then cast back to its source width where that is lossless.

```python
from batcher.config import Config, ExecutionConfig, config_context

with config_context(Config().replace(execution=ExecutionConfig(shrink_output_dtypes=True))):
    narrowed = bt.from_arrow(table).select("i32", "f32").collect()
    print(narrowed.schema.field("i32").type, narrowed.schema.field("f32").type)
# int32 float
```

It is off by default because it is data-dependent: a derived column has no source width to shrink to, so only a straight pass-through narrows. Do not rely on it to control the type of a computed column. `cast` that one explicitly.

:::{note}
The re-narrowing happens on the way out of the engine, so a {py:meth}`collect() <batcher.Dataset.collect>` with no operations at all ({py:func}`bt.from_arrow(t).collect() <batcher.from_arrow>`, a bare scan) skips it and hands back the normalized `Int64`. Any real query takes the engine path and narrows, whether that is a `select`, a `filter`, or anything else. If you want the narrow type from a bare scan, project the columns.
:::

## Cast with cast and try_cast

`cast(type)` converts, and fails loudly on a value that cannot convert. `try_cast(type)` turns the unconvertible value into a null.

::::{tab-set}
:::{tab-item} try_cast

```python
raw = bt.from_pydict({"s": ["1", "2", "oops", "4"]})
print(raw.select(n=bt.col("s").try_cast("int64")).to_pydict())
# {'n': [1, 2, None, 4]}
```

:::

:::{tab-item} cast

```python
# docs: skip
# The same data, strictly: "oops" is not an integer, so the query raises
# rather than quietly nulling the row.
raw.select(n=bt.col("s").cast("int64")).to_pydict()
```

:::
::::

:::{tip}
When you ingest anything you did not produce yourself, {py:meth}`try_cast <batcher.plan.expr_ir.core.Expr.try_cast>` is nearly always the right one, and a `filter(col("n").is_null())` afterwards tells you exactly what it could not parse.
:::

### Naming a cast target

Cast targets are named as strings, and the name is matched case-insensitively, so `"Int64"`, `"int64"` and `"BIGINT"` are the same target.

Names that take no parameters:

| Name | Aliases | Arrow type |
| --- | --- | --- |
| `int64` | `long`, `bigint` | 64-bit signed integer |
| `int32` | `int`, `integer` | 32-bit signed integer |
| `int16` | `smallint` | 16-bit signed integer |
| `int8` | `tinyint` | 8-bit signed integer |
| `uint64` | `ubigint` | 64-bit unsigned integer |
| `uint32` | `uinteger` | 32-bit unsigned integer |
| `uint16` | `usmallint` | 16-bit unsigned integer |
| `uint8` | `utinyint` | 8-bit unsigned integer |
| `float64` | `double` | 64-bit float |
| `float32` | `float`, `real` | 32-bit float |
| `float16` | `half` | 16-bit float |
| `string` | `utf8`, `varchar`, `text` | UTF-8 string |
| `large_string` | `large_utf8` | UTF-8 string, 64-bit offsets |
| `binary` | `blob`, `bytea` | raw bytes |
| `large_binary` | | raw bytes, 64-bit offsets |
| `bool` | `boolean` | boolean |
| `date32` | `date` | days since epoch |
| `date64` | | milliseconds since epoch |
| `timestamp` | `datetime` | microsecond timestamp |
| `null` | | the empty type |

Names that carry parameters in parentheses. Spaces inside the parentheses are ignored, so `decimal(12,4)` and `decimal(12, 4)` are the same target:

| Name | Example | What it means |
| --- | --- | --- |
| `decimal(p, s)` | `decimal(12, 4)` | Exact decimal, `p` total digits and `s` after the point. Scale defaults to 0. Aliases: `decimal128`, `numeric`. |
| `decimal256(p, s)` | `decimal256(50, 10)` | The same, past 38 digits. |
| `timestamp(unit)` | `timestamp(ns)` | An instant at `s`, `ms`, `us` or `ns` resolution. |
| `timestamp(unit, tz)` | `timestamp(us, UTC)` | The same, carrying a timezone. |
| `time(unit)` | `time(us)` | Time of day, at the width the resolution requires. |
| `time32(unit)` / `time64(unit)` | `time64(ns)` | Time of day at a specific width. |
| `duration(unit)` | `duration(s)` | An elapsed span. Alias: `interval`. |

```python
money = bt.from_pydict({"raw": ["1.50", "2.25"]})
print(money.select(amt=bt.col("raw").cast("decimal(12,4)")).schema)
# amt: decimal128(12, 4)
```

### Casting in SQL

SQL `CAST` and `TRY_CAST` resolve against the same table, so a SQL type name means the width it says. `CAST(x AS TINYINT)` produces an 8-bit column and raises on a value that does not fit, and `TRY_CAST(x AS TINYINT)` produces the same column with those values nulled. That makes `TRY_CAST` a range check, which is the usual reason to reach for it:

```python
wide = bt.from_pydict({"n": [1, 300, -5]})
print(bt.sql("SELECT TRY_CAST(n AS TINYINT) AS small FROM wide", wide=wide).to_pydict())
# {'small': [1, None, -5]}
```

A type name Batcher has no dtype for raises rather than casting to something else.

:::{important}
A timezone keeps its case where the type name does not. Arrow compares a timezone byte-for-byte, so `timestamp(us, UTC)` and `timestamp(us, utc)` are different types. Write the zone exactly as the IANA name spells it.
:::

An out-of-range parameter is rejected rather than clamped: `decimal(39, 2)` raises, because quietly building the widest decimal that fits would overflow on exactly the values the extra digits were asked for. So does `time32(us)`, since a 32-bit time cannot carry microseconds. Write `time(us)` and let the width follow the resolution.

A cast *inside* a query does produce the narrow type. The boundary normalization is about what crosses the FFI edge, not about what an expression may compute.

```python
print(ds.select(small=bt.col("i32").cast("int32")).schema)
# small: int32
```

{py:meth}`ds.cast({"col": "type"}) <batcher.Dataset.cast>` casts several columns at once, and `strict=False` makes the whole set behave the way `try_cast` does.

```python
print(ds.cast({"i32": "float64", "i8": "string"}).schema)
# i8: string
# i32: double
# f32: double
# name: string
# day: date32[day]
```

## Null is absence, NaN is a value

They are not the same thing and no operator conflates them. A null has no value. A NaN is a float, the result of an operation such as `0.0 / 0.0`. {py:meth}`is_null() <batcher.plan.expr_ir.core.Expr.is_null>` never sees a NaN, and `fill_null()` never replaces one. `fill_nan()` does.

```python
mixed = bt.from_pydict({"x": [1.0, float("nan"), None]})
print(
    mixed.select(
        null=bt.col("x").is_null(),
        nan=bt.col("x").is_nan(),
        filled=bt.col("x").fill_null(-1.0),
    ).to_pydict()
)
# {'null': [False, False, True], 'nan': [False, True, None], 'filled': [1.0, nan, -1.0]}
```

:::{important}
Look at the `nan` column: {py:meth}`is_nan() <batcher.plan.expr_ir.core.Expr.is_nan>` on a *null* is null, not False. Three-valued logic applies to every predicate, which is why `filter(bt.col("x") > 0)` drops null rows: `null > 0` is null, and a filter keeps only rows that are *true*. A predicate you expect to partition the data into two halves partitions it into three.
:::

Where NaN and `-0.0` do get canonicalized is in a hash key: grouping, `distinct`, joins, and shuffles all treat every NaN as one key and `-0.0` as `0.0`, so a group cannot split across partitions. See {doc}`distinct and dedup </user-guide/transform/rows/distinct-and-dedup>`.

## Integer division and mixed arithmetic

Arithmetic between an integer and a float promotes to float, as it does in SQL and NumPy. Integer-by-integer division promotes too, so `7 / 2` is `3.5` and not `3`.

```python
nums = bt.from_pydict({"a": [7, 8], "b": [2, 3]})
print(
    nums.select(
        div=bt.col("a") / bt.col("b"),
        mod=bt.col("a") % bt.col("b"),
        mixed=bt.col("a") + 0.5,
    ).to_pydict()
)
# {'div': [3.5, 2.6666666666666665], 'mod': [1, 2], 'mixed': [7.5, 8.5]}
```

If you want floor division, be explicit: `(bt.col("a") / bt.col("b")).floor()`.

## When two columns must become one

A union, a `coalesce`, a `when`/`otherwise`, a `greatest`, a comparison, and a join key all have to hold two differently-typed values in one place. Batcher answers that with a single *promotion lattice*: the one type both sides widen into, with neither narrowed. The same lattice decides what `schema` reports, so what you see before the query runs is what the query produces.

```python
one = bt.from_arrow(pa.table({"amt": pa.array([1, 2], pa.int64())}))
two = bt.from_arrow(pa.table({"amt": pa.array([1.5, 2.5], pa.float64())}))
print(one.union(two).schema.field("amt").type)
# double
print(sorted(one.union(two).to_pydict()["amt"]))
# [1.0, 1.5, 2.0, 2.5]
```

These are the rules, ordered from the pairs you meet most often to the ones you meet on a bad day:

| The two types | Promote to | Why |
|---|---|---|
| `null` and anything | the other side | An all-null column has no values to lose. |
| two integers of any width | `int64` | The width every integer normalizes to anyway. |
| an integer and a float | `double` | SQL's one deliberately inexact promotion. |
| `bool` and an integer | `int64` | `true` reads as 1, as in SQL. |
| two decimals | the finer scale, the wider integer part | `decimal(10,2)` with `decimal(12,4)` gives `decimal(12,4)`. |
| a decimal and an integer | a decimal wide enough for both | Keeps the cents, which a float round-trip would not. |
| a decimal and a float | `double` | `DOUBLE` dominates `DECIMAL`, as in DuckDB. |
| two timestamps, same zone | the finer resolution | `timestamp[ms]` with `timestamp[us]` gives `timestamp[us]`. |
| a date and a timestamp | the timestamp | A date is midnight, so nothing is lost. |
| `string` and `large_string` | `large_string` | A wider offset holds the narrower one. |

Anything not on that list has no lossless common type, and the query raises instead of guessing. An `int64` column unioned with a `string` one is a data-contract problem, and Batcher will not resolve it by stringifying your numbers.

:::{note}
A join reaches the lattice by a slightly different route. Its row encoder compares keys byte-for-byte and needs the two sides to have the identical type, so Batcher widens both key columns to their common supertype before the encoder sees them. Widening cannot change a key's value, so no match is gained or lost. A pair with no common type still raises, naming both columns.
:::

:::{warning}
Two timestamps in *different* timezones are the one pair that looks promotable and is not. The same stored value denotes a different instant in each, so there is no type that holds both without deciding which zone was meant. Cast one side explicitly.
:::

Three consequences are worth seeing run:

```python
partial = bt.from_arrow(
    pa.table({"k": pa.array([1, 2], pa.int64()), "v": pa.array([None, None], pa.null())})
)
# An all-null column coalesces, compares, and unions like any other.
print(partial.select(v=bt.coalesce(bt.col("v"), bt.col("k"))).to_pydict())
# {'v': [1, 2]}

# Two decimals of differing scale join as numbers, not as encodings.
from decimal import Decimal

coarse = bt.from_arrow(pa.table({"amt": pa.array([Decimal("1.50")], pa.decimal128(10, 2))}))
fine = bt.from_arrow(pa.table({"amt": pa.array([Decimal("1.5000")], pa.decimal128(12, 4))}))
print(coarse.join(fine, on="amt", how="inner").count())
# 1

# Files written at different timestamp resolutions read as one column.
ms = bt.from_arrow(pa.table({"ts": pa.array([1_000], pa.timestamp("ms"))}))
us = bt.from_arrow(pa.table({"ts": pa.array([2_000_000], pa.timestamp("us"))}))
print(ms.union(us).schema.field("ts").type)
# timestamp[us]
```

## Inspect types without running anything

`schema` gives the pyarrow `Schema`, `dtypes` the list of types, and `columns` the names. They are answered from the plan, so they cost nothing.

```python
print(ds.columns)
# ['i8', 'i32', 'f32', 'name', 'day']
print(ds.dtypes[:3])
# [DataType(int64), DataType(int64), DataType(double)]
```

Because these are plan-derived, they are also the fastest way to catch a schema mistake: a bad `select` or a missing `output_columns` on a UDF fails here, before a single row is read.

### The declared schema is what an empty result is made of

The declared types are not only for inspection. A query that matches no rows has no data to take its types from, so the engine builds the empty result out of this same schema. A filter matching nothing still returns properly typed columns rather than null ones.

The schema is all or nothing, so one column the control plane cannot type costs the whole projection its types and every column reports `null`. If `schema` says `null` for a column you know is typed, the cause is usually a different expression in the same `select`. A column carrying no type of its own, such as an all-null column read from JSON, adopts the type of whatever you combine it with. Alone, a numeric function reads it as a double.

```python
src = bt.from_arrow(
    pa.table({"v": pa.array([1.5, 2.5], pa.float64()), "u": pa.array([None, None], pa.null())})
)
nothing = src.filter(bt.col("v") > 100).select(total=bt.col("v").abs())
print(nothing.schema, nothing.collect().schema, sep=" | ")
# total: double | total: double
print(src.select(adopts=bt.col("u") + bt.col("v"), alone=bt.col("u").abs()).schema)
# adopts: double
# alone: double
```

## Nested types

Lists, structs, and maps pass through the boundary unchanged, and each has an accessor namespace rather than a pile of top-level functions ({py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>`, {py:class}`.struct <batcher.plan.expr_ir.namespaces.collections._StructNamespace>`, {py:class}`.map <batcher.plan.expr_ir.namespaces.collections._MapNamespace>`, `.json`). `explode` turns a list column into rows, and `unnest` lifts a struct's fields into top-level columns.

```python
nested = bt.from_pydict({"id": [1, 2], "tags": [["x", "y"], ["z"]]})
print(nested.select("id", n=bt.col("tags").list.len()).to_pydict())
# {'id': [1, 2], 'n': [2, 1]}
print(nested.explode("tags").to_pydict())
# {'id': [1, 1, 2], 'tags': ['x', 'y', 'z']}
```

A nested column can be a *key* too, with one exception. Grouping, `DISTINCT`, joins, windows and `UNION` with `distinct` all identify rows by encoding the key columns into a single comparable byte string, and that encoding is defined for lists, structs, lists of structs and dictionary-encoded columns but not for maps. A map's entries have no canonical order, so there is no stable way to tell two maps apart. A map used as a key is refused with a {py:exc}`PlanError <batcher.PlanError>` naming the column, and the refusal covers a map nested inside a struct or a list as well.

```python
import pyarrow as pa

# `from_pydict` infers a struct from a dict, so a genuine map column needs the type.
maps = bt.from_arrow(
    pa.table(
        {
            "m": pa.array([[("a", 1)], [("a", 2)]], type=pa.map_(pa.string(), pa.int64())),
            "v": pa.array([1, 2], pa.int64()),
        }
    )
)
print(maps.group_by("v").agg(n=bt.count()).to_pydict())
# {'v': [1, 2], 'n': [1, 1]}

try:
    maps.group_by("m").agg(n=bt.count())
except bt.PlanError as exc:
    print(str(exc).split(" \u2014 ")[0])
# group_by(): column 'm' is map<string, int64>, and a map cannot be a key
```

Key on something derived from the map instead, such as `col("m").map.keys()`, `col("m").map.values()`, or a specific lookup. Sorting *by* a map column is unaffected, because a sort compares values directly rather than through that encoder, and carrying a map through a query that does not key on it was never restricted.

A fixed-shape tensor column (every row the same N-dimensional shape) is Arrow's canonical tensor type, so the shape travels with the data across the FFI edge and arrives at a model stage correctly shaped. See {doc}`multimodal </ml/preparing/multimodal/index>`.

## See also

- {doc}`Expressions </user-guide/transform/columns/expressions>`: the full method surface, per type.
- {doc}`Reading data </user-guide/moving-data/reading-data>`: schema inference and schema evolution on the way in.
- {doc}`Data quality </user-guide/trust/data-quality>`: assert the types and ranges you expect, instead of discovering them.
- {doc}`Arrow memory </architecture/deep-dives/memory/arrow-memory>`: the zero-copy boundary the normalization happens at, and why it happens exactly once.
- {doc}`Expression evaluation </architecture/deep-dives/query/expression-evaluation>`: the two numeric paths the widening buys, and what the JIT does with them.
- {doc}`Tensor columns </architecture/deep-dives/memory/tensor-columns>`: fixed-shape tensors, the one nested type with a shape contract.
- {doc}`Schema evolution </cookbook/data-engineering/modeling/schema-evolution>`: types that change under you between files.
- {doc}`Expressions API </api/relational/expressions>`: the `cast` / `try_cast` reference.
- {doc}`/cookbook/expressions/scalar/nulls_and_casting`: the two places a pipeline quietly changes its answer.
