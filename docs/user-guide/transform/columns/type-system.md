# The type system

Read a Parquet file with an `Int32` id column, print the schema, and you get `int64`.
That is not a bug and it is not a lossy cast. It is the boundary contract: Batcher
normalizes narrow numeric types **once**, at the FFI edge, so every operator, the
interpreter, and the JIT work on two numeric paths (`Int64` and `Float64`) instead of
twelve. Knowing this up front saves you an afternoon of confusion the first time a schema
assertion fails.

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

This is the whole table. Read it once and the rest of the page follows.

| Source type | Becomes | Why |
| --- | --- | --- |
| `Int8`, `Int16`, `Int32` | `Int64` | one signed integer path, not four |
| `UInt8`, `UInt16`, `UInt32`, `UInt64` | `Int64` | the same, and unsigned arithmetic stops being a special case |
| `Float16`, `Float32` | `Float64` | one floating-point path |
| Dictionary-encoded | its value type, decoded | no operator has to special-case an encoding |
| Everything else | itself | strings, dates, timestamps, decimals, lists, structs, tensors |

```python
print(ds.schema)
# i8: int64
# i32: int64
# f32: double
# name: string
# day: date32[day]
```

The widening is **value-preserving** (`Int32` fits in `Int64`, `Float32` in `Float64`)
and it happens on the way in, so `ds.schema` tells you the truth without executing
anything.

Consequences worth internalizing:

An `Int32` overflow that would have wrapped in another engine does not wrap here, because
the arithmetic runs in 64 bits. A `Float32` sum accumulates in double precision, so it
differs slightly from a `Float32` engine's answer, and it is the more accurate of the two.

:::{warning}
A schema assertion copied from a pandas or Spark test fails on the type *name*, and it
reads as a data bug when it is not one. `int32` in the file is `int64` in the
`Dataset`, every time. Assert on the value, or assert on `int64`.
:::

## Getting narrow types back on output

By default, output types match `Dataset.schema` exactly: what you see before running is
what you get after. If a narrow output column matters, because you are writing Parquet
and want the smaller footprint, turn on `shrink_output_dtypes`. A pass-through of a narrow
*source* column is then cast back to its source width where that is lossless.

```python
from batcher.config import Config, ExecutionConfig, config_context

with config_context(Config().replace(execution=ExecutionConfig(shrink_output_dtypes=True))):
    narrowed = bt.from_arrow(table).select("i32", "f32").collect()
    print(narrowed.schema.field("i32").type, narrowed.schema.field("f32").type)
# int32 float
```

It is off by default because it is data-dependent: a derived column has no source width
to shrink to, so only a straight pass-through narrows. Do not rely on it to control the
type of a computed column. `cast` that one explicitly.

:::{note}
The re-narrowing happens on the way out of the engine, so a `collect()` with no operations
at all (`bt.from_arrow(t).collect()`, a bare scan) skips it and hands back the normalized
`Int64`. Any real query takes the engine path and narrows, whether that is a `select`, a
`filter`, or anything else. If you want the narrow type from a bare scan, project the
columns.
:::

## cast and try_cast

`cast(type)` converts, and fails loudly on a value that cannot convert. `try_cast(type)`
turns the unconvertible value into a null.

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
On ingest of anything you did not produce yourself, `try_cast` is nearly always the right
one, and a `filter(col("n").is_null())` afterwards tells you exactly what it could not
parse.
:::

Cast targets are named as strings. The accepted names, with their aliases:

| Name | Aliases | Arrow type |
| --- | --- | --- |
| `int64` | `long` | 64-bit signed integer |
| `int32` | `int` | 32-bit signed integer |
| `float64` | `double` | 64-bit float |
| `float32` | `float` | 32-bit float |
| `string` | `utf8` | UTF-8 string |
| `bool` | `boolean` | boolean |
| `date32` | `date` | days since epoch |
| `timestamp` | `datetime` | microsecond timestamp |

A cast *inside* a query does produce the narrow type. The boundary normalization is
about what crosses the FFI edge, not about what an expression may compute.

```python
print(ds.select(small=bt.col("i32").cast("int32")).schema)
# small: int32
```

`ds.cast({"col": "type"})` casts several columns at once, and `strict=False` makes the
whole set behave the way `try_cast` does.

```python
print(ds.cast({"i32": "float64", "i8": "string"}).schema)
# i8: string
# i32: double
# f32: double
# name: string
# day: date32[day]
```

## Null is absence, NaN is a value

They are not the same thing and no operator conflates them. A null has no value. A NaN
is a float, the result of an operation such as `0.0 / 0.0`. `is_null()` never sees a NaN, and
`fill_null()` never replaces one. `fill_nan()` does.

```python
mixed = bt.from_pydict({"x": [1.0, float("nan"), None]})
print(mixed.select(
    null=bt.col("x").is_null(),
    nan=bt.col("x").is_nan(),
    filled=bt.col("x").fill_null(-1.0),
).to_pydict())
# {'null': [False, False, True], 'nan': [False, True, None], 'filled': [1.0, nan, -1.0]}
```

:::{important}
Look at the `nan` column: `is_nan()` on a *null* is null, not False. Three-valued logic
applies to every predicate, which is why `filter(bt.col("x") > 0)` drops null rows:
`null > 0` is null, and a filter keeps only rows that are *true*. A predicate you expect
to partition the data into two halves partitions it into three.
:::

Where NaN and `-0.0` do get canonicalized is in a hash key: grouping, `distinct`, joins,
and shuffles all treat every NaN as one key and `-0.0` as `0.0`, so a group cannot split
across partitions. See {doc}`distinct and dedup </user-guide/transform/rows/distinct-and-dedup>`.

## Integer division and mixed arithmetic

Arithmetic between an integer and a float promotes to float, as it does in SQL and
NumPy. Integer-by-integer division promotes too, so `7 / 2` is `3.5` and not `3`.

```python
nums = bt.from_pydict({"a": [7, 8], "b": [2, 3]})
print(nums.select(
    div=bt.col("a") / bt.col("b"),
    mod=bt.col("a") % bt.col("b"),
    mixed=bt.col("a") + 0.5,
).to_pydict())
# {'div': [3.5, 2.6666666666666665], 'mod': [1, 2], 'mixed': [7.5, 8.5]}
```

If you want floor division, be explicit: `(bt.col("a") / bt.col("b")).floor()`.

## Inspecting types without running anything

`schema` gives the pyarrow `Schema`, `dtypes` the list of types, and `columns` the names.
They are answered from the plan, so they cost nothing.

```python
print(ds.columns)
# ['i8', 'i32', 'f32', 'name', 'day']
print(ds.dtypes[:3])
# [DataType(int64), DataType(int64), DataType(double)]
```

Because these are plan-derived, they are also the fastest way to catch a schema mistake:
a bad `select` or a missing `output_columns` on a UDF fails here, before a single row is
read.

## Nested types

Lists, structs, and maps pass through the boundary unchanged, and each has an accessor
namespace rather than a pile of top-level functions (`.list`, `.struct`, `.map`,
`.json`). `explode` turns a list column into rows; `unnest` lifts a struct's fields into
top-level columns.

```python
nested = bt.from_pydict({"id": [1, 2], "tags": [["x", "y"], ["z"]]})
print(nested.select("id", n=bt.col("tags").list.len()).to_pydict())
# {'id': [1, 2], 'n': [2, 1]}
print(nested.explode("tags").to_pydict())
# {'id': [1, 1, 2], 'tags': ['x', 'y', 'z']}
```

A fixed-shape tensor column (every row the same N-dimensional shape) is Arrow's
canonical tensor type, so the shape travels with the data across the FFI edge and
arrives at a model stage correctly shaped. See {doc}`multimodal </ml/preparing/multimodal/index>`.

## See also

- {doc}`Expressions </user-guide/transform/columns/expressions>`: the full method surface, per type.
- {doc}`Reading data </user-guide/moving-data/reading-data>`: schema inference and schema evolution on the way in.
- {doc}`Data quality </user-guide/trust/data-quality>`: assert the types and ranges you expect, instead of
  discovering them.
- {doc}`Arrow memory </architecture/deep-dives/memory/arrow-memory>`: the zero-copy boundary the
  normalization happens at, and why it happens exactly once.
- {doc}`Expression evaluation </architecture/deep-dives/query/expression-evaluation>`: the two numeric paths
  the widening buys, and what the JIT does with them.
- {doc}`Tensor columns </architecture/deep-dives/memory/tensor-columns>`: fixed-shape tensors, the one nested
  type with a shape contract.
- {doc}`Schema evolution </cookbook/data-engineering/modeling/schema-evolution>`: types that change
  under you between files.
- {doc}`Expressions API </api/relational/expressions>`: the `cast` / `try_cast` reference.
- {doc}`/cookbook/expressions/scalar/nulls_and_casting`: the two places a pipeline quietly changes its answer.
