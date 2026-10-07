# Nulls and NaN

This page is the reference for how a null and a float NaN behave in every place a value is compared, combined, counted, ordered, or used as a key. Each answer is a block that runs, and `tests/differential/test_diff_null_nan_matrix.py` checks every one of them against DuckDB.

## What is the difference between null and NaN?

A *null* is a missing value. It has no value at all, so almost anything that touches it is null too: SQL's three-valued logic. A *NaN* is a float value, the result of an operation such as `0.0 / 0.0`. It takes part in arithmetic and comparisons like any other number.

Batcher orders NaN the way DuckDB and PostgreSQL do. NaN is equal to itself and greater than every other number, infinity included. That is not IEEE 754, where `NaN == NaN` is false, but it is what makes NaN sortable and groupable. The {doc}`type system <type-system>` page covers how each one is produced and replaced.

The examples share one small table. In `x`, rows 2 and 5 hold nulls and rows 3 and 4 hold NaN.

```python
import batcher as bt

nan = float("nan")
pairs = bt.from_pydict(
    {"x": [1.0, None, nan, nan, None, 2.0], "y": [1.0, None, nan, 1.0, 1.0, nan]}
)
```

## Comparisons

The following table summarizes the comparison operators:

| Comparison | Answer |
|---|---|
| anything `==`, `<`, `>` null | null |
| `NaN == NaN` | true |
| `NaN > x` for any number `x` | true |
| `x == None` | null on every row, whatever the column's type |
| `x.eq_missing(y)` with both null | true |
| `x.eq_missing(y)` with one null | false |

```python
print(
    pairs.select(
        eq=bt.col("x") == bt.col("y"),
        lt=bt.col("x") < bt.col("y"),
        eq_missing=bt.col("x").eq_missing(bt.col("y")),
        gt_huge=bt.col("x") > 1e308,
    ).to_pydict()
)
# {'eq': [True, None, True, False, None, False],
#  'lt': [False, None, False, False, None, True],
#  'eq_missing': [True, True, True, False, False, False],
#  'gt_huge': [False, None, True, True, None, False]}
```

Comparing with a bare `None` follows SQL rather than Python: `x == None` is not a null test, it is null. Use `is_null()` to test, or `eq_missing(None)`, which is the same thing.

```python
words = bt.from_pydict({"s": ["a", None]})
print(
    words.select(
        eq_none=bt.col("s") == None,  # noqa: E711
        is_null=bt.col("s").is_null(),
        eq_missing=bt.col("s").eq_missing(None),
    ).to_pydict()
)
# {'eq_none': [None, None], 'is_null': [False, True], 'eq_missing': [False, True]}
```

## Boolean logic

`&`, `|`, and `~` are Kleene logic. A null is "unknown", so `False & null` is false and `True | null` is true, because the unknown side cannot change the answer. Every other combination with a null is null.

```python
flags = bt.from_pydict(
    {"a": [True, True, False, False, None, None, None],
     "b": [None, False, None, True, None, True, False]}
)
print(
    flags.select(
        a_and_b=bt.col("a") & bt.col("b"),
        a_or_b=bt.col("a") | bt.col("b"),
        not_a=~bt.col("a"),
    ).to_pydict()
)
# {'a_and_b': [None, False, False, False, None, None, False],
#  'a_or_b': [True, True, None, True, None, True, None],
#  'not_a': [False, False, True, True, None, None, None]}
```

A filter keeps a row only where its predicate is true, so a null predicate drops the row exactly as false does.

## Membership

`is_in` is an `OR` of equality tests, so it inherits both rules above. A null input is null. A `None` in the list turns every non-match into null, because the row might have equalled the missing value. A NaN in the list matches a NaN input. Pass `nulls_equal=True` for the null-safe reading, where the answer is always true or false.

```python
print(
    pairs.select(
        "x",
        in_with_null=bt.col("x").is_in([1.0, None]),
        in_with_nan=bt.col("x").is_in([1.0, nan]),
        null_safe=bt.col("x").is_in([1.0, None], nulls_equal=True),
    ).to_pydict()
)
# {'x': [1.0, None, nan, nan, None, 2.0],
#  'in_with_null': [True, None, None, None, None, None],
#  'in_with_nan': [True, None, True, True, None, False],
#  'null_safe': [True, True, False, False, True, False]}
```

## Aggregates

An aggregate skips nulls. A NaN is a value, so it is not skipped. It propagates through `sum` and `mean`, and because it is the largest number it is the `max`. `count(x)` counts NaN and skips null, and `count()` counts rows.

```python
print(
    pairs.agg(
        sum=bt.col("x").sum(),
        mean=bt.col("x").mean(),
        max=bt.col("x").max(),
        min=bt.col("x").min(),
        count_x=bt.col("x").count(),
        rows=bt.count(),
    ).to_pydict()
)
# {'sum': [nan], 'mean': [nan], 'max': [nan], 'min': [1.0], 'count_x': [4], 'rows': [6]}
```

## Sorting

NaN sorts as the largest number, after every finite value ascending and before them descending. A null is not a number, so it goes last in both directions by default. `nulls_first=True` moves the nulls to the front.

```python
print(pairs.sort("x").to_pydict()["x"])
# [1.0, 2.0, nan, nan, None, None]
print(pairs.sort("x", descending=True).to_pydict()["x"])
# [nan, nan, 2.0, 1.0, None, None]
```

## Keys: grouping, distinct, and joins

A group key, a `distinct` key, and a join key are compared as values rather than through three-valued logic. Grouping and `distinct` put every null in one group and every NaN in one group, as SQL's `GROUP BY` does. Hash keys also canonicalize `-0.0` to `0.0`, so a group cannot split across partitions that disagree about the sign bit of zero. See {doc}`distinct and dedup </user-guide/transform/rows/distinct-and-dedup>` for why.

```python
print(pairs.group_by("x").agg(n=bt.count()).sort("x").to_pydict())
# {'x': [1.0, 2.0, nan, None], 'n': [1, 1, 2, 2]}
```

A join follows the same rule for NaN and zero, as DuckDB's does, and the SQL rule for null: a null key matches nothing, not even another null.

```python
left = bt.from_pydict({"k": [1.0, None, nan, -0.0]})
right = bt.from_pydict({"k": [1.0, None, nan, 0.0], "v": [1, 2, 3, 4]})
print(left.join(right, on="k").sort("v").to_pydict())
# {'k': [1.0, nan, -0.0], 'v': [1, 3, 4]}
```

## See also

- {doc}`The type system <type-system>`: where null and NaN come from, and `fill_null` versus `fill_nan`.
- {doc}`Expressions <expressions>`: `is_null`, `is_nan`, `eq_missing`, and `is_in`.
- {doc}`Sorting </user-guide/transform/rows/sorting>`: `nulls_first` and multi-key sorts.
- {doc}`Distinct and dedup </user-guide/transform/rows/distinct-and-dedup>`: float keys in a dedup.
