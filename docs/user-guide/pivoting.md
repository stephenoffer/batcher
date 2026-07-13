# Pivoting and reshaping

`pivot` is the one relational operator whose *output schema depends on the data*. To
know that the columns are `q1`, `q2`, `q3`, something has to read the `quarter` column
first. That costs an eager pre-pass, and it means a pivot cannot stream. Everything
awkward about pivoting follows from that one fact, so start there.

## Setup

```python
import batcher as bt

sales = bt.from_pydict(
    {
        "region": ["west", "west", "east", "east", "west"],
        "quarter": ["q1", "q2", "q1", "q2", "q1"],
        "amount": [10.0, 20.0, 30.0, 40.0, 5.0],
    }
)
```

## pivot: long to wide

`pivot(index=, on=, values=, aggregate="sum")` groups by `index`, spreads the distinct
values of `on` into columns, and fills each cell with `aggregate(values)`.

```python
wide = sales.pivot(index=["region"], on="quarter", values="amount")
print(wide.sort("region").to_pydict())
# {'region': ['east', 'west'], 'q1': [30.0, 15.0], 'q2': [40.0, 20.0]}
```

West has two `q1` rows (10.0 and 5.0); they add to 15.0. There is no such thing as a
pivot without an aggregate: if a cell can hold two rows, something has to combine them.
`aggregate` is one of sum, mean, min, max, count.

```python
print(sales.pivot(index=["region"], on="quarter", values="amount", aggregate="mean")
      .sort("region").to_pydict())
# {'region': ['east', 'west'], 'q1': [30.0, 7.5], 'q2': [40.0, 20.0]}
```

## Fix the columns and skip the pre-pass

:::{warning}
Omit `columns` and the engine runs an eager pass over `on` to discover the distinct
values, before the real query even starts. On a large scan that is a second read of the
data, and worse, it makes the output schema unpredictable: a month with no rows yet
simply has no column, so a downstream `select("q3")` fails on Tuesday and works on
Wednesday.
:::

Pass `columns=[...]` when you know the vocabulary. The pre-pass disappears, the schema
is fixed, and a missing value shows up as a null column instead of a missing one.

```python
fixed = sales.pivot(
    index=["region"], on="quarter", values="amount", columns=["q1", "q2", "q3"]
)
print(fixed.sort("region").to_pydict())
# {'region': ['east', 'west'], 'q1': [30.0, 15.0], 'q2': [40.0, 20.0], 'q3': [None, None]}
```

A value present in the data but absent from `columns` is dropped. That is the trade:
you get a stable schema by declaring it, and declaring it means owning it.

:::{note}
Mind the cardinality. `on` a column with 50,000 distinct values produces a 50,000 column
table, and nothing in the API stops you. Pivot on a dimension with a small, known domain
(quarter, status, country); for anything wider, keep it long and `group_by` it.
:::

## unpivot: wide to long

The inverse, and the one you reach for far more often. A wide input from a spreadsheet or
a warehouse export is usually the wrong shape for everything downstream.

```python
report = bt.from_pydict({"region": ["west", "east"], "q1": [15.0, 30.0], "q2": [20.0, 40.0]})
print(report.unpivot(index=["region"]).to_pydict())
# {'region': ['west', 'east', 'west', 'east'], 'variable': ['q1', 'q1', 'q2', 'q2'],
#  'value': [15.0, 30.0, 20.0, 40.0]}
```

Every non-`index` column melts by default. Name the outputs to get something you can
read, and pass `on` to melt only some of the columns.

```python
long = report.unpivot(
    index=["region"], on=["q1", "q2"], variable_name="quarter", value_name="amount"
)
print(long.to_pydict())
# {'region': ['west', 'east', 'west', 'east'], 'quarter': ['q1', 'q1', 'q2', 'q2'],
#  'amount': [15.0, 30.0, 20.0, 40.0]}
```

The melted columns must share a type, since they end up in one output column and Arrow
has no union-typed column here. Melting an int column and a string column together is an
error, not a silent cast; `cast` them to a common type first if that is really what you
mean.

`unpivot` is a pure row-wise operator: no breaker, no pre-pass, no schema surprise. It
distributes and streams like a `select`. Side by side, the two are not mirror images at
all:

| | `pivot` | `unpivot` |
| --- | --- | --- |
| Direction | long → wide | wide → long |
| Output schema | data-dependent, unless you pass `columns` | fixed by the arguments |
| Extra pass over the data | yes, unless you pass `columns` | never |
| Pipeline breaker | yes, it groups | no, it streams |
| Needs an aggregate | yes, a cell can hold many rows | no, a row becomes rows |

## Pivot is a grouped conditional aggregate

:::{tip}
`pivot` lowers to `group_by(index).agg(...)` with one conditional aggregate per pivot
value. Once `aggregate=` stops being enough, write that out yourself: same plan shape,
same cost, and you get a different aggregate per column or a filter inside one cell.
:::

Writing it out by hand is what you do when you need something the operator does not
offer.

```python
by_hand = sales.group_by("region").agg(
    q1=bt.when(bt.col("quarter") == "q1").then(bt.col("amount")).otherwise(bt.lit(0.0)).sum(),
    q2_rows=bt.when(bt.col("quarter") == "q2").then(bt.lit(1)).otherwise(bt.lit(0)).sum(),
)
print(by_hand.sort("region").to_pydict())
# {'region': ['east', 'west'], 'q1': [30.0, 15.0], 'q2_rows': [1, 1]}
```

Same plan shape, same cost, and full control. Reach for it as soon as `aggregate=` stops
being enough.

## Round-tripping

Pivot then unpivot returns you to the long shape, with the nulls that the wide shape
introduced. They are real: a `(region, quarter)` pair with no rows had no value, and
the wide form had to invent a cell for it. Drop them explicitly if long-form means
"observed rows only".

```python
back = wide.unpivot(index=["region"], variable_name="quarter", value_name="amount")
print(back.drop_nulls().sort("region", "quarter").to_pydict())
# {'region': ['east', 'east', 'west', 'west'], 'quarter': ['q1', 'q2', 'q1', 'q2'],
#  'amount': [30.0, 40.0, 15.0, 20.0]}
```

## See also

- [Aggregations](aggregations.md): the aggregate a pivot cell is built from.
- [Transformations](transformations.md): `explode` and `unnest`, the other two reshapers.
- [SQL](sql.md): the SQL surface. Note that SQL `PIVOT` / `UNPIVOT` are *not* supported and
  raise `NotImplementedError` — reshaping goes through `ds.pivot(...)` / `ds.unpivot(...)`,
  which you can call on the result of a `bt.sql(...)` query.
- [Aggregation internals](../deep-dives/aggregation-internals.md): the grouped hash
  aggregate a pivot cell is computed by.
- [Time-series rollups](../examples/analytics/time-series-rollups.md): a wide report
  built from a long fact table.
- [Cohort analysis](../examples/analytics/cohort-analysis.md): the other classic pivot,
  with a declared column vocabulary.
- [Dataset API](../api/dataset.md): the `pivot` and `unpivot` reference.
