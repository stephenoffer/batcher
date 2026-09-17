# Pivoting and reshaping

This page covers reshaping a dataset between long and wide form with {py:meth}`pivot <batcher.Dataset.pivot>` and {py:meth}`unpivot <batcher.Dataset.unpivot>`, from Python or from SQL.

`pivot` is the one relational operator whose *output schema depends on the data*. To know that the columns are `q1`, `q2`, `q3`, something has to read the `quarter` column first. That costs an eager pre-pass, and it means a pivot cannot stream. Everything awkward about pivoting follows from that one fact, so start there.

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

## Pivot long to wide

{py:meth}`pivot(index=, on=, values=, aggregate="sum") <batcher.Dataset.pivot>` groups by `index`, spreads the distinct values of `on` into columns, and fills each cell with `aggregate(values)`.

The figure follows the setup data through a pivot and back out through an unpivot. The two west `q1` rows are the ones to watch, because they're the cell that needs the aggregate.

![Three tables, left to right. The long setup table has five rows of region, quarter and amount: west q1 10.0, west q2 20.0, east q1 30.0, east q2 40.0 and west q1 5.0. A pivot with on="quarter" and aggregate="sum" turns it into a wide table with one row per region: east with q1 30.0 and q2 40.0, and west with q1 15.0, the sum of 10.0 and 5.0, and q2 20.0. An unpivot turns the columns back into rows, one per region and quarter: east q1 30.0, east q2 40.0, west q1 15.0 and west q2 20.0. Below, the pivot's output columns come from the data, so a pre-pass reads quarter unless you pass columns=, and it groups, so it is a pipeline breaker. The unpivot's schema is fixed by its arguments, with no pre-pass and no breaker, so it streams and distributes like a select.](/_static/diagrams/pivot_long_wide.svg)

```python
wide = sales.pivot(index=["region"], on="quarter", values="amount")
print(wide.sort("region").to_pydict())
# {'region': ['east', 'west'], 'q1': [30.0, 15.0], 'q2': [40.0, 20.0]}
```

West has two `q1` rows, 10.0 and 5.0, and they add to 15.0. There is no such thing as a pivot without an aggregate: if a cell can hold two rows, something has to combine them. `aggregate` is one of `sum`, `mean`, `min`, `max`, or `count`, and `aggfunc` is accepted as the pandas spelling of the same argument.

```python
print(
    sales.pivot(index=["region"], on="quarter", values="amount", aggregate="mean")
    .sort("region")
    .to_pydict()
)
# {'region': ['east', 'west'], 'q1': [30.0, 7.5], 'q2': [40.0, 20.0]}
```

## Fix the columns and skip the pre-pass

:::{warning}
Omit `columns` and the engine runs an eager pass over `on` to discover the distinct values, before the real query even starts. On a large scan that is a second read of the data, and worse, it makes the output schema unpredictable: a month with no rows yet has no column, so a downstream {py:meth}`select("q3") <batcher.Dataset.select>` fails on Tuesday and works on Wednesday.
:::

Pass `columns=[...]` when you know the vocabulary. The pre-pass disappears, the schema is fixed, and a missing value shows up as a null column instead of a missing one.

```python
fixed = sales.pivot(index=["region"], on="quarter", values="amount", columns=["q1", "q2", "q3"])
print(fixed.sort("region").to_pydict())
# {'region': ['east', 'west'], 'q1': [30.0, 15.0], 'q2': [40.0, 20.0], 'q3': [None, None]}
```

A value present in the data but absent from `columns` is dropped. That is the trade: you get a stable schema by declaring it, and declaring it means owning it.

:::{note}
Mind the cardinality. `on` a column with 50,000 distinct values produces a 50,000 column table, and nothing in the API stops you. Pivot on a dimension with a small, known domain, such as quarter, status, or country. For anything wider, keep it long and {py:meth}`group_by <batcher.Dataset.group_by>` it.
:::

## Unpivot wide to long

The inverse, and the one you reach for far more often. A wide input from a spreadsheet or a warehouse export is usually the wrong shape for everything downstream.

```python
report = bt.from_pydict({"region": ["west", "east"], "q1": [15.0, 30.0], "q2": [20.0, 40.0]})
print(report.unpivot(index=["region"]).to_pydict())
# {'region': ['west', 'east', 'west', 'east'], 'variable': ['q1', 'q1', 'q2', 'q2'],
#  'value': [15.0, 30.0, 20.0, 40.0]}
```

Every non-`index` column melts by default. Name the outputs to get something you can read, and pass `on` to melt only some of the columns.

```python
long = report.unpivot(
    index=["region"], on=["q1", "q2"], variable_name="quarter", value_name="amount"
)
print(long.to_pydict())
# {'region': ['west', 'east', 'west', 'east'], 'quarter': ['q1', 'q1', 'q2', 'q2'],
#  'amount': [15.0, 30.0, 20.0, 40.0]}
```

The melted columns must share a type, since they end up in one output column and Arrow has no union-typed column here. Melting an int column and a string column together is an error, not a silent cast. `cast` them to a common type first if that is really what you mean.

`unpivot` is a pure row-wise operator: no breaker, no pre-pass, no schema surprise. It distributes and streams like a `select`. Side by side, the two are not mirror images at all:

| | `pivot` | `unpivot` |
| --- | --- | --- |
| Direction | long to wide | wide to long |
| Output schema | data-dependent, unless you pass `columns` | fixed by the arguments |
| Extra pass over the data | yes, unless you pass `columns` | never |
| Pipeline breaker | yes, it groups | no, it streams |
| Needs an aggregate | yes, a cell can hold many rows | no, a row becomes rows |

## Pivot is a grouped conditional aggregate

:::{tip}
`pivot` lowers to `group_by(index).agg(...)` with one conditional aggregate per pivot value. Once `aggregate=` stops being enough, write that out yourself. It is the same plan shape at the same cost, and you get a different aggregate per column or a filter inside one cell.
:::

Written out by hand, one cell can sum while another counts:

```python
by_hand = sales.group_by("region").agg(
    q1=bt.when(bt.col("quarter") == "q1").then(bt.col("amount")).otherwise(bt.lit(0.0)).sum(),
    q2_rows=bt.when(bt.col("quarter") == "q2").then(bt.lit(1)).otherwise(bt.lit(0)).sum(),
)
print(by_hand.sort("region").to_pydict())
# {'region': ['east', 'west'], 'q1': [30.0, 15.0], 'q2_rows': [1, 1]}
```

## Round-tripping

Pivot then unpivot returns you to the long shape, with the nulls that the wide shape introduced. They are real: a `(region, quarter)` pair with no rows had no value, and the wide form had to invent a cell for it. Drop them explicitly if long-form means "observed rows only".

```python
back = wide.unpivot(index=["region"], variable_name="quarter", value_name="amount")
print(back.drop_nulls().sort("region", "quarter").to_pydict())
# {'region': ['east', 'east', 'west', 'west'], 'quarter': ['q1', 'q2', 'q1', 'q2'],
#  'amount': [30.0, 40.0, 15.0, 20.0]}
```

## Pivot and unpivot in SQL

SQL `PIVOT` and `UNPIVOT` translate straight onto the same two methods, so they share the plan, the cost, and the column rules above. The index columns are whatever the relation has left once the pivot's own columns are accounted for, and the `IN` list plays the part of `columns`, so a SQL pivot never needs the pre-pass.

```python
print(
    bt.sql(
        "SELECT * FROM sales PIVOT (sum(amount) FOR quarter IN ('q1', 'q2')) ORDER BY region",
        sales=sales,
    ).to_pydict()
)
# {'region': ['east', 'west'], 'q1': [30.0, 15.0], 'q2': [40.0, 20.0]}
print(
    bt.sql(
        "SELECT * FROM report UNPIVOT (amount FOR quarter IN (q1, q2)) ORDER BY region, quarter",
        report=report,
    ).to_pydict()
)
# {'region': ['east', 'east', 'west', 'west'], 'quarter': ['q1', 'q2', 'q1', 'q2'],
#  'amount': [30.0, 40.0, 15.0, 20.0]}
```

One `PIVOT` or `UNPIVOT` modifier per table reference is supported. Stacking two raises `NotImplementedError`.

## Transposing rows into columns

{py:meth}`transpose <batcher.Dataset.transpose>` turns each input column into a row and each input row into a column. It suits a small summary table that reads better turned on its side, such as one row per statistic. `column_names` names the output columns by one column's values, and `include_header=True` keeps a column holding each input column's name.

```python
summary = bt.from_pydict({"stat": ["min", "max"], "price": [1.5, 9.0], "qty": [1, 12]})
print(summary.transpose(column_names="stat", include_header=True).to_pydict())
# {'column': ['price', 'qty'], 'max': [9.0, 12.0], 'min': [1.5, 1.0]}
```

The output columns ascend by name. The values share one column type, so `qty` became a float to sit beside `price`, and a set of columns with no common type becomes strings. Without `column_names` the output columns are `column_0`, `column_1` and so on, which ties a name to a row position, so pass `order_by` to say which row comes first.

Like `pivot`, the output schema depends on the data, so `transpose` reads the naming column, or counts the rows, before it builds the plan. Keep it for summary-sized frames.

## See also

- {doc}`Aggregations </user-guide/analyze/aggregations>`: the aggregate a pivot cell is built from.
- {doc}`Transformations </user-guide/transform/rows/transformations>`: `explode` and `unnest`, the other two reshapers.
- {doc}`SQL </user-guide/analyze/sql>`: the rest of the SQL surface, which {py:func}`bt.sql(...) <batcher.sql>` shares with these methods.
- {doc}`Aggregation internals </architecture/deep-dives/operators/aggregation-internals>`: the grouped hash aggregate a pivot cell is computed by.
- {doc}`Time-series rollups </cookbook/analytics/aggregates/time-series-rollups>`: a wide report built from a long fact table.
- {doc}`Cohort analysis </cookbook/analytics/behavior/cohort-analysis>`: the other classic pivot, with a declared column vocabulary.
- {doc}`Dataset API </api/relational/dataset>`: the `pivot`, `unpivot` and `transpose` reference.
- {doc}`/cookbook/dataset/verbs/reshaping`: pivot, unpivot, explode, and unnest, as a runnable script.
