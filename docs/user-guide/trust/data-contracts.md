# Data contracts

Some failures no row is responsible for. The table arrived with a tenth of yesterday's rows,
the average order value halved overnight, a column that was 1% null is now 40% null, the feed
stopped three days ago. Every row is individually well-formed, and every row-level constraint
passes.

This page covers the checks that measure the *table*: relation-level aggregates, freshness,
and the schema itself. For the row-level vocabulary and the fail/drop/quarantine choice, see
{doc}`/user-guide/trust/data-quality`.

## What makes a contract relation-level

A relation-level constraint is one number over the whole table, compared against bounds. It
lives on the same {py:obj}`ds.dq <batcher.Dataset.dq>` accessor and mixes freely with row-level checks in one chain,
but it behaves differently at the terminal: there is no violating row, so `drop` and
`quarantine` refuse it rather than quietly enforcing a subset of your contract. Check these
with `validate` or `fail`.

```python
import batcher as bt

orders = bt.from_pydict(
    {
        "order_id": [1, 2, 3, 4, 5],
        "customer_id": [10, 20, 10, 30, None],
        "amount": [12.5, 40.0, 7.25, 99.0, 15.0],
    }
)

report = (
    orders.dq.row_count_between(1, 1_000_000)
    .mean_between("amount", 1.0, 500.0)
    .null_rate_below("customer_id", 0.5)
    .validate()
)
print(report.ok)
# True
```

## Volume

`row_count_between` is the cheapest useful contract there is, and the one most often
missing. Either bound may be `None` for an open side.

```python
print(orders.dq.row_count_between(low=3).validate().ok)
# True
print(str(orders.dq.row_count_between(low=1000).validate()))
# ValidationReport(violations: row_count_between(1000, None)=1)
```

## Distribution

A bound on a summary statistic catches the failures that arrive as valid values: a unit
change, a truncated feed, a default silently applied upstream. `mean_between` and
`sum_between` are the direct measures; `median_between` and `quantile_between` survive the
outliers that move a mean, which makes them the right choice on a skewed column;
`stddev_between` catches a column that was replaced by a constant.

```python
report = (
    orders.dq.median_between("amount", 5.0, 50.0)
    .quantile_between("amount", 0.9, 40.0, 200.0)
    .stddev_between("amount", 0.01, None)
    .validate()
)
print(report.ok)
# True
```

Each result carries the number it measured, so a failing contract says how far off it was
rather than only that it failed.

```python
result = orders.dq.mean_between("amount", 1.0, 10.0).validate().result(
    "mean_between(amount, 1.0, 10.0)"
)
print(result.ok, round(result.value, 2))
# False 34.75
```

## Missingness and cardinality

`null_rate_below` is the tolerated form of `not_null`, for a column that is legitimately
sparse but whose sparsity is itself the signal. `distinct_count_between` bounds a
categorical column's vocabulary, and `unique_ratio_above` states "this is nearly a key" in a
way that stays true as the table grows.

```python
report = (
    orders.dq.null_rate_below("customer_id", 0.25)
    .distinct_count_between("customer_id", 1, 100)
    .unique_ratio_above("order_id", 0.99)
    .validate()
)
print(report.ok, round(report.result("null_rate_below(customer_id, 0.25)").value, 2))
# True 0.2
```

One of the five customer ids is NULL, so the null rate is 0.2 and a 0.25 bound holds. Read
`result(...).value` whenever a bound surprises you: the measured number is carried on the
result, so you never have to re-derive it by hand.

## Freshness

Nothing about the values in a table is wrong when an upstream feed stops. `fresh_within`
measures the age of the newest row against the wall clock, which is the only check that
notices.

```python
import datetime as dt

feed = bt.from_pydict({"event_time": [dt.datetime(2020, 1, 1)]})
print(str(feed.dq.fresh_within("event_time", "1d").validate()))
# ValidationReport(violations: fresh_within(event_time, 1d)=1)
```

`max_age` accepts a duration string (`"1d"`, `"6h"`, `"90m"`), a `datetime.timedelta`, or a
number of seconds. The clock is read once when the constraint is built and enters the plan
as a literal, so the answer is identical single-node and distributed.

Its counterpart is row-level: `not_in_future` flags a timestamp dated ahead of now, with a
`tolerance` to absorb clock skew between producers.

```python
mixed = bt.from_pydict({"ts": [dt.datetime(2020, 1, 1), dt.datetime(2999, 1, 1)]})
print(mixed.dq.not_in_future("ts", tolerance="5m").validate().violations)
# {'not_in_future(ts, tolerance=300s)': 1}
```

## Schema contracts

The schema is known before anything runs, so a schema constraint costs nothing and is worth
putting first in every chain. When a column is missing, every value constraint written
against it fails too, and a report naming five broken checks hides the one cause.

`has_columns` requires columns to be present, `column_types` pins their types, and
`no_unexpected_columns` catches the *widening* change: a new column is harmless to every
query that names its columns, right up to the point where it carries data nobody has
classified.

```python
contract = orders.dq.has_columns("order_id", "amount").column_types(
    {"order_id": "int64", "amount": "float64"}
)
print(contract.validate().ok)
# True

drifted = bt.from_pydict({"order_id": ["1"], "amount": [1.0], "internal_note": ["x"]})
report = (
    drifted.dq.column_types({"order_id": "int64"})
    .no_unexpected_columns("order_id", "amount")
    .validate()
)
print(report.result("column_types(order_id)").detail)
# order_id: string != int64
print(report.result("no_unexpected_columns(2 allowed)").detail)
# unexpected: internal_note
```

Because a missing column cannot be quarantined, an unsatisfied schema contract raises
{py:exc}`DataQualityError <batcher.DataQualityError>` from `drop`, `quarantine`, and `annotate` rather than producing rows
that would be wrong for a reason the error would not name.

## Publishing the result

`ValidationReport.to_dict` renders the whole report as plain data: the summary counts plus
one entry per constraint with its severity, tolerance, pass rate, and measured value. That
is the shape a metrics sink or a run log wants.

```python
payload = orders.dq.row_count_between(1).mean_between("amount", 1.0, 500.0).validate().to_dict()
print(payload["ok"], len(payload["constraints"]))
# True 2
print(sorted(payload["constraints"][1]))
# ['kind', 'mostly', 'name', 'ok', 'pass_rate', 'rows', 'severity', 'value', 'violations']
```

Pair it with `severity="warn"` to chart a contract before enforcing it: the constraint is
measured and reported on every run, and never fails one.

## Profiling before you write the contract

Writing bounds without profiling first is guessing. {py:meth}`describe <batcher.Dataset.describe>`,
{py:meth}`null_count <batcher.Dataset.null_count>`, and {py:meth}`n_unique <batcher.Dataset.n_unique>` give you the numbers the
bounds should be built from, and {doc}`/user-guide/analyze/metadata-shortcuts` answers many of
them from file footers without reading the data at all.

```python
print(orders.select("amount").describe().to_pydict()["statistic"])
# ['count', 'null_count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
print(orders.null_count().to_pydict()["customer_id"])
# [1]
```

## See also

- {doc}`Data quality </user-guide/trust/data-quality>`: the row-level constraint vocabulary, and the fail/drop/quarantine choice.
- {doc}`Metadata shortcuts </user-guide/analyze/metadata-shortcuts>`: answers read from footers, which is where a contract that holds should get its answer.
- {doc}`Aggregations </user-guide/analyze/aggregations>`: the same statistics, when you want them as a result rather than a bound.
- {doc}`Agent skills </agents>`: `validate-data-quality`.
