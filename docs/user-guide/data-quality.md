# Data quality

Data-quality checks live on the `ds.dq` accessor: a chain of expectations over a
dataset, then a terminal action. A constraint is just a boolean expression that is
TRUE for a valid row, so checks compose like any other operation and lower to the
same relational operators — no separate validation engine.

## Setup

```python
import batcher as bt

people = bt.from_pydict(
    {
        "id": [1, 2, 3, 4, 5],
        "email": ["a@x.io", "b@x.io", None, "d@x.io", "e@x.io"],
        "age": [34, 28, 51, 200, 40],
        "country": ["US", "CA", "US", "ZZ", "CA"],
    }
)
```

## Constraints

Constraint methods accumulate on `ds.dq` and return a new accessor, so they chain.
`not_null` forbids nulls, `in_range` bounds a numeric column, and
`accepted_values` restricts a column to a fixed set. The value constraints treat a
NULL as valid so they compose independently — add `not_null` to forbid nulls
explicitly. A terminal method then applies the accumulated checks.

```python
report = (
    people.dq.not_null("email")
    .in_range("age", 0, 120)
    .accepted_values("country", ["US", "CA", "MX"])
    .validate()
)
print(str(report))
# ValidationReport(violations: not_null(email)=1, in_range(age, 0, 120)=1, accepted_values(country)=1)
```

`matches` requires a column to match a regular expression (NULL passes). `check`
takes any boolean expression as a custom constraint with a name.

```python
codes = bt.from_pydict({"sku": ["A1", "B2", "zz", "C3"]})
print(str(codes.dq.matches("sku", r"^[A-Z][0-9]$").validate()))
# ValidationReport(violations: matches(sku, '^[A-Z][0-9]$')=1)

print(str(people.dq.check(bt.col("age") >= 18, name="adult").validate()))
# ValidationReport(ok)
```

## The validation report

`validate` runs the checks and returns a `ValidationReport` of per-constraint
violation counts without raising. Use `.ok` for a single pass/fail signal and
`.total_violations` for the count summed across every constraint.

```python
report = people.dq.not_null("email").in_range("age", 0, 120).validate()
print(report.ok, report.total_violations)
# False 2
```

## Drop invalid rows

`drop` returns only the rows that satisfy every constraint — the cleansing path
when bad rows should simply be removed.

```python
clean = people.dq.in_range("age", 0, 120).not_null("email").drop()
print(clean.sort("id").to_pydict())
# {'id': [1, 2, 5], 'email': ['a@x.io', 'b@x.io', 'e@x.io'], 'age': [34, 28, 40],
#  'country': ['US', 'CA', 'CA']}
```

## Quarantine

`quarantine` returns a `(clean, rejected)` pair so the violating rows route to a
dead-letter sink instead of failing the run. The split is total: every input row
lands in exactly one side.

```python
good, bad = people.dq.in_range("age", 0, 120).quarantine()
print(good.sort("id").to_pydict())
# {'id': [1, 2, 3, 5], 'email': ['a@x.io', 'b@x.io', None, 'e@x.io'],
#  'age': [34, 28, 51, 40], 'country': ['US', 'CA', 'US', 'CA']}
print(bad.to_pydict())
# {'id': [4], 'email': ['d@x.io'], 'age': [200], 'country': ['ZZ']}
```

## Fail the pipeline

`fail` is the data-contract gate at a pipeline boundary: it raises
`DataQualityError` (carrying the per-constraint counts) if any constraint is
violated, and otherwise returns the dataset unchanged so the chain continues.

```python
from batcher._internal.errors import DataQualityError

try:
    people.dq.in_range("age", 0, 120).fail()
except DataQualityError as err:
    print(type(err).__name__)
# DataQualityError

ok = bt.from_pydict({"age": [10, 20, 30]})
print(ok.dq.in_range("age", 0, 120).fail().to_pydict())
# {'age': [10, 20, 30]}
```

## Uniqueness and referential integrity

`unique` requires a key (or key combination) to occur at most once; the report
counts the duplicated keys. `foreign_key` returns the orphan rows whose key has no
match in a reference dataset — an empty result means every key resolves.

```python
dupes = bt.from_pydict({"id": [1, 1, 2, 3, 3]})
print(str(dupes.dq.unique("id").validate()))
# ValidationReport(violations: unique(id)=2)

orders = bt.from_pydict({"order_id": [1, 2, 3], "customer_id": [10, 20, 99]})
customers = bt.from_pydict({"customer_id": [10, 20, 30]})
orphans = orders.dq.foreign_key("customer_id", references=customers)
print(orphans.to_pydict())
# {'order_id': [3], 'customer_id': [99]}
```

## Deduplication

`distinct` removes duplicate rows. With no argument it deduplicates over all
columns; with a `subset` it keeps one row per key combination. Pass
`keep="first"`/`"last"` with `order_by` to pick which row survives — the standard
"latest record per key" pattern.

```python
events = bt.from_pydict(
    {
        "user": ["a", "a", "b", "b"],
        "ts": [1, 2, 1, 2],
        "val": [10, 11, 20, 21],
    }
)
latest = events.distinct(subset=["user"], keep="last", order_by="ts")
print(latest.sort("user").to_pydict())
# {'user': ['a', 'b'], 'ts': [2, 2], 'val': [11, 21]}
```

## Evolving schemas

When a directory of files was written over time, later files may add columns or
widen a type. Pass `schema_mode="union"` to a read so the files reconcile into one
schema — the union of columns, each promoted to a common type, with missing columns
filled as null. Use `"latest"` to let the newest file's schema win, or `"strict"`
(the default) to require every file to match.

```python
import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

root = tempfile.mkdtemp()
pq.write_table(pa.table({"id": [1, 2], "amount": [10, 20]}), os.path.join(root, "day1.parquet"))
pq.write_table(
    pa.table({"id": [3], "amount": [30], "region": ["us"]}), os.path.join(root, "day2.parquet")
)

evolved = bt.read.parquet(root, schema_mode="union").sort("id")
print(evolved.to_pydict())
# {'id': [1, 2, 3], 'amount': [10, 20, 30], 'region': [None, None, 'us']}
```

## Next steps

- [Reading data](reading-data.md): ingest files and reconcile evolving schemas.
- [Transformations](transformations.md): cleanse and reshape the validated data.
- [Aggregations](aggregations.md): summarize the validated data.
- [Dataset API](../api/dataset.md): the full reference for `ds.dq` and `distinct`.
