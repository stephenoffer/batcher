# Data quality

Data-quality checks live on the {py:obj}`ds.dq <batcher.Dataset.dq>` accessor: a chain of expectations over a
dataset, then a terminal action. A constraint is a boolean expression that is
TRUE for a valid row. Checks therefore compose like any other operation and lower to
the same relational operators. There is no separate validation engine.

This page covers the row-level vocabulary and what you can do with the rows that fail it. For the checks that no single row can violate, such as a row count, a mean, or a freshness bound, see {doc}`/user-guide/trust/data-contracts`.

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
{py:meth}`not_null <batcher.api.dataset.dq.DatasetDQ.not_null>` forbids nulls, {py:meth}`in_range <batcher.api.dataset.dq.DatasetDQ.in_range>` bounds a numeric column, and
{py:meth}`accepted_values <batcher.api.dataset.dq.DatasetDQ.accepted_values>` restricts a column to a fixed set. The value constraints treat a
NULL as valid so they compose independently; add `not_null` to forbid nulls
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

The vocabulary below is the complete row-level set. Each entry is a method on `ds.dq`, and every one of them takes the `mostly` and `severity` modifiers described further down.

| Constraint | A row is valid when |
|---|---|
| `not_null(*cols)` | every named column is present |
| `in_range(col, low, high, closed=...)` | the value falls inside the bounds |
| `positive(col, strict=...)` | the value is above zero, or at least zero |
| `is_finite(col)` | the value is neither NaN nor infinite |
| `accepted_values(col, allowed)` | the value is in the allow-list |
| `rejected_values(col, forbidden)` | the value is not in the deny-list |
| `matches(col, pattern)` | the text matches the regular expression |
| `not_matches(col, pattern)` | the text does not match it |
| `matches_format(col, fmt)` | the text is a valid `email`, `url`, `uuid`, or `ipv4` |
| `str_length_between(col, low, high)` | the character length is inside the bounds |
| `not_empty(col, strip=...)` | the text is not empty, or not blank |
| `compare_columns(left, op, right)` | the two columns compare as stated |
| `not_in_future(col, tolerance=...)` | the timestamp is not dated ahead of now |
| `unique(keys)` | the key combination occurs exactly once |
| `references(cols, to=...)` | the foreign key resolves in another dataset |
| `check(predicate, name=...)` | your own boolean expression is TRUE |

`matches` requires a column to match a regular expression (NULL passes). `check`
takes any boolean expression as a custom constraint with a name.

```python
codes = bt.from_pydict({"sku": ["A1", "B2", "zz", "C3"]})
print(str(codes.dq.matches("sku", r"^[A-Z][0-9]$").validate()))
# ValidationReport(violations: matches(sku, '^[A-Z][0-9]$')=1)

print(str(people.dq.check(bt.col("age") >= 18, name="adult").validate()))
# ValidationReport(ok)
```

Named formats save you writing the pattern that everyone writes slightly wrong. `matches_format` understands `email`, `url`, `uuid`, and `ipv4`, and reports under a name that says which one failed.

```python
contacts = bt.from_pydict({"email": ["a+tag@sub.example.com", "not-an-address", None]})
print(str(contacts.dq.matches_format("email", "email").validate()))
# ValidationReport(violations: is_email(email)=1)
```

Two columns of the same row compare with `compare_columns`, which is NULL-safe on both sides.

```python
spans = bt.from_pydict({"start": [1, 5, None], "end": [3, 2, 9]})
print(spans.dq.compare_columns("start", "<=", "end").drop().to_pydict())
# {'start': [1, None], 'end': [3, 9]}
```

## The validation report

`validate` runs the checks and returns a {py:class}`ValidationReport <batcher.api.dataset.dq.ValidationReport>` without raising. It holds one
{py:class}`ConstraintResult <batcher.api.dataset.dq.ConstraintResult>` per constraint, in the order you declared them. Use `.ok` for a
single pass/fail signal, `.violations` for the per-constraint counts, and `.to_dict()`
to hand the whole thing to a log line or a metrics sink.

```python
report = people.dq.not_null("email").in_range("age", 0, 120).validate()
print(report.ok, report.total_violations, report.rows)
# False 2 5

worst = report.result("in_range(age, 0, 120)")
print(worst.violations, round(worst.pass_rate, 2))
# 1 0.8
```

## Drop invalid rows

`drop` returns only the rows that satisfy every constraint. It is the cleansing
path, for when bad rows should go away.

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

Both sides are lazy datasets built on the same input, so consuming both reads that input
twice. When it is expensive to produce, {py:meth}`cache <batcher.Dataset.cache>` it before
the split.

```python
good, bad = people.dq.in_range("age", 0, 120).quarantine()
print(good.sort("id").to_pydict())
# {'id': [1, 2, 3, 5], 'email': ['a@x.io', 'b@x.io', None, 'e@x.io'],
#  'age': [34, 28, 51, 40], 'country': ['US', 'CA', 'US', 'CA']}
print(bad.to_pydict())
# {'id': [4], 'email': ['d@x.io'], 'age': [200], 'country': ['ZZ']}
```

## Annotate instead of splitting

A quarantined row on its own says only that something rejected it. `annotate` keeps every
row and adds a text column naming the constraints it failed, so triage becomes a `group_by`
rather than a re-run of the checks one at a time.

```python
labelled = people.dq.not_null("email").in_range("age", 0, 120).annotate()
print(labelled.select("id", "dq_failed").sort("id").to_pydict())
# {'id': [1, 2, 3, 4, 5], 'dq_failed': ['', '', 'not_null(email)', 'in_range(age, 0, 120)', '']}
```

A clean row carries the empty string, so the rejected set is `dq_failed != ''` and the
failure counts by rule are one aggregation away.

## Fail the pipeline

`fail` is the data-contract gate at a pipeline boundary: it raises
{py:exc}`DataQualityError <batcher.DataQualityError>` (carrying the per-constraint counts) if any constraint is
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

## Tolerance and severity

Real data is never perfectly clean, and a check that fails a run over one row in a million
gets turned off. Two per-constraint modifiers keep it turned on instead.

`mostly` is the fraction of rows that must pass for the constraint to *pass*. It moves the
pass/fail line only. The violating rows are still counted, and `drop` still removes them,
because a tolerated violation is one you chose not to fail the run over, not a row that
became valid.

```python
sample = bt.from_pydict({"x": [1, 2, 3, -4]})
lenient = sample.dq.positive("x", mostly=0.75)
print(lenient.validate().ok, lenient.validate().violations, lenient.drop().count())
# True {'positive(x)': 1} 3
```

`severity="warn"` reports a violation without enforcing it anywhere: it never raises in
`fail`, never removes a row in `drop`, and never lands on the rejected side of a
`quarantine`. That is how a new rule is watched in production before it is switched on.

```python
watched = sample.dq.positive("x", severity="warn")
print(watched.validate().ok, [r.name for r in watched.validate().warnings])
# True ['positive(x)']
print(watched.drop().count())
# 4
```

## Scope a constraint to some rows

Some rules apply to part of a table. `where` scopes every constraint added after it to the
rows matching a predicate; a row outside the scope passes vacuously, so scoped and unscoped
constraints compose in one chain.

```python
addresses = bt.from_pydict(
    {"country": ["US", "FR", "US"], "state": ["CA", None, None]}
)
scoped = addresses.dq.where(bt.col("country") == "US").not_null("state")
print(scoped.validate().violations)
# {'not_null(state)': 1}
```

## Uniqueness and referential integrity

`unique` requires a key (or key combination) to occur at most once; the report counts the
duplicated *rows*, which is the same number `drop` removes.

```python
dupes = bt.from_pydict({"id": [1, 1, 2, 3, 3]})
print(str(dupes.dq.unique("id").validate()))
# ValidationReport(violations: unique(id)=4)
```

Referential integrity comes in two shapes. {py:meth}`references <batcher.api.dataset.dq.DatasetDQ.references>` is a constraint, so orphans
can be counted, dropped, or quarantined alongside every other check in the chain.
{py:meth}`foreign_key <batcher.api.dataset.dq.DatasetDQ.foreign_key>` is a terminal that hands back the orphan rows themselves, which is
what you want when the orphans are the answer. A NULL key is not an orphan in either: it
means "no reference", not "a broken reference".

```python
orders = bt.from_pydict({"order_id": [1, 2, 3], "customer_id": [10, 20, 99]})
customers = bt.from_pydict({"customer_id": [10, 20, 30]})

print(orders.dq.references("customer_id", to=customers).validate().violations)
# {'references(customer_id)': 1}
print(orders.dq.foreign_key("customer_id", references=customers).to_pydict())
# {'order_id': [3], 'customer_id': [99]}
```

## Start from the data, not a blank page

The hardest part of a contract is the first draft. Nobody knows from memory which of two
hundred columns are never null, which are keys, and which are enumerations with nine values,
so the contract that gets written is the one somebody remembered. {py:meth}`suggest <batcher.api.dataset.dq.DatasetDQ.suggest>` reads
the shape of the data and proposes the constraints it already satisfies.

```python
proposed = people.dq.suggest()
print(repr(proposed))
# DatasetDQ(not_null(id), unique(id), positive(id), null_rate_below(email, 0.25),
#           not_null(age), unique(age), positive(age), not_null(country))
print(proposed.validate().ok)
# True
```

It executes, so treat it as a profiling step rather than part of a pipeline. Everything it
proposes is true of *this* data now, which is both the point and the limit: read the chain,
delete what is a coincidence of today's sample, and keep what is a contract. It deliberately
never proposes a range read off an observed minimum and maximum, because tomorrow's
legitimate value is outside today's and a check that cries wolf gets deleted.

## Reuse a contract across datasets

A contract is written once and run against many tables: today's partition and yesterday's,
the staging copy and the production one. `on` rebinds an accumulated chain to another
dataset, so there is no second way to spell the constraints.

```python
contract = people.dq.not_null("email").in_range("age", 0, 120)
other_day = bt.from_pydict({"id": [9], "email": ["z@x.io"], "age": [31], "country": ["US"]})
print(contract.on(other_day).validate().ok)
# True
```

## Deduplication

`distinct` removes duplicate rows. With no argument it deduplicates over all
columns; with a `subset` it keeps one row per key combination. Pass
`keep="first"`/`"last"` with `order_by` to pick which row survives. That is the
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
widen a type. Pass `schema_mode="union"` to a read and the files reconcile into one
schema: the union of columns, each promoted to a common type, missing columns filled
as null. Use `"latest"` to let the newest file's schema win. `"strict"`, the default,
requires every file to match.

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

## See also

- {doc}`Data contracts </user-guide/trust/data-contracts>`: the checks a whole table fails, not a row — row counts, distributions, freshness, and schema.
- {doc}`Reading data </user-guide/moving-data/reading-data>`: ingest files, and reconcile the schemas that
  drifted between them.
- {doc}`Transformations </user-guide/transform/rows/transformations>`: cleanse and reshape what survived the checks.
- {doc}`Aggregations </user-guide/analyze/aggregations>`: summarize it.
- {doc}`Dataset API </api/relational/dataset>`: the full reference for `ds.dq` and `distinct`.
- {doc}`Agent skills </agents>`: `validate-data-quality`, on choosing between
  fail, drop, and quarantine, and profiling before you write the checks.
- {doc}`/cookbook/dataset/cleaning/dq_contracts`: validate, fail, drop, or quarantine, as a runnable script.
