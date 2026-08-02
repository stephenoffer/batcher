# Quality gates

A partner changed their export format. Amounts arrive negative, a currency code is
garbage, one row has no amount at all.

:::{warning}
Your loader does not care: it is a `read` and a `write`, and both succeed. The bad rows
land in the warehouse, the finance dashboard reads them, and you hear about it nine days
later from someone who is not in a good mood.
:::

The load was never the problem. The absence of a gate was.

```python
import batcher as bt

batch = bt.from_pydict(
    {
        "order_id": [1, 2, 3, 4, 5, 6],
        "customer_id": [10, 20, 10, 99, 30, 20],
        "amount": [19.99, -5.0, 42.0, 7.5, None, 12.0],
        "currency": ["USD", "USD", "EUR", "XXX", "USD", "USD"],
    }
)
customers = bt.from_pydict({"customer_id": [10, 20, 30], "name": ["ann", "bob", "cat"]})
```

## Say what a good row is

A constraint is a boolean expression that is TRUE for a valid row. `ds.dq` accumulates
them and returns a new accessor each time, so they chain, and the whole chain lowers to
the same relational operators as everything else. There is no separate validation engine
scanning your data a second time.

```python
checks = (
    batch.dq.not_null("amount")
    .in_range("amount", 0.0, 10_000.0)
    .accepted_values("currency", ["USD", "EUR", "GBP"])
    .unique("order_id")
)
report = checks.validate()

print(str(report))
# ValidationReport(violations: not_null(amount)=1, in_range(amount, 0.0, 10000.0)=1,
#  accepted_values(currency)=1)
print(report.ok, report.total_violations)
# False 3
```

`validate()` counts violations per constraint and does not raise. `unique("order_id")`
passed, so it is not in the report.

:::{note}
The value constraints treat NULL as valid: `in_range` does not fire on the NULL amount,
`not_null` does. That is deliberate, so the checks compose independently instead of
double-counting one bad row. State the null rule explicitly when you mean it.
:::

## Three ways to spend a report

A report ends in one of three verdicts, and the choice decides what happens to the rows
that failed:

| Verdict | What happens to a bad row | What it is for |
|---|---|---|
| `fail()` | nothing loads; `DataQualityError` carries the counts | a data contract at a pipeline boundary |
| `drop()` | discarded, with no record it was ever there | a bad row that is genuinely noise |
| `quarantine()` | routed to a second dataset, whole | a partner feed you have to keep loading |

::::{tab-set}

:::{tab-item} Stop the load

`fail()` raises `DataQualityError` (carrying the counts) if anything is violated, and
otherwise returns the dataset so the chain continues.

```python
from batcher._internal.errors import DataQualityError

try:
    batch.dq.not_null("amount").fail()
except DataQualityError as err:
    print(err)
# data-quality check failed: ValidationReport(violations: not_null(amount)=1)
```
:::

:::{tab-item} Drop the bad rows

`drop()` returns only rows satisfying every constraint.

```python
# docs: skip
loadable = checks.drop()
```

Use it when a bad row is genuinely noise and nobody will ever ask about it. That is rarer
than people think.
:::

:::{tab-item} Quarantine them

`quarantine()` splits the dataset in two. The split is total: every input row lands on
exactly one side, so nothing evaporates.

```python
clean, rejected = checks.quarantine()

print(clean.count())
# 3
print(rejected.sort("order_id").to_pydict())
# {'order_id': [2, 4, 5], 'customer_id': [20, 99, 30], 'amount': [-5.0, 7.5, None],
#  'currency': ['USD', 'XXX', 'USD']}
```

Quarantine is the default answer for a partner feed. The good rows load on time, the bad
rows go to a dead-letter table with the rest of the row intact, and someone can look at
them on Monday instead of at 2am.
:::

::::

:::{important}
`drop()` is the only one of the three that loses data. The rows it removes leave no trace,
so the question "why is the total short" has no answer anywhere in the pipeline. Prefer
`quarantine()` unless you can say out loud that nobody will ever ask.
:::

## A gate needs a threshold

Quarantining three rows out of a million is Tuesday. Quarantining half the batch means
the partner broke their export and you should not load *any* of it, because a
half-loaded day is worse than a missing one: it looks complete.

So make the decision on the rate, not on the presence of a violation:

```python
reject_rate = rejected.count() / batch.count()
print(reject_rate)
# 0.5

if reject_rate > 0.01:
    print("halt: batch is broken, not dirty")
else:
    print(f"load {clean.count()} rows, quarantine {rejected.count()}")
# halt: batch is broken, not dirty
```

:::{tip}
Pick the threshold from what the feed actually does on a normal day, and alert on the rate
itself, not only on the breach. A feed whose reject rate walks from 0.1% to 0.9% over a
month is telling you something before it trips the gate.
:::

## Referential integrity

`order_id=4` references customer 99, who does not exist. No constraint above catches it,
because the answer is not in this dataset. `foreign_key` joins against the reference and
returns the orphans, so an empty result means every key resolves:

```python
orphans = batch.dq.foreign_key("customer_id", references=customers)
print(orphans.to_pydict())
# {'order_id': [4], 'customer_id': [99], 'amount': [7.5], 'currency': ['XXX']}
```

Run this *before* the join that consumes the key. An inner join with an orphan key drops
the row and reports nothing, so you find out from a row count that does not tie out. See
{doc}`multi-source join </cookbook/data-engineering/modeling/multi-source-join>`, where that is the whole story.

## Where the gate goes

Before the write, not after. A gate downstream of the load is a report. A gate upstream
of it is a gate.

```python
# docs: skip
clean, rejected = (
    bt.read.parquet("s3://landing/orders/2024-01-02/")
    .dq.not_null("amount")
    .in_range("amount", 0.0, 10_000.0)
    .accepted_values("currency", ["USD", "EUR", "GBP"])
    .quarantine()
)
rejected.write.parquet("s3://lake/_rejects/orders/2024-01-02/")
clean.write.delta("s3://lake/orders", merge_on="order_id")
```

Two things this does not do. It will not catch a plausible-but-wrong value: an amount of
`42.0` that should have been `4.20` passes every constraint you can write. And the checks
cost a pass over the data. They are ordinary relational operators, so they fuse and run in
Rust rather than in a Python loop, but "cheap" is not "free". Put the expensive ones
(`unique`, `foreign_key`, which both hash) on the path where they earn it, and the scalar
predicates everywhere.

## See also

- {doc}`Deduplication </cookbook/data-engineering/maintenance/deduplication>`: what `unique` found, and what to do about it.
- {doc}`Schema evolution </cookbook/data-engineering/modeling/schema-evolution>`: gating the shape rather than the values.
- {doc}`Multi-source join </cookbook/data-engineering/modeling/multi-source-join>`: the orphan key, and what an inner join does
  with it.
- {doc}`Data quality </user-guide/trust/data-quality>`: every constraint in the accessor.
- {doc}`Writing data </user-guide/moving-data/writing-data>`: where to land the quarantined rows.
- {doc}`Delta Lake </integrations/lakehouse/delta-lake>`: the target the gate stands in front of.
- {doc}`Dataset API </api/relational/dataset>`: `ds.dq`, `validate`, `fail`, `drop`, `quarantine`.
- {doc}`Exceptions </api/operations/exceptions>`: `DataQualityError` and what it carries.
