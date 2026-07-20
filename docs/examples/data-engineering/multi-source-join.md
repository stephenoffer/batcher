# Multi-source join

The facts are in Parquet, in your lake. The customer dimension is a CSV a partner drops
on SFTP. The FX rates come from a service. Nobody owns all three, and the join that
stitches them together is where the revenue number quietly stops being the revenue number.

```python
import datetime
import os
import tempfile

import batcher as bt

work = tempfile.mkdtemp()
orders_path = os.path.join(work, "orders.parquet")

bt.from_pydict(
    {
        "order_id": [1, 2, 3, 4],
        "customer_id": [10, 20, 30, 99],
        "amount": [50.0, 25.0, 80.0, 15.0],
        "placed_at": [
            datetime.date(2024, 5, 1),
            datetime.date(2024, 5, 2),
            datetime.date(2024, 5, 2),
            datetime.date(2024, 5, 3),
        ],
    }
).write.parquet(orders_path)

# The partner's CSV. Customer 10 appears twice: they re-sent a corrected row and
# left the old one in. Customer 99 is not in the file at all.
customers_csv = os.path.join(work, "customers.csv")
with open(customers_csv, "w") as fh:
    fh.write("customer_id,name,segment\n10,ann,smb\n20,bob,ent\n30,cat,smb\n10,ann,ENT\n")

orders = bt.read.parquet(orders_path)
customers = bt.read.csv(customers_csv)

print(orders.count(), orders.sum("amount"))
# 4 170.0
```

Four orders, $170 of revenue. Remember that number.

## The join that looks fine

```python
enriched = orders.join(customers, on="customer_id")
print(enriched.count(), enriched.sum("amount"))
# 4 205.0
```

:::{warning}
Four rows out, four rows in. A row-count check passes. And revenue is now $205.
:::

Two bugs cancelled each other out on the count and compounded on the money. Order 4
(customer 99) has no matching customer, so the inner join dropped it: minus $15. Customer
10 appears twice in the CSV, so order 1 matched twice and its $50 was counted twice: plus
$50. Net: same four rows, $35 of fiction.

| Failure | What it does to the facts | The check | The fix |
|---|---|---|---|
| Orphan key | the fact row is dropped | an anti-join returns the orphans | `how="left"`, or fail the run |
| Fan-out | the fact row is multiplied | `dq.unique` on the dimension key | collapse the dimension to one row per key |

This is the failure mode of every multi-source join, and it does not announce itself.
Check both sides before you trust it.

## Orphans: the keys with no home

An anti-join returns exactly the left rows that found no match:

```python
orphans = orders.join(customers, on="customer_id", how="anti")
print(orphans.to_pydict())
# {'order_id': [4], 'customer_id': [99], 'amount': [15.0],
#  'placed_at': [datetime.date(2024, 5, 3)]}
```

:::{important}
Order 4 is real. It has money attached. It is not in the report, because an inner join is
a filter and nobody told you it filtered. The row is gone from the output and nothing in
the run records that it existed.
:::

You have two honest options, and what you must not do is inner join and move on.

::::{tab-set}

:::{tab-item} Fail the run

An order for an unknown customer means the dimension load is behind and you should wait
for it.

```python
# docs: skip
if orders.join(customers, on="customer_id", how="anti").count():
    raise RuntimeError("orders reference customers the dimension does not have")
```
:::

:::{tab-item} Keep the fact

`how="left"` keeps every fact row and lets the dimension columns be NULL, so the money
stays in the total and a human can see what is unattributed.

```python
# docs: skip
enriched = orders.join(customers, on="customer_id", how="left")
```
:::

::::

## Fan-out: the key that matches twice

The dimension side must be unique on the join key. If it is not, every fact row matching a
duplicated key is *multiplied*, and every additive measure downstream is inflated.

Check it directly:

```python
print(str(customers.dq.unique("customer_id").validate()))
# ValidationReport(violations: unique(customer_id)=1)
```

One duplicated key. Collapse the dimension to one row per key before joining, and be
deliberate about which row survives:

```python
one_per_customer = customers.distinct(subset=["customer_id"], keep="last", order_by="segment")
```

Then join, keeping every fact:

```python
final = orders.join(one_per_customer, on="customer_id", how="left")
print(final.count(), final.sum("amount"))
# 4 170.0
print(final.sort("order_id").select("order_id", "customer_id", "name", "amount").to_pydict())
# {'order_id': [1, 2, 3, 4], 'customer_id': [10, 20, 30, 99],
#  'name': ['ann', 'bob', 'cat', None], 'amount': [50.0, 25.0, 80.0, 15.0]}
```

Four rows, $170, and order 4 is in the output with a NULL name that a human can act on.

:::{tip}
The rule to carry: a fact-to-dimension join must not change the fact row count. Assert it
if you like. It is one `count()` on each side and it catches both bugs above.
:::

## Key types

Two datasets from two systems will eventually disagree about whether an id is a number or
a string. Batcher does not paper over it:

```python
from_api = bt.from_pydict({"customer_id": ["10", "20"], "region": ["us", "eu"]})

try:
    orders.join(from_api, on="customer_id")
except Exception as err:
    print(type(err).__name__)
# PlanError
```

Note the failure lands on `join` itself, not on a later `count()`: the key types are
checked when the plan is built, so you hear about it before any data is read.

A loud failure, which is the correct one. A silent cast would either match nothing (and
you would ship a report full of NULLs) or match by string coercion and give `010` and `10`
different fates. Fix it at the boundary with an explicit `cast`:

```python
aligned = orders.cast({"customer_id": "string"}).join(from_api, on="customer_id")
print(aligned.select("order_id", "region").sort("order_id").to_pydict())
# {'order_id': [1, 2], 'region': ['us', 'eu']}
```

Decide which side is the canonical type and cast the other one, once, where the data
enters. Do not scatter casts through the pipeline.

## When the key is "nearest", not "equal"

The FX rate table has a row when the rate changed, not a row per day. An equi-join on the
date matches nothing. `join_asof` matches each fact to the nearest earlier rate:

```python
fx = bt.from_pydict(
    {
        "as_of": [datetime.date(2024, 4, 30), datetime.date(2024, 5, 2)],
        "usd_per_eur": [1.05, 1.10],
    }
)
priced = orders.sort("placed_at").join_asof(fx, left_on="placed_at", right_on="as_of")
print(priced.select("order_id", "placed_at", "usd_per_eur").to_pydict())
# {'order_id': [1, 2, 3, 4],
#  'placed_at': [datetime.date(2024, 5, 1), datetime.date(2024, 5, 2),
#                datetime.date(2024, 5, 2), datetime.date(2024, 5, 3)],
#  'usd_per_eur': [1.05, 1.1, 1.1, 1.1]}
```

May 1 gets the April 30 rate, May 2 and 3 get the May 2 rate. Both sides must be sorted on
the match key (hence the `sort`), and `by=` adds an exact-match grouping column when the
rate is per currency rather than global.

## What the engine does with it

You do not pick the build side, and you should not want to. The optimizer estimates the
cardinality of each input and builds the hash table on the smaller one.

:::{dropdown} Checking which join the planner picked

```python
print("hash_join" in orders.join(one_per_customer, on="customer_id").explain())
# True
```

`explain()` prints the plan with the row estimates it used, and `explain(analyze=True)` runs it
and prints what actually happened. When a join is slow, that gap is the first place to look:
an estimate that was wrong by 100x usually means a stale or missing statistic, and the
adaptive layer will correct it mid-query at the pipeline breaker, but a plan that started
from a bad guess still paid for the start.
:::

## See also

- [Quality gates](quality-gates.md): `foreign_key` as a pre-join contract.
- [Slowly changing dimensions](slowly-changing-dimensions.md): joining to the version of
  the dimension that was current at the time.
- [Schema evolution](schema-evolution.md): where the key's type quietly changed.
- [Joins](../../user-guide/joins.md): every join type and their semantics.
- [Custom connectors](../../user-guide/custom-connectors.md): reading the source that has
  no reader yet.
- [Explain plans](../../user-guide/explain-plans.md): reading what the optimizer decided.
- [Join algorithms](../../deep-dives/join-algorithms.md): how the build side is chosen.
- [Dataset API](../../api/dataset.md): `join`, `join_asof`, `cast`, `distinct`.
