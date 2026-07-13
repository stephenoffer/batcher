# Slowly changing dimensions

Customer 1 moved from New York to San Francisco in April. Your nightly dimension load
does what dimension loads do: it upserts the new city over the old one.

:::{warning}
In May, someone reruns the Q1 revenue-by-city report. New York is down $50 and San
Francisco is up $50, and no transaction in Q1 changed. The report is not wrong about
today; it is wrong about February, because the dimension no longer remembers February.
:::

That is the entire case for SCD type 2, and it is the reason "just upsert the dimension"
is a decision, not a default.

| Type | The table keeps | You can answer | You lose |
|---|---|---|---|
| type 1 | one row per key, current values only | "where do they live now" | every past value, permanently |
| type 2 | one row per key *per version*, with validity intervals | "where did they live in February" | nothing — at the cost of a growing table |
| type 3 | one row per key, plus a `<attr>_prev` column | "did they move recently" | everything before the previous value |

## Type 2: versions, not values

`ds.scd.type2` maintains the dimension from an incoming snapshot. You give it the natural
key, the attributes whose change starts a new version, and the effective date of the
batch:

```python
import os
import tempfile

import batcher as bt

work = tempfile.mkdtemp()
dim = os.path.join(work, "customer_dim.parquet")

bt.from_pydict(
    {"customer_id": [1, 2], "city": ["NYC", "LA"], "tier": ["gold", "free"]}
).scd.type2(dim, keys="customer_id", track=["city", "tier"], as_of="2024-01-01")
```

April's snapshot: customer 1 moved, customer 2 did not, customer 3 is new.

```python
bt.from_pydict(
    {
        "customer_id": [1, 2, 3],
        "city": ["SF", "LA", "BER"],
        "tier": ["gold", "free", "free"],
    }
).scd.type2(dim, keys="customer_id", track=["city", "tier"], as_of="2024-04-01")

history = bt.read.parquet(dim).sort("customer_id", "valid_from")
print(history.to_pydict())
# {'customer_id': [1, 1, 2, 3], 'city': ['NYC', 'SF', 'LA', 'BER'],
#  'tier': ['gold', 'gold', 'free', 'free'],
#  'valid_from': ['2024-01-01', '2024-04-01', '2024-01-01', '2024-04-01'],
#  'valid_to': ['2024-04-01', None, None, None],
#  'is_current': [False, True, True, True]}
```

Customer 1 now has two rows. The January version was expired (`valid_to = 2024-04-01`,
`is_current = False`) and the April version opened. Customer 2 changed nothing, so
nothing happened to their row: no new version, no churn, no rewrite of a row that is
still true. Customer 3 was inserted with a first version.

The load is a no-op when nothing changed, which means running it twice with the same
snapshot and the same `as_of` costs you a rewrite and produces the same table.

## Reading it back

The current dimension is one filter:

```python
print(history.filter(bt.col("is_current")).select("customer_id", "city").to_pydict())
# {'customer_id': [1, 2, 3], 'city': ['SF', 'LA', 'BER']}
```

The dimension *as it was* on a given date is another. A version covers a date when it
started on or before it and had not yet ended (an open version has `valid_to = NULL`):

```python
as_of = "2024-02-15"
snapshot = history.filter(
    (bt.col("valid_from") <= as_of)
    & (bt.col("valid_to").is_null() | (bt.col("valid_to") > as_of))
)
print(snapshot.select("customer_id", "city").sort("customer_id").to_pydict())
# {'customer_id': [1, 2], 'city': ['NYC', 'LA']}
```

In February, customer 1 was in NYC and customer 3 did not exist yet. That is the answer
the Q1 report needed, and a type-1 dimension cannot produce it at any price.

## Joining a fact to the version that was current

This is what the history is *for*. Join on the natural key, then keep the version whose
validity interval contains the fact's date:

```python
orders = bt.from_pydict(
    {
        "order_id": [10, 11],
        "customer_id": [1, 1],
        "order_date": ["2024-02-01", "2024-05-01"],
        "revenue": [50.0, 70.0],
    }
)

attributed = orders.join(history, on="customer_id").filter(
    (bt.col("valid_from") <= bt.col("order_date"))
    & (bt.col("valid_to").is_null() | (bt.col("valid_to") > bt.col("order_date")))
)
print(attributed.select("order_id", "order_date", "city", "revenue").sort("order_id").to_pydict())
# {'order_id': [10, 11], 'order_date': ['2024-02-01', '2024-05-01'],
#  'city': ['NYC', 'SF'], 'revenue': [50.0, 70.0]}
```

February's order is credited to NYC and May's to SF, from one customer. Aggregate and the
history holds:

```python
print(attributed.group_by("city").agg(rev=bt.col("revenue").sum()).sort("city").to_pydict())
# {'city': ['NYC', 'SF'], 'rev': [50.0, 70.0]}
```

Rerun that in 2026 and you get the same two numbers. That is the property the business
actually asked for when they said "the report keeps changing."

Mechanically this is an equi-join on the key followed by a range filter, not a native
interval join. The optimizer pushes the filter down, but the join still produces one row
per (fact, version) pair before the filter cuts it. For a dimension with a handful of
versions per key that is nothing. For a key with hundreds of versions, reach for
`join_asof` on the effective date instead, which matches each fact to the nearest earlier
version directly.

## Type 1 and type 3

::::{tab-set}

:::{tab-item} Type 1

`type1` is the keyed upsert: overwrite the attributes, keep no history.

```python
# docs: skip
bt.from_pydict({"customer_id": [1], "city": ["SF"]}).scd.type1(dim, keys="customer_id")
```

It is the right choice for an attribute nobody will ever ask a historical question about
(a corrected spelling, an internal flag). Do not reach for it because it is simpler; reach
for it because the history has no value.
:::

:::{tab-item} Type 3

`type3` is the middle ground: keep exactly the previous value of each tracked attribute
in a `<attr>_prev` column.

```python
t3 = os.path.join(work, "city_t3.parquet")
bt.from_pydict({"customer_id": [1], "city": ["NYC"]}).scd.type3(
    t3, keys="customer_id", track=["city"]
)
bt.from_pydict({"customer_id": [1], "city": ["SF"]}).scd.type3(
    t3, keys="customer_id", track=["city"]
)

print(bt.read.parquet(t3).to_pydict())
# {'customer_id': [1], 'city': ['SF'], 'city_prev': ['NYC']}
```

One row per key, "before and after" available, and the row count never grows. It answers
"did they move recently" and nothing else. If a third move happens, NYC is gone for good.
:::

::::

## What to watch

The incoming snapshot must have **one row per natural key**. Two rows for the same key is
a cardinality violation and the merge will tell you so; deduplicate first (see
[deduplication](deduplication.md)).

:::{warning}
`as_of` must move forward across loads. It is stored, compared, and written into the
effective dates. Feed it yesterday's date after loading today's and the intervals overlap,
silently. Every point-in-time query over that key then returns two versions.
:::

The target is copy-on-write and single-writer: each load rewrites the files its keys reach
and atomically replaces them. Fine for a dimension (thousands to millions of rows, loaded
once a day). Not a design for a table with concurrent writers, and not a design for a
type-2 table that keeps every version of a hundred-million-row entity forever. Point it at
Delta if you want a real transaction around the commit.

:::{important}
Type 2 only tracks what you list in `track`. An attribute outside that list is overwritten
in place, type-1 style, in whatever version happens to be current — so its past values are
gone from every historical version too, not just from the current one. Choose the tracked
set deliberately, because it is the one thing here you cannot fix retroactively.
:::

## See also

- [CDC pipeline](cdc-pipeline.md): when the source is a change feed, not a snapshot.
- [Multi-source join](multi-source-join.md): joining facts to dimensions in general.
- [Deduplication](deduplication.md): getting to one row per key before the load.
- [Lakehouse tables](../../user-guide/lakehouse.md): the `ds.scd` reference.
- [Joins](../../user-guide/joins.md): the equi-join plus range filter, and `join_asof`.
- [Delta Lake](../../integrations/delta-lake.md): a real transaction around the commit.
- [Dataset API](../../api/dataset.md): `type1`, `type2`, `type3`, `apply_changes`.
