# Metadata shortcuts

Most of what people ask a dataset, the dataset already knows.

A Parquet footer records every column's minimum, maximum, and null count. An ORC stripe
header records its row count. A lakehouse manifest records both, per file. A warehouse
catalog records them per table. And an in-memory relation, being immutable, can compute
them once and remember them forever. So when you ask "how many rows is that?", or "does
this column have gaps?", or "is `id` actually unique?", there is very often nothing to
compute, only something to read.

**You do not have to ask for this.** It happens underneath the API you already use:

```python
# docs: skip
import batcher as bt

ds = bt.read.parquet("s3://warehouse/events/")      # 10 billion rows

ds.count()                                          # a footer read
ds.min("amount"), ds.max("amount")                  # a footer read
ds.null_count()                                     # a footer read
ds.filter(bt.col("amount") > 10**9).collect()       # provably empty — the files go unread
ds.join(dim, on="region_id").collect()              # key ranges disjoint? no shuffle at all
ds.dq.not_null("id").in_range("amount", 0, 1e6).fail()   # a contract the footer discharges
```

Each of those is an ordinary call. Each one, on this data, costs a metadata round trip
instead of a scan. Nothing in that snippet mentions metadata, and that is the point.

The rest of this page explains **when** it fires, so you can tell why something was slow.
It then covers `ds.meta`, an *optional* introspection namespace for asking the metadata
layer directly. You will rarely need it. Reach for it when you want to know what the engine
knows, or to ask something the ordinary API has no spelling for, such as "would this join
match anything?" or "how many files am I about to open?".

## The one rule that makes it safe

**A shortcut returns exactly what executing would return.** Not an estimate of it, not a
usually-right version of it. The same value.

That is not a hope, it is how the layer is built. Kyber only answers from a statistic whose
provenance is *exact*, meaning a footer bound, a manifest count, or an immutable relation's
own measurement. If the statistic it needs is missing or merely estimated, it declines, and
`ds.meta` quietly runs the query that computes the answer instead. Which of the two happened
is invisible to you, because the answers are identical. Only the cost moves.

So you never have to decide whether a shortcut is "safe here". It degrades to the query you
would have written anyway.

The one exception is deliberately named: everything under `ds.meta.approx` is **approximate
and never executes**. It answers from a sketch or returns `None`. More on that below.

## What the ordinary API gets for free

These are the calls people already write. Nothing here needs `ds.meta`.

| you write | what it costs, when the metadata is there |
|---|---|
| `ds.count()`, `len(ds)`, `ds.is_empty()`, `ds.has_rows` | a recorded row count |
| `ds.min(c)`, `ds.max(c)` | a footer bound |
| `ds.n_null(c)`, `ds.has_nulls(c)`, `ds.all_null(c)`, `ds.null_count()` | a footer null count |
| `ds.filter(p)` where `p` is refuted by the bounds | the files go unread |
| `ds.filter(p).count()` where `p` is `IS NULL` / out of range | the null count, or zero |
| `ds.drop_nulls(c)` / `ds.fill_null(...)` on a column with no nulls | a no-op |
| `ds.limit(n)` with `n` at or above the row count | a no-op |
| `ds.join(other, on=k)` whose key ranges are **disjoint** | no build, no probe, no shuffle |
| `ds.dq.not_null(...).fail()`, `.drop()`, or `.validate()` on a contract that holds | three numbers |

The last two are the ones that change what a query costs rather than shaving it. A join whose
key ranges cannot overlap emits nothing, which is provable from four numbers with neither
side read. And a data-quality contract exists precisely to *confirm* that data is fine,
which is the answer a footer usually already contains.

## Introspection: `ds.meta`

Everything from here on is the **optional** namespace. You do not need it for any of the speed
above. It exists to ask the metadata layer directly: what does the engine know, why wasn't
that free, and the handful of questions the ordinary API has no spelling for.

The examples run against this dataset:

```python
import datetime as dt

import batcher as bt

ds = bt.from_pydict(
    {
        "id": [1, 2, 3, 4],
        "user_id": [10, 10, 11, 12],
        "amount": [10.5, 25.0, 3.25, 99.0],
        "day": [dt.date(2024, 1, d) for d in (1, 1, 2, 3)],
        "country": ["US", "US", "DE", "FR"],
        "status": ["ok", "ok", "ok", "ok"],
        "flag": [True, False, True, True],
        "tags": [["a"], ["b"], ["a", "c"], []],
    }
)
```

### What it costs, and when it doesn't help

A filter is what usually takes the shortcut away. A footer's minimum is the smallest value
*in the file*; once you filter the file, it is only a bound on the smallest surviving value,
and a bound is not an answer. Joins, computed columns, and `map_batches` do the same.

`ds.meta.explain()` tells you exactly what is known, and is the fastest way to see why
something fell back to a scan. It reports the exact row count (or `None`), the estimate, the
recorded ordering, and per column every facet provable without a scan.

```python
report = ds.meta.explain()
assert report["rows"] == 4
assert report["columns"]["amount"]["min"] == 3.25
```

## Rows, and whole-relation questions

`shape` gives you `(rows, columns)`. `count_where` counts a filter's survivors, often for
free, since `col IS NULL` is a recorded null count and a comparison above the recorded
maximum is provably zero. `is_empty_where`, `any_match`, `none_match`, and `all_match` are
the boolean forms, and `none_match` is the one a pruning decision reads best as.

```python
assert ds.meta.shape() == (4, 8)
assert ds.meta.count_where(bt.col("amount").is_null()) == 0
assert ds.meta.none_match(bt.col("amount") > 1_000_000)  # provably no such row — no scan
assert ds.meta.all_match(bt.col("status") == "ok")
assert ds.meta.any_match(bt.col("amount") > 50)
assert ds.meta.is_empty_where(bt.col("amount") > 1_000_000)
```

`is_key` checks a candidate primary key (unique *and* never null), for one column or a
composite. `sorted_by` reports the ordering the data is already known to carry, and
`is_known_sorted_by` tells you whether a sort would be a no-op. Both are one-sided: they
report what is *recorded*, never guessing that unrecorded means unsorted.

```python
assert ds.meta.is_key("id")
assert not ds.meta.is_key("user_id")
assert ds.meta.is_key(["day", "id"])
assert ds.meta.sorted_by() == ()
assert ds.meta.is_known_sorted_by("day") is False
```

## One column at a time with `ds.meta.col(...)`

`ds.meta.col(name)` narrows the namespace to a single column. Every method below is
answered from a recorded statistic when there is one, and from a query when there is not.

```python
c = ds.meta.col("amount")

assert c.bounds() == (3.25, 99.0)   # (min, max) — one footer read, not two passes
assert c.range() == 95.75           # max - min
assert c.midpoint() == 51.125       # the centre of the range (not the mean, not the median)
assert c.abs_max() == 99.0          # max(|min|, |max|) — does this fit in an int32?
assert c.n_unique() == 4            # exact COUNT(DISTINCT)
assert c.is_unique()                # every non-null value occurs once?
assert not c.has_duplicates()
assert c.duplicate_count() == 0     # how many rows a DISTINCT would remove
assert c.is_key()                   # unique and never null
assert not c.is_constant()          # min == max would mean one value, and no other
assert c.constant_value() is None
assert c.is_low_cardinality(128)    # dictionary-encode it? one-hot it?
assert not c.is_binary_valued()     # a flag, a label, a mask
assert c.null_fraction() == 0.0
assert c.no_nulls()
assert c.sum() == 137.75            # from a recorded total, when the source has one
assert c.mean() == 34.4375
assert c.summary()["n_unique"] == 4  # all of the above, as one dict
```

`sum` and `mean` are the ones with an interesting economics. No footer records a sum, so
they usually run an aggregate. But an immutable in-memory relation *computes and caches* one
the first time you ask, so the second query that needs it is free. That is the
learned-metadata idea in miniature: a query that gets cheaper the more it runs.

## Predicates on a column with `ds.meta.col(...).check`

A minimum and a maximum are not only statistics. They are *values that occur in the column*,
and that turns a whole class of questions into arithmetic on two numbers.

```python
amt = ds.meta.col("amount").check

assert amt.all_positive()          # decided by the minimum, alone
assert amt.all_non_negative()
assert not amt.all_negative()
assert not amt.all_non_positive()
assert not amt.all_zero()
assert amt.all_between(0, 1000)    # the range check a quality gate runs on every row
assert amt.all_greater_than(0)
assert amt.all_greater_equal(3.25)
assert amt.all_less_than(1000)
assert amt.all_less_equal(99.0)

assert not amt.any_greater_than(1_000_000)  # the maximum decides — in *both* directions
assert amt.any_greater_equal(99.0)
assert not amt.any_less_than(0)
assert not amt.any_less_equal(0)
```

`any_greater_than` is worth dwelling on. A maximum *above* the threshold proves a match
exists, and a maximum *at or below* it proves none does. The second half is what lets
`WHERE amount > 1000000` over a column whose maximum is 99 be answered "no rows" without
opening the file.

Membership is the other half, and it is **asymmetric** on purpose:

```python
ids = ds.meta.col("id").check

assert not ids.contains(9999)     # absence is provable; presence usually is not
assert ids.never_equals(9999)     # the spelling a skip decision reads as
assert not ids.may_contain(9999)  # free, one-sided: False is a proof of absence
assert ids.contains(3)
assert ids.any_in([3, 4])         # SQL IN — refuted for free when every candidate is out of range
assert ids.none_in([9998, 9999])
```

A value outside `[min, max]`, or one a membership bloom rejects, is *not in the column*, and
cannot be in any subset of it. That refutation is what skips a file, a partition, or a whole
query. Presence is the other direction, and bounds cannot confirm it unless the column is
constant, so a "maybe" runs the filter. `may_contain` never executes at all, and a `False`
from it is always safe to act on.

## Missing data with `ds.meta.nulls`

The `nulls` namespace answers whole-relation completeness questions across every column at
once.

```python
assert ds.meta.nulls.counts()["amount"] == 0  # every column, one question
assert ds.meta.nulls.fractions()["amount"] == 0.0
assert ds.meta.nulls.total() == 0
assert not ds.meta.nulls.any()
assert ds.meta.nulls.is_complete()  # no null anywhere — the data-contract question
assert ds.meta.nulls.columns_with_nulls() == []
assert "amount" in ds.meta.nulls.complete_columns()
```

When the footers cannot answer, this runs **one** aggregate covering every column, never
one pass per column.

## Types with `ds.meta.schema`

The cheapest shortcuts here: the plan knows its own output schema, so these never touch data
and can never be wrong.

```python
schema = ds.meta.schema

assert schema.num_columns() == 8
assert schema.has("amount")
assert schema.index("id") == 0
assert schema.dtype("id") is not None

assert schema.is_numeric("amount")
assert schema.is_integer("id")
assert schema.is_float("amount")
assert schema.is_string("country")
assert schema.is_boolean("flag")
assert schema.is_temporal("day")
assert schema.is_nested("tags")

assert schema.numeric() == ["id", "user_id", "amount"]
assert schema.strings() == ["country", "status"]
assert schema.booleans() == ["flag"]
assert schema.temporal() == ["day"]
assert schema.nested() == ["tags"]
assert schema.select("numeric").columns == ["id", "user_id", "amount"]
```

## Physical layout with `ds.meta.storage`

What a scan *would* read, before it reads it. "340 files, 12 GB, partitioned by day" is a
sentence you can act on, and it costs one metadata round trip to say.

```python
storage = ds.meta.storage

assert storage.num_sources() == 1
assert storage.row_count() == 4  # rows the sources *hold*, not the query's result count
assert storage.has_exact_row_count()
assert storage.num_files() == 0  # an in-memory relation has no files
assert storage.files() == []
assert storage.partition_keys() == ()
assert not storage.is_partitioned()
assert storage.sorted_by() == ()
assert storage.total_bytes() is None  # a Parquet source reports its footer's byte size,
assert storage.row_group_count() is None  # ...and its row-group count
assert storage.bytes_per_row() is None
```

On a Parquet source, `num_files` is the small-files diagnosis without a scan. A thousand
files for a gigabyte means the query is about to spend its time on footers rather than on
data. `row_group_count` is the granularity a zone-map prune actually skips at.

## Joins with `ds.meta.against(other)`

The shortcut that saves the most work in absolute terms. If one side's key range is `[1, 10]`
and the other's is `[900, 999]`, the inner join is **empty**, provably, from four numbers,
with neither side read. No build, no probe, no shuffle.

```python
absent = bt.from_pydict({"user_id": [900, 901]})
present = bt.from_pydict({"user_id": [10, 11]})

assert ds.meta.against(absent).join_is_empty("user_id")  # disjoint ranges → no shuffle at all
assert not ds.meta.against(absent).overlaps("user_id")
assert ds.meta.against(present).overlaps("user_id")
assert ds.meta.against(present).key_overlap("user_id") == (10, 11)
assert ds.meta.against(present).estimated_rows("user_id") >= 0
```

Only *emptiness* is proved. Overlapping ranges do not imply a match exists, because two key
columns can share a range and share no value, so an overlap runs the join.

## Approximate answers with `ds.meta.approx`

Different contract, and it is named so it cannot be confused with the rest. **Nothing here
executes, and nothing here is exact.** Each method reads a sketch a previous run recorded, and
returns `None` if nobody has measured it yet.

```python
approx = ds.meta.approx

assert approx.rows() == 4.0        # the cost model's estimate — always available
assert approx.memory_bytes() > 0   # size a buffer, a broadcast, a spill threshold
assert approx.row_bytes() > 0
assert approx.column_bytes("amount") == 32.0
assert 0.0 <= approx.selectivity(bt.col("amount") > 20) <= 1.0
assert approx.count_where(bt.col("amount") > 20) >= 0
assert isinstance(approx.is_measured("amount"), bool)  # why is the rest returning None?

# These read sketches a *previous* run recorded, so they are None until the query has run.
approx.n_unique("user_id")  # from an HLL sketch, or None
approx.cardinality_ratio("user_id")  # key-like (→ 1) or categorical (→ 0)?
approx.top_k("country", 5)  # a skewed join key, with no GROUP BY
approx.frequency("country", "US")
approx.histogram("amount", 4)  # equal-probability buckets from a KLL grid
```

`is_measured` is the introspection that explains the rest. These sketches are written by the
executor when a query runs, so a column nobody has read has nothing measured. Run the query
once and the second run answers for free.

Read it as the coarse question "is anything recorded for this column", because a distinct
count, a quantile grid, a top-values map, and a plain column width all count. An in-memory
source already knows a column's width, so `is_measured` reads `True` there while the
distinct-count sketch is still absent. Test the value you are about to use rather than the
column as a whole.

If you need an approximate quantile *now*, use the `Dataset` terminals `ds.approx_median`,
`ds.approx_quantile`, and `ds.approx_n_unique`. They consult the same learned sketches first
and then stream one if there is none. `ds.meta.approx` is the free-or-nothing probe.

## Where this comes from

`ds.meta` is the user-facing half of Batcher's metadata-first design. The other half is
invisible: `ds.count()`, `ds.min()`, `ds.n_unique()`, and the optimizer's own pruning all
consult the same statistics before executing. The `meta` namespace opens that layer up and
lets you ask it directly. Crucially, it also lets you ask things no SQL terminal has a
spelling for, such as "would this join match anything?" or "how many files am I about to
open?".

Two deep dives explain the machinery underneath. {doc}`../deep-dives/learned-metadata`
covers the sketches a finished query records for the next one, and
{doc}`../deep-dives/cardinality-estimation` covers how the optimizer turns them into row
estimates. The measured payoff is on {doc}`../benchmarks/analytics`, where a `count()`
after a transform chain returns in 0.05 ms because nothing is scanned.

## See also

- {doc}`explain-plans`: confirm that a predicate reached the scan, and that pruning
  actually happened.
- {doc}`data-quality`: turn the `ds.meta.col(...).check` probes here into enforced
  contracts with `ds.dq`.
- {doc}`performance`: the other levers when a query is slower than its metadata suggests
  it should be.
- {doc}`reading-data`: where footer statistics come from, per format.
- {doc}`lakehouse`: manifest-level statistics on Delta, Iceberg, and Hudi tables.
- {doc}`../api/dataset`: the `ds.meta` reference, namespace by namespace.
- {doc}`../internals/execution`: the exact-or-fall-back rule that decides when a
  statistic is allowed to answer a terminal.
