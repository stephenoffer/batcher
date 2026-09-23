# Distinct and deduplication

This page covers removing duplicates in Batcher, exactly, per key, approximately, and over a stream.

There are three different jobs hiding under the word "dedupe", and picking the wrong one is where the bugs come from. Dropping fully identical rows is {py:meth}`distinct() <batcher.Dataset.distinct>`. Keeping the newest row per key is `distinct(subset, keep="last", order_by=...)`. Collapsing rows that are *nearly* the same text is {py:meth}`ds.ml.drop_near_duplicates <batcher.api.dataset.ml.DatasetML.drop_near_duplicates>`. They have different costs and different failure modes.

A stream adds a fourth, because it cannot remember every key forever. Ask the questions in this order and the first "yes" names the operator:

![Choosing a deduplication operator, as four questions in order. First, if rows are nearly the same rather than identical, such as a re-sent row or a re-headed article, use ml.drop_near_duplicates on the column, which may miss a pair but never drops a row that is not a duplicate. Second, if the input is an unbounded stream whose memory has to stay bounded, use drop_duplicates_within_watermark, which forgets a key once the watermark passes it. Third, if every column defines a duplicate, use distinct(), which is SQL DISTINCT star. Fourth, if it matters which row survives, such as the newest or the first, use distinct with the keys, keep="first" or "last", and order_by. Otherwise use distinct with the keys and the default keep="any", which takes an arbitrary row per key and may pick a different one between runs, so use it only when the other columns follow the key. distinct hashes a float key canonically, so 0.0 and -0.0 are one key and every NaN is one key. Count before you drop: value_counts and is_duplicated mark rows without removing them.](/_static/diagrams/dedup_choice.svg)

## Setup

```python
import batcher as bt

events = bt.from_pydict(
    {
        "user": ["a", "a", "b", "b", "c"],
        "ts": [1, 3, 2, 4, 5],
        "status": ["new", "paid", "new", "paid", "new"],
    }
)
```

## Distinct over every column

With no arguments, `distinct()` is SQL `DISTINCT *`, so rows that agree on every column collapse to one.

```python
dupes = bt.from_pydict({"city": ["nyc", "sf", "nyc", "la"], "n": [1, 2, 1, 3]})
print(dupes.distinct().sort("city").to_pydict())
# {'city': ['la', 'nyc', 'sf'], 'n': [3, 1, 2]}
```

It is a pipeline breaker with a hash table sized by the number of *distinct* rows, not input rows. On a wide table it hashes every column, so if you only care about a few columns, `select` them first and the hash gets much cheaper.

## One row per key

`distinct(subset)` keeps one row per distinct key combination and carries the other columns along. Which row survives is the whole question, and `keep` answers it.

| Argument | What it does | Default |
| --- | --- | --- |
| `subset` | the columns that define the key, while the rest ride along | every column |
| `keep` | which row of a key group survives: `"any"`, `"first"`, `"last"` | `"any"` |
| `order_by` | the order `"first"` and `"last"` are measured in, as a column, a list, or `[(col, descending)]` pairs | none |

::::{tab-set}
:::{tab-item} DataFrame

```python
latest = events.distinct(["user"], keep="last", order_by="ts")
print(latest.sort("user").to_pydict())
# {'user': ['a', 'b', 'c'], 'ts': [3, 4, 5], 'status': ['paid', 'paid', 'new']}
```

:::

:::{tab-item} SQL

```python
print(
    bt.sql(
        """
    SELECT user, ts, status
    FROM (
        SELECT *, row_number() OVER (PARTITION BY user ORDER BY ts DESC) AS rn FROM t
    )
    WHERE rn = 1
    ORDER BY user
    """,
        t=events,
    ).to_pydict()
)
# {'user': ['a', 'b', 'c'], 'ts': [3, 4, 5], 'status': ['paid', 'paid', 'new']}
```

:::
::::

The two lower to the same plan, which is the point. The SQL is a way of saying the same thing, not a different query.

`keep="first"` takes the earliest row in `order_by` order, and `keep="last"` takes the latest. Both *require* `order_by`, because without an order there is no first.

`keep="any"`, the default, skips the ordering and takes an *arbitrary* row. Arbitrary means arbitrary: which row you get may differ between runs, and between a single-node and a distributed run, because the engine keeps whichever row it reached first. That is fine when the non-key columns are functionally dependent on the key, because the rows it is choosing between are then identical. It is wrong when they are not. If the surviving row's other columns matter, pass `order_by` with `"first"` or `"last"` and the choice becomes a property of your data rather than of the schedule.

```python
print(events.distinct(["user"], keep="first", order_by="ts").sort("user").to_pydict())
# {'user': ['a', 'b', 'c'], 'ts': [1, 2, 5], 'status': ['new', 'new', 'new']}
```

`order_by` also takes a list, and `[("ts", True)]` reverses a key, so `keep="first", order_by=[("ts", True)]` is another spelling of "newest wins".

This runs as a single reduction, not as a ranking. The engine hashes each row by its key once and keeps the minimum under the ordering, so nothing is sorted and no rank column is built. It is also *mergeable*, which is what lets it stay bounded under spill and lets a distributed run reduce each partition before the shuffle, sending one row per key instead of every row.

That matters most at scale, because the two shapes grow differently. A ranking sorts, so its cost grows faster than the data. A reduction is a hash pass, so it grows with it. On a key that is 5% distinct, the reduction is 2 to 3x faster than the equivalent window at 2 million rows and 7 to 12x faster at 32 million. The dedup section of [`benchmarks/BENCHMARK_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/BENCHMARK_RESULTS.md) has the measurement and the hardware.

If you want the rank itself rather than one row per key, write the window described in {doc}`window functions </user-guide/analyze/window-functions>`.

## Float keys: NaN and -0.0 collapse

A dedup key is a hash key, and Batcher hashes floats from their *canonicalized* IEEE bits. So `0.0` and `-0.0` are one key, and every NaN is one key, even though `NaN == NaN` is false in an expression.

```python
floats = bt.from_pydict({"x": [0.0, -0.0, float("nan"), float("nan")]})
print(floats.distinct().to_pydict())
# {'x': [0.0, nan]}
```

:::{warning}
That collapsing is the behavior you want, and it is what keeps a float key from splitting one group across two partitions that disagree about the bits of the same value. Grouping, distinct, shuffle, and join all share the one canonical key form, so a distinct is identical single-node and distributed. If you are deduping on a float you computed, such as a ratio or a rounded price, round it explicitly and dedupe on the rounded column, so the key is a value you can reason about instead of an accident of the last arithmetic op.
:::

## Counting duplicates before you drop them

:::{tip}
Look before you delete. Duplicates are often a finding about the upstream system rather than noise, and once they are dropped the finding is gone with them.
:::

{py:meth}`value_counts <batcher.Dataset.value_counts>` gives the per-value tally, and {py:meth}`.is_duplicated() <batcher.plan.expr_ir.core.Expr.is_duplicated>` flags every row that shares its key with another.

```python
print(events.value_counts("status").to_pydict())
# {'status': ['new', 'paid'], 'count': [3, 2]}

flagged = events.select("user", "ts", dup=bt.col("user").is_duplicated())
print(flagged.to_pydict())
# {'user': ['a', 'a', 'b', 'b', 'c'], 'ts': [1, 3, 2, 4, 5],
#  'dup': [True, True, True, True, False]}
```

{py:meth}`.is_unique() <batcher.plan.expr_ir.core.Expr.is_unique>` is the complement. Both are window expressions, `count(*) OVER (PARTITION BY x)`, so they mark rows without collapsing them. That is what you want when the duplicates are a data-quality finding to report rather than noise to drop.

## Set operations dedupe too

`union(other, distinct=True)`, `intersect`, and `except_` carry set semantics. A plain `union` concatenates and keeps duplicates, which is the cheaper and usually correct default for appending partitions.

```python
a = bt.from_pydict({"id": [1, 2, 3]})
b = bt.from_pydict({"id": [3, 4]})

print(a.union(b, distinct=True).sort("id").to_pydict())
# {'id': [1, 2, 3, 4]}

print(a.intersect(b).to_pydict())
# {'id': [3]}

print(a.except_(b).sort("id").to_pydict())
# {'id': [1, 2]}
```

## Near-duplicates: same text, different bytes

Exact dedup does nothing to a crawl where the same article appears under three headers, or a product feed where a supplier re-sends a row with a trailing space. That is a similarity problem, and `ds.ml.drop_near_duplicates(column, threshold=...)` solves it with MinHash and LSH. It shingles the text, signs each document, and only compares documents that collide in a hash band.

```python
docs = bt.from_pydict(
    {
        "id": [1, 2, 3],
        "body": ["the quick brown fox", "the quick brown fox!", "entirely other text"],
    }
)
kept = docs.ml.drop_near_duplicates("body", threshold=0.7, key="id")
print(sorted(kept.to_pydict()["id"]))
# [1, 3]
```

{py:meth}`ds.ml.near_duplicates <batcher.api.dataset.ml.DatasetML.near_duplicates>` returns the pairs instead of dropping rows, so you can inspect what would go before you commit to it.

```python
pairs = docs.ml.near_duplicates("body", threshold=0.7, key="id")
print(pairs.to_pydict())
# {'key_a': [1], 'key_b': [2], 'jaccard': [0.8984375]}
```

:::{important}
Recall is not total, because a similar pair can miss every LSH band and survive. `bands` is the dial, trading candidate pairs for recall. Precision is guaranteed instead, since every returned pair is verified against `threshold`, so nothing below it is ever dropped. In other words, near-duplicate dedup can leave a duplicate behind, but it will not delete a row that was not one. See the {doc}`preprocessors guide </ml/preparing/preprocessors/index>` for the tuning detail.
:::

## Streams: bound the state with a watermark

An unbounded stream cannot remember every key it has ever seen. {py:meth}`drop_duplicates_within_watermark <batcher.Dataset.drop_duplicates_within_watermark>` forgets a key once the event-time watermark passes it, so memory stays bounded and a redelivery that arrives inside the lateness window is still caught.

```python
# docs: skip
deduped = stream.drop_duplicates_within_watermark(["event_id"], event_time="ts", lateness="10m")
```

Over a bounded source it degrades to plain exact deduplication. See {doc}`streaming </user-guide/moving-data/streaming/index>`.

## See also

- {doc}`Aggregations </user-guide/analyze/aggregations>`: `count_distinct` or `approx_count_distinct` when you want the count rather than the rows.
- {doc}`Sorting </user-guide/transform/rows/sorting>`: the tie-order rules that decide which row `keep="any"` gives you.
- {doc}`Data quality </user-guide/trust/data-quality>`: assert uniqueness instead of silently repairing it.
- {doc}`Aggregation internals </architecture/deep-dives/operators/aggregation-internals>`: the canonical hash key every dedup path shares, and why it has to be shared.
- {doc}`Deduplication recipe </cookbook/data-engineering/maintenance/deduplication>`: the same three jobs, on a real table.
- {doc}`Training-data dedup </cookbook/ml/pipelines/features/training-data-dedup>`: near-duplicate removal before an expensive embed.
- {doc}`Dataset API </api/relational/dataset>`: the `distinct`, `union`, `intersect`, and `except_` reference.
- {doc}`/cookbook/dataset/cleaning/deduplication`: exact keys, whole rows, and keeping a chosen survivor, as a script.
