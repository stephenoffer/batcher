# Schema evolution

Somebody upstream adds a column. They tell you in a Slack thread you are not in. Their
Monday file has `region`, their Friday file does not, and your directory now holds two
different schemas under one path.

:::{warning}
Nothing in your pipeline errors. Under the default `schema_mode="strict"`, the reader
takes the first file's schema as the truth for the whole directory, and the added column is
not in the result. The row counts are right, the sums are right, and a column of
data is missing.
:::

```python
import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

import batcher as bt

work = tempfile.mkdtemp()
raw = os.path.join(work, "raw")
os.makedirs(raw)

pq.write_table(
    pa.table({"id": [1, 2], "amount": pa.array([10, 20], pa.int64())}),
    os.path.join(raw, "2024-01-01.parquet"),
)
pq.write_table(
    pa.table({"id": [3], "amount": pa.array([30], pa.int64()), "region": ["us"]}),
    os.path.join(raw, "2024-01-02.parquet"),
)
```

## The default is a guess

`schema_mode` defaults to `"strict"`, and strict means *the first file's schema stands
for the whole directory*. It does not check the others. Read the directory and `region`
is not there:

```python
print(bt.read.parquet(raw).columns)
# ['id', 'amount']
print(bt.read.parquet(raw).sort("id").to_pydict())
# {'id': [1, 2, 3], 'amount': [10, 20, 30]}
```

Three rows, right counts, right sums. And a column of data that your dashboard will
never see, because the reader never looked at the second file's footer. That is the fast
path (one footer read instead of N), and it is the right default when every file really
does share a schema. It is a lie the moment they do not.

If the *type* moved rather than the column set (`amount` written as `int32` in January
and `float64` in March), strict does not go quiet. The first file's type is the contract,
March cannot be cast to it without losing the fractional part, and the read stops with a
{py:exc}`SchemaError <batcher.SchemaError>` naming the file, the column, both types, and `schema_mode="union"` as the
fix. An error you can see beats a column you cannot, but neither is what you wanted.

A file that is *missing* a declared column fails the same way and for the same reason:
strict promised that column for the whole directory. Only an **extra** column is dropped
rather than reported, because it was never part of the contract the first file set. That
is the one case above, and the reason it is the dangerous one.

It is no longer silent, though it is still a drop. Strict reads one extra footer, the
*last* file's, and warns when that file carries columns the result will not: enough to
catch a schema that evolved forward, which is how almost every directory acquires a new
column, and cheap enough not to give up what strict is for. The warning names the columns
and points at `schema_mode="union"`. It claims only what it looked at: a column that
appears in a middle file and not the last one is still dropped without a word.

## Union: read every footer, reconcile

```python
evolved = bt.read.parquet(raw, schema_mode="union")
print(evolved.sort("id").to_pydict())
# {'id': [1, 2, 3], 'amount': [10, 20, 30], 'region': [None, None, 'us']}
```

`union` reads every file's schema (concurrently, so a thousand files is one round of
metadata reads, not a thousand serialized ones) and unifies them: the union of columns
in first-seen order, each promoted to the common supertype of its occurrences, and a
column a given file lacks is filled with NULL for that file's rows.

The old rows get `region = NULL`, which is the honest answer. They were written before
the column existed, and nobody knows what region they were.

Type promotion follows a lattice, not a coin flip. NULL adopts the other side, integers
widen to `int64`, floats to `float64`, an int/float mix promotes to `float64`. So the
file where `amount` turned into a float pulls the whole column up with it, without
losing the integer rows:

```python
pq.write_table(
    pa.table({"id": [4], "amount": pa.array([40.5], pa.float64()), "region": ["eu"]}),
    os.path.join(raw, "2024-01-03.parquet"),
)
print(bt.read.parquet(raw, schema_mode="union").sort("id").to_pydict())
# {'id': [1, 2, 3, 4], 'amount': [10.0, 20.0, 30.0, 40.5], 'region': [None, None, 'us', 'eu']}
```

The lattice deciding what your column ends up as is the same one the engine uses for a
`union`, a `coalesce`, a comparison, and a join key, so a directory that reconciles on
read also joins and unions afterwards. The pairs a drifting directory actually produces:

| The column appears as | It reads back as |
|---|---|
| NULL and anything | the other side's type |
| `int32` and `int64` | `int64` |
| `float32` and `float64` | `float64` |
| an integer and a float | `float64` |
| `decimal(10,2)` and `decimal(12,4)` | `decimal(12,4)` — the finer scale, the wider integer part |
| a decimal and an integer | a decimal wide enough for both, so the cents survive |
| `timestamp[ms]` and `timestamp[us]` | `timestamp[us]` — the finer resolution |
| a date and a timestamp | the timestamp, since a date is midnight |
| `string` and `large_string` | `large_string` |
| a dictionary-encoded column and a plain one | the plain value type |
| two timestamps in different timezones | nothing, because the read fails |
| `int64` and `string` | nothing, because the read fails |

The full table, including the nested cases, is on
{doc}`the type system page </user-guide/transform/columns/type-system>`.

`schema_mode="latest"` is the other useful mode: the newest file's schema wins outright
and older files are cast toward it. Reach for it when the newest file *is* the contract
and older columns are debris you want gone.

| `schema_mode` | What it reads | Cost | Use it when |
|---|---|---|---|
| `strict` (default) | the first file's footer, applied to all | one metadata read | every file really does share a schema |
| `union` | every footer, unified column-wise and promoted | one concurrent round of metadata reads | the directory has drifted and you want all of it |
| `latest` | the newest file's schema, older files cast toward it | every footer | the newest file is the contract and old columns are debris |

## When it cannot be reconciled

There is no common type for `int64` and `string`, and Batcher will not invent one by
stringifying your numbers:

```python
from batcher._internal.errors import SchemaError

bad = os.path.join(work, "bad")
os.makedirs(bad)
pq.write_table(pa.table({"id": [1], "amount": pa.array([10], pa.int64())}), f"{bad}/a.parquet")
pq.write_table(pa.table({"id": [2], "amount": pa.array(["12.00 USD"])}), f"{bad}/b.parquet")

try:
    bt.read.parquet(bad, schema_mode="union").to_pydict()
except SchemaError as err:
    print(err)
# column 'amount' has incompatible types across files: int64 vs string (no non-lossy
# common type). Cast explicitly or use schema_mode='latest'.
```

Somebody started writing amounts as `"12.00 USD"`. This is a data contract violation,
and the read failing is the correct outcome. Fix it upstream, or quarantine that file,
or accept `schema_mode="latest"` and handle the string yourself. Do not paper over it in
the reader.

## Publishing the evolved table

Raw files evolve. Curated tables should not evolve *by accident*. Reconcile at the read,
then republish:

```python
curated = os.path.join(work, "curated")
bt.read.parquet(raw, schema_mode="union").write.delta(curated, mode="overwrite")

print(bt.read.delta(curated).sort("id").to_pydict())
# {'id': [1, 2, 3, 4], 'amount': [10.0, 20.0, 30.0, 40.5], 'region': [None, None, 'us', 'eu']}
```

The overwrite is one atomic commit, so no reader ever sees a half-evolved table, and the
pre-evolution version is still there at `version=N-1` if the new column turns out to be
garbage.

:::{important}
**Appending a wider batch to a Delta table does not widen the table.** The sink writes to
the table's committed schema. A column the table does not know about would land in the data
files but stay invisible to the table, which is a silent loss that resurfaces as wrong data
the day someone adds the column for real. The engine refuses that write rather than let it
happen.
:::

```python
narrow = os.path.join(work, "narrow")
bt.from_pydict({"id": [1], "amount": [10]}).write.delta(narrow, mode="overwrite")
try:
    bt.from_pydict({"id": [2], "amount": [20], "region": ["us"]}).write.delta(
        narrow, mode="append"
    )
except Exception as exc:
    print(type(exc).__name__)
# CommitError
```

To actually widen the table, say so. `merge_schema=True` commits the new column and
backfills `NULL` for every row already there:

```python
bt.from_pydict({"id": [2], "amount": [20], "region": ["us"]}).write.delta(
    narrow, mode="append", merge_schema=True
)

print(bt.read.delta(narrow).sort("id").to_pydict())
# {'id': [1, 2], 'amount': [10, 20], 'region': [None, 'us']}
```

:::{tip}
`merge_schema=True` is a widening operation only. It adds columns. It will not change a
column's type. A type change is still a rewrite.
:::

That leaves two workable shapes for the table the new column has to reach, and the choice
comes down to who pays: the writer once, or every reader forever.

::::{tab-set}

:::{tab-item} Curated Delta table

```python
# docs: skip
bt.read.parquet(raw, schema_mode="union").write.delta(curated, mode="overwrite")
```

To evolve the table you rewrite it, which is a full rewrite and costs what a full rewrite
costs. Budget for it. What you get back is one committed schema, swapped in atomically,
with the pre-evolution version still queryable.
:::

:::{tab-item} Raw Parquet, reconciled on read

```python
# docs: skip
bt.read.parquet(raw, schema_mode="union")
```

Keep the raw layer in plain Parquet, where `schema_mode="union"` does the reconciling on
every read and nothing has to be rewritten at all. The cost moves to the read: every
footer, every time.
:::

::::

## Guard the boundary

The cheapest protection is asserting the shape you expect, at the point where you depend
on it:

```python
expected = {"id", "amount", "region"}
actual = set(bt.read.parquet(raw, schema_mode="union").columns)
print(sorted(expected - actual), sorted(actual - expected))
# [] []
```

An empty left side means nothing you need has disappeared. A non-empty right side means
somebody added a column, which is usually fine and always worth knowing.

:::{tip}
Put this check in the job and log the diff. You will then hear about the schema change
from your own pipeline instead of from Slack, and you will hear about it on the first run
that sees the new column rather than on the day someone notices the dashboard is short a
dimension.
:::

## Compaction reads the union, always

Compacting a drifted directory is the one place the narrowing is not recoverable.
{py:func}`bt.compact <batcher.compact>` rewrites many small files into fewer large ones and
**deletes the files it replaced**, so a column the read failed to see is not missing from
one result, it is gone from the table. Compaction therefore reads the union whatever you
would have passed, and you do not have to remember to ask:

```python
compacted = tempfile.mkdtemp()
for name, table in (
    ("a.parquet", pa.table({"id": [1], "amount": [10]})),
    ("b.parquet", pa.table({"id": [2], "amount": [20], "region": ["us"]})),
):
    pq.write_table(table, os.path.join(compacted, name))
bt.compact(compacted)
print(sorted(bt.read.parquet(compacted, schema_mode="union").columns))
# ['amount', 'id', 'region']
```

The same holds for a Hive-partitioned directory, where the partition column comes back
alongside the evolved one rather than instead of it.

## See also

- {doc}`Quality gates </cookbook/data-engineering/maintenance/quality-gates>`: the same idea, applied to values instead of columns.
- {doc}`Multi-source join </cookbook/data-engineering/modeling/multi-source-join>`: where a type mismatch does its real damage.
- {doc}`Incremental ingest </cookbook/data-engineering/ingest/incremental-ingest>`: the loader dropping new files into the
  directory that drifted.
- {doc}`Reading data </user-guide/moving-data/reading-data>`: the reader options in full.
- {doc}`Type system </user-guide/transform/columns/type-system>`: the promotion rules behind the lattice.
- {doc}`Delta Lake </integrations/lakehouse/delta-lake>`: what a committed schema is.
- {doc}`IO API reference </api/relational/io>`: `schema_mode` and the rest of the reader arguments.
