# Governance bypass audit: what a policy was keyed on, and everything that evaded it

**Status:** audit, 2026-09-01. Internal working document, excluded from the published site.

This is the record of one pass over `governance/` and its call sites. It started from three
cells the [platform parity scorecard](platform_parity_scorecard.md) listed as governance gaps
-- write privileges, revoke, path aliasing -- and the pass found rather more than those,
because closing the third one meant asking, for the first time, *what name is a policy
actually matched against*. Every answer to that question was a bypass.

The findings are recorded together because they are one defect wearing many faces, and
because the method that found them generalises: **sweep the registry, do not audit the
examples.** Almost all of them were invisible to a reader auditing the code path by hand and
obvious to a twenty-line script that constructed every registered source and compared its
governance name to its path.

## The one defect

`Source.identity()` names a **relation**. `SecurityCatalog` keys on a **table**. Governance
read the first and called it the second.

The distinction is not pedantic and neither side is wrong on its own terms. An identity has
to distinguish a source pinned to some of a directory's files, or capped at `n_rows`, or
narrowed to `columns`, or resolved to Delta version 7, because the statistics cache is keyed
on it and one relation's row count must not be handed to another. That is a real bug it
prevents, and `io/base/source.py::identity` records it: a pruned MERGE estimated a
100,000-row join at 2.4 TB and spilled it, because a one-file source inherited the whole
table's statistics. Cited from that docstring, not re-measured here.

A policy is the other thing entirely. It is written before anyone has read the table, by
someone who does not know which version a given job will land on or how it will slice the
files. It has to be keyed on what that person can write down.

## What evaded it

Every fix has a test that fails without it. **What was reproduced differs by row, and the
difference is worth stating rather than glossing.** The first four rows were run end to end:
the query was executed under a catalog and the withheld column came back. The rest were
established at the *name*, by constructing the source and comparing `table_name(src)` to the
path -- from which "ungoverned" follows, because `SecurityCatalog.governs` is an exact match
and a name no policy contains cannot match one. That is a sound derivation and it is not the
same as having read a governed Delta table on a live cluster, which nothing here did.

| Read | Returned | Because |
|---|---|---|
| `read.parquet(path)` | the mask, and no withheld column | correct |
| `read.parquet(path, n_rows=2)` | **the raw value and the withheld column** | identity is `<path>#<digest>` |
| `read.parquet(path, columns=[...])` | **the raw value and the withheld column** | same |
| `read.parquet([a, b])` | **every column of `a`** | modelled as their common parent, which no policy names |
| any Delta / Delta CDF / Hudi read | **ungoverned entirely** | identity is `<path>@<version>` |
| any Iceberg read | **ungoverned entirely** | identity is `<catalog>:<identifier>@<snapshot>` |
| any text read | **ungoverned entirely** | identity is `<mode>:<path>` |
| any Kafka / Kinesis / Pulsar / Event Hubs / Pub/Sub read | **ungoverned entirely** | identity folds in a connection fingerprint |
| any Snowflake / BigQuery / Databricks / ADBC / DB-API read | **ungoverned entirely** | same |
| any Mongo / Cassandra / Elasticsearch / DynamoDB / Couchbase / Redis / HBase / Neo4j read | **ungoverned entirely** | same |
| any HDF5, protobuf, or autoloader read | **ungoverned entirely** | identity carries the dataset, or prefixes the format |

The first three need no privileged API and no internal call. They are one ordinary keyword
argument typed by an ordinary user against the documented public reader. The rest need
nothing at all: reading a governed Delta table, or a governed Snowflake table, was enough.

A warehouse or store read defined by a **query** rather than a table is a separate case and
is now named ``""`` -- honestly ungovernable. A raw SQL string names no table this engine can
resolve without parsing it, and a policy matched against a guessed table governs the wrong
data. Saying so is also what lets `governance.mode` refuse or warn about the read, which it
cannot do for a source returning a plausible-looking name that matches nothing.

Two more rows belong in the table and are of a different kind, because they were governed
*too well* rather than too little:

| Operation | Did | Because |
|---|---|---|
| `bt.compact()` on a governed table | **destroyed the data**: `email` became `XXXXXXX` and `ssn` was dropped, permanently | it reads the table and writes the result back, and inside a block that read is the principal's view |
| a copy-on-write `MERGE` into a governed table | **destroyed every row the clauses did not match** | same shape |

Neither raised. Neither left anything in the result to suggest it had happened.

## The fixes, and the one that is a refusal

A source that can name its table now says so: `governed_name()`, which
`api/security/_binding.py::table_name` asks before falling back to parsing an identity
string. Returning the name rather than stripping the qualifier off the identity is
deliberate -- a path may legitimately contain `#`, and a rule that guessed where the suffix
began would fail *open* on exactly such a path.

Brokers are named by their topic, which is the durable thing an operator writes a policy
about without knowing which cluster a job will point at. The rate generator and the raw
socket return the empty string, because `3.0:100` and `localhost:9999` looked governable
while being governed by nothing, and `governance.mode` can only refuse or warn about an
ungovernable read if the source admits to being one.

A pinned multi-path read is **refused** when one of its paths lies under a policy the scan
is not already governed by. Governing it properly means resolving several policies into one
scan, and `enforce` takes one table per scan. The refusal is narrow: reading some files of
one governed table still works, including a Hive-partitioned file two directories inside the
table the policy names.

`compact` and the copy-on-write `MERGE` are **refused** on a governed table. This is the
decision most worth arguing with, so the argument is written down. The obvious repair is to
let the operation read the raw table with the statement's own authority, which is what SQL
does and is safe on the face of it, since the rows never surface to the caller. It is not
safe here: a `when_matched` clause writes arbitrary expressions into target columns, so
`update(notes=col("email"))` moves a masked column's raw value into an unmasked column of
the same table, which the principal then reads. Making the merge safe means checking every
clause expression against the mask policy. That is a real piece of design; a refusal is not.
A native Delta or Iceberg `MERGE` is **not** refused, because its client merges against the
raw table inside itself and nothing is read through the principal's view.

## Method, and why it is the part worth keeping

Three techniques did the work, and only the first is unusual.

**Sweep the registry, not the examples.** Eleven of the findings came from a script that
constructed every source in `SOURCES` with the same path and compared `table_name(src)` to
it. Reading the code would have found the two formats a reader happens to think of. The
sweep found every one, including formats nobody would have guessed carried a version in
their identity. `tests/unit/test_governed_source_names.py` is that script as a contract, so
a source added next year is covered without anyone remembering the file exists -- with a
floor on how many sources it constructed, because a registry-driven test that sweeps nothing
passes.

**A skip count is not a pass, and this sweep proved it on itself.** The first form of the
script constructed each source with a bare path and *skipped whatever raised*. It reported
"35 governed correctly, 11 mis-named, 23 could not be constructed", and the 23 read as
nothing to check. They were the entire database, warehouse and document-store family --
Snowflake, BigQuery, Databricks, ADBC, DB-API, Mongo, Cassandra, Scylla, Elasticsearch,
DynamoDB, Couchbase, Redis, HBase, Neo4j -- and every one of them was mis-named, because
each folds a `connection_fingerprint` into its identity. That digest is a sha256 of the
connection options, so the name was not merely wrong, it was **unwritable**: no operator
could have typed it into a policy even knowing the rule.

Finding them took a second table giving each source the least it needs to construct. That
table is now in the contract, and so is
`test_the_connector_table_covers_the_skipped_sources`, which fails when a registered source
is constructible by neither sweep -- because a source that both tests skip is covered by
neither while both stay green, which is exactly the state the first sweep left thirteen
connectors in. It caught `protobuf` on its first run.

The fix for the whole NoSQL family was one method on `nosql/base.py::ScanSource`, returning
the `_identity_suffix()` that base already had: the "human locator" the identity docstring
names, with the fingerprint stripped back off. The hook existed; nothing had asked it the
governance question.

**Assert the deny and the allow.** Every test here proves the refusal *and* the matching
permission. A check that refuses everything passes the first half of every security test
ever written, and two of the fixes in this pass are refusals, which is exactly the shape
that failure hides in.

**Mutation-test the assertions rather than trusting them.** Restoring the old `table_name`
fails 19 of the 27 source-naming tests; removing Kinesis's child-shard adoption fails 8 of
its 14, and taking the highest-numbered parent instead of the lowest fails 3. A test whose
discriminating power has not been measured is a claim, and this repository's rules are
explicit that a claim is to be run rather than argued.

## What this pass did not do

- **`governance.default_deny` is still unimplemented**, and still refused at config time,
  which is the honest state. Implementing it is now cheap -- the catalog already treats a
  table carrying any grant as deny-by-default for every privilege, and `default_deny` is
  that rule extended to a table carrying none -- but it also needs the config validator's
  refusal removed, and that file was mid-refactor.
- **`ResidencyCatalog` is consulted by nothing.** Its matching was fixed here (it had the
  same alias hole) and its call site is still missing. See the note in
  `governance/residency.py`, which was verified by instrumentation rather than by reading.
- **`Redact` disclosed a value shorter than what it reveals. Fixed.**
  Measured: under `Redact(show_last=4)`, the values `abcd`, `abc`, `ab` and `a` came back
  **unchanged**; under `Redact(show_first=2)`, `ab` and `a` did; under
  `Redact(show_first=2, show_last=2)`, everything up to four characters did. It is
  semantically consistent -- showing the last four of a three-character value shows all
  three -- and it was still a masking control returning the raw value, on exactly the column
  where it matters: most values in a name or postcode column are short.

  `Redact.__post_init__` already stated the principle it broke, while rejecting a negative
  count: under-masking "is the one direction a redaction policy must never be wrong in".

  The fix is in `Redact`, not in `batcher.mask`. The expression is a general string utility
  and its literal semantics are right; the security primitive is what must not under-mask. A
  value no longer than `show_first + show_last` is now masked completely.

  The two costs this entry previously held it back for are now measured rather than assumed.
  The full-redaction case pays nothing at all: with nothing revealed, no value can be short
  enough to escape, so `show_first == show_last == 0` keeps the bare `mask` it always lowered
  to. And the partial reveal, which does take a length test and a branch per row, costs
  **+2.7 ms over 4,000,000 rows** against the bare `mask` -- 234.7 ms against 232.0 ms, about
  1%, with the full-mask path unchanged at 0.97x. That is the number the entry was waiting
  for and it does not justify leaving a masking control returning cleartext.

  It changed three doctests, not one. `masks.py` was obvious; `policy.py` and `enforce.py`
  assert the rendered mask expression verbatim and were missed, shipped, and printed
  `***Test Failed*** 2 failures.` into two consecutive `just docs` runs that both exited 0 --
  which is its own defect, fixed separately in `tools/check_doctests.py`.

  `tests/unit/test_redact_short_values.py` pins the behaviour; reverting the fix fails 12 of
  its 34 tests. It holds distributed too: a governed read over four Parquet parts returned
  800 rows with zero unmasked `ssn` under `distributed=True`, so the conditional survives
  being sharded.

- **The distributed run has now happened.** On the shared multi-node cluster, over four
  Parquet parts on `/mnt/cluster_storage` so every worker can see them: the governed read
  returned **800 rows, region `eu` only, zero unmasked `ssn`** both single-node and under
  `distributed=True`, with the same row multiset, the same row count and the same schema.
  The control is what makes that mean anything -- the *ungoverned* plan run distributed
  returned 1,600 rows across both regions with all 1,600 `ssn` values in the clear, so the
  check demonstrably sees a leak when there is one.

  What this does and does not settle. The write authorization still happens on the driver
  inside `terminal.core._write` before any fan-out, so that ordering is argued from structure
  and pinned by `test_the_check_precedes_any_distributed_fan_out`, not by this run. What was
  unmeasured and now is not: that the *read* rewrite -- row filter and column mask -- survives
  being sharded across machines.

  One methodological note, because it cost an hour here and would cost it again.
  `collect(num_partitions=N)` is **not** a stand-in for this. It is a spill knob: varying it
  from 1 to 8 visibly changes physical execution (a spilled 300,000-row sort came back in 61
  batches against 72), yet it did not move the `LIMIT`-over-unordered-`group_by` result at
  all -- not at 200 rows, not at 300,000, with spill on or off, in memory or over four
  Parquet files. That shape *is* the one `.claude/rules/python-control-plane.md` records as
  diverging between a single-node run and a two-worker one, on a four-file Parquet source:
  groups 0, 1, 2 against 3, 5, 8. That divergence is cited from the rules, not re-measured
  here; what was measured here is only that `num_partitions` does not reproduce it. A "stable
  across partition counts" test therefore proves something about spilling and nothing about
  distribution.
