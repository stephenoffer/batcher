# Reading query plans

When a query is slow, the first question is not "which operator is slow". It is "which
plan did I actually get". A predicate that failed to push below a join, a build side
chosen the wrong way round, a cardinality estimate off by 100x: all of them look
identical from the outside, and all of them are visible in one line of output.
`explain()` shows the plan the optimizer built. `explain(analyze=True)` runs it and
shows what really happened.

## Setup

```python
import batcher as bt

orders = bt.from_pydict(
    {
        "id": [1, 2, 3, 4, 5, 6],
        "customer": ["a", "b", "a", "c", "b", "a"],
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "status": ["paid", "paid", "open", "paid", "open", "paid"],
    }
)
customers = bt.from_pydict({"customer": ["a", "b", "c"], "region": ["us", "eu", "us"]})

query = (
    orders.filter(bt.col("status") == "paid")
    .join(customers, on="customer")
    .group_by("region")
    .agg(revenue=bt.col("amount").sum())
    .sort("revenue", descending=True)
)
```

## explain(): the plan, without running it

```python
print(query.explain())
```

:::{dropdown} The plan it prints
```text
sort                            est≈1 (default)
  aggregate                     est≈1 (default)
    hash_join                   est≈3 (default)
      scan                      est≈3 (exact)
      filter                    est≈1 (default)
        scan                    est≈6 (exact)

decisions:
  - [kyber/selection] join build side: left≈1 right≈3 [default] → swap build→left + broadcast
```
:::

Read it inside out: the leaves run first. Three things to look at.

The **tree shape** is the optimized plan, not the one you typed. `filter` sits directly
above the `orders` scan, below the join, so the predicate was pushed down. If you write a
filter after a join and it does *not* appear below the join here, something blocked the
pushdown (a UDF the optimizer cannot see through is the usual culprit), and you are
joining rows you are about to throw away.

The **estimate and its provenance** are the `est≈N (source)` column. `exact` means the
row count came from real metadata: a file footer, or an in-memory table. `default` means
the optimizer had nothing and used a heuristic. A plan full of `(default)` is a plan
making decisions in the dark.

The **decisions** block is the optimizer narrating itself. Here Kyber compared the two
sides, found the filtered orders side smaller, and swapped the build side of the hash
join so the smaller relation builds the table. That is the single most consequential
join decision, and this is where you check it.

## Estimates improve as the query runs

Core measures what actually came out of each operator and records it; Kyber reads that
back on the next plan. So the same `explain()` says something different after a run.

```python
q = orders.filter(bt.col("status") == "paid").group_by("customer").agg(n=bt.count())
print("--- cold")
print(q.explain())
q.collect()
print("--- after one run")
print(q.explain())
```

:::{dropdown} Cold, then warm
```text
--- cold
aggregate                       est≈1 (default)
  filter                        est≈1 (default)
    scan                        est≈6 (exact)

--- after one run
aggregate                       est≈3 (learned)
  filter                        est≈3 (default)
    scan                        est≈6 (exact)

decisions:
  - [core/io] source read at 6 MB/s (learned)
```
:::

`(learned)` is a measured cardinality, not a guess. This is why a query gets a better
plan the second time you run it, and why a one-off `EXPLAIN` on a cold cache can look
worse than the plan you actually get in production.

## explain(analyze=True): estimate vs actual

`analyze=True` executes the query and annotates every operator with what it measured.
This is the one you want when the plan looks fine but the query is slow.

```python
print(query.explain(analyze=True))
```

:::{dropdown} The annotated plan, with the summary and the decision log
```text
sort                            est≈2 actual=2 (1.0x)  0.0ms (0%)  cpu=93%  out=28B  interp
  aggregate                     est≈2 actual=2 (1.0x)  0.0ms (1%)  cpu=100%  out=28B  rss+256KB  interp
    hash_join                   est≈3 actual=4 (1.3x)  0.1ms (2%)  cpu=100%  out=76B  rss+1.0MB  interp
      filter                    est≈3 actual=4 (1.3x)  0.2ms (3%)  cpu=100%  out=84B  rss+1.5MB  interp
        scan                    est≈6 actual=6 (1.0x)  0.0ms (1%)  cpu=100%  out=126B  rss+126B  interp
      scan                      est≈3 actual=3 (1.0x)  0.0ms (0%)  cpu=89%  out=33B  interp

total: 6.19 ms, 2 rows out
bottleneck: filter (op 3), 3% of wall time — compute-bound (filter)
cpu utilization: 99% of cores (target >90%), peak memory 1.5MB (0% of budget, target >80%) — cores saturated

decisions:
  - [kyber/selection] join build side: left≈3 right≈3 [default] → broadcast
  - [carbonite/admission] feasible
```
:::

Every field on an operator line, and what it is telling you:

| Field | Reads as | What it means |
| --- | --- | --- |
| `est≈N (source)` | `(exact)`, `(learned)`, `(default)` | where the row estimate came from: real metadata, a measured run, or a heuristic |
| `est≈N actual=M (Kx)` | `1.0x` is perfect | the estimation error; a wild ratio is the root cause of most bad join orders |
| `0.2ms (3%)` | share of wall time | which operator the query is actually spending itself on |
| `cpu=100%` | core utilization | whether that operator saturated the cores it was given |
| `out=84B`, `rss+1.5MB` | bytes out, peak memory | the operator's footprint |
| `spill` | present or absent | the operator went out of core, so the constraint is memory, not CPU |
| `interp` / `jit` | the backend | `interp` on a hot arithmetic expression means the JIT fell back |

The estimation error is the one to read first. A `(0.01x)` on a join input means the
optimizer planned for 100 rows and got 10,000, and the fix for that is upstream (stale
statistics, a predicate on a column with no stats) rather than in the join. A `jit` that
says `interp` is correct, just slower than it could be: the compiler hit an expression or
a type it does not support and handed the batch back to the interpreter.

:::{warning}
On real data these numbers are wall-clock, so they move between runs. Compare shapes and
ratios, not milliseconds, and never gate a test on one of these timings.
:::

## stats(): the same measurements as a table

`stats()` runs the query and returns the per-operator measurements as a `RunStats`
object rather than a rendered tree, which is the shape you want when a script is checking
a number rather than a human reading a plan.

```python
print(query.stats())
```

:::{dropdown} The per-operator table
```text
 op  kind             rows_in    rows_out        ms      out_kb  backend
------------------------------------------------------------------------
  0  sort                   2           2      0.01           0  interp
  1  aggregate              4           2      0.02           0  interp
  2  hash_join              4           4      0.02           0  interp
  3  filter                 6           4      0.03           0  interp
  4  scan                   6           6      0.01           0  interp
  5  scan                   3           3      0.00           0  interp
------------------------------------------------------------------------
total: 2.93 ms, 2 rows out
bottleneck: filter (op 3), 1% of wall time — compute-bound (filter)
```
:::

That table is one run, so on a query this small the bottleneck line can name a
different operator each time; it earns its keep on data where one operator dominates.

:::{note}
`stats()` raises `BackendError` on a `map_batches` / ML pipeline, which runs outside the
relational engine, so there is nothing to measure per operator. Use
`explain(analyze=True)` there.
:::

## format="json": for tooling

Same profile, machine-readable. Use it to assert a plan property in a test, or to feed
a dashboard.

```python
import json

profile = json.loads(query.explain(analyze=True, format="json"))
print(sorted(profile["ops"][0]))
# ['algorithm', 'backend', 'cpu_util', 'depth', 'elapsed_ms', 'est_error', 'est_rows',
#  'kind', 'measured', 'op_id', 'peak_rss_bytes', 'provenance', 'result_bytes',
#  'rows_in', 'rows_out', 'selectivity', 'spill_bytes', 'spilled', 'threads']

print(profile["rows"], profile["spilled"], profile["carbonite_summary"])
# 2 False feasible
```

The document also carries `logical_ir` and `optimized_ir` (the plan before and after
Kyber), `decisions`, `adaptive_stages`, and the memory budget. Asserting on
`optimized_ir` is how you write a regression test that a predicate stays pushed down.

## A checklist for a slow query

Run `explain(analyze=True)` and go down the tree once:

- Is the filter below the join? If not, the optimizer could not see through something.
- Is `est` within an order of magnitude of `actual` at every level? If not, fix the
  estimate before you touch anything else.
- Did the build side get chosen the way you would have chosen it?
- Does the bottleneck operator say `spilled`? Then it is memory, not CPU.
- Does a hot expression say `interp` where you expected `jit`?

Then, and only then, start changing the query.

## See also

- [Performance](performance.md): the tuning knobs behind these measurements.
- [Caching](caching.md): stop re-running the plan you just read.
- [Troubleshooting](troubleshooting.md): what to do about what you found.
- [Query lifecycle](../deep-dives/query-lifecycle.md): the stages the plan passes
  through, which is what the tree is a picture of.
- [The plan IR](../deep-dives/plan-ir.md): the JSON document the tree is printed from, and
  the contract the Rust engine reads it under.
- [Cost model](../deep-dives/cost-model.md): how an `est≈N` becomes a join order.
- [Adaptive re-optimization](../deep-dives/adaptive-reoptimization.md): why the same
  `explain()` says something different after a run.
- [Optimizing a slow query](../tutorials/optimizing-a-slow-query.md): this checklist,
  walked end to end on a query that is actually slow.
- [Dataset API](../api/dataset.md): the `explain` and `stats` reference.
