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
#  'invol_ctx_switches', 'io_read_bytes', 'io_write_bytes', 'kind', 'major_faults',
#  'measured', 'minor_faults', 'op_id', 'peak_rss_bytes', 'preemption_rate',
#  'provenance', 'result_bytes', 'rows_in', 'rows_out', 'selectivity', 'spill_bytes',
#  'spilled', 'threads', 'vol_ctx_switches']

print(profile["rows"], profile["spilled"], profile["carbonite_summary"])
# 2 False feasible
```

The document also carries `logical_ir` and `optimized_ir` (the plan before and after
Kyber), `decisions`, `adaptive_stages`, the memory budget, and `machine`. Asserting on
`optimized_ir` is how you write a regression test that a predicate stays pushed down.

`machine` names the hardware the run was measured on, as a readable label and a fingerprint:

```text
machine: GenuineIntel/16c/64GiB/l3=32MiB/nvme [a2f5aeb968ef]
```

Every timing above it is relative to that machine, so it's what makes two profiles from
different nodes comparable. The fingerprint is also the key the engine stores its learned
costs under, which makes it the answer to "why did this node plan worse than that one?" A
different fingerprint means the two learned separately and neither inherited the other's
measurements. {doc}`/user-guide/operate/performance` covers what that changes.

## What the operator cost the machine

Rows and milliseconds say what an operator did. They don't say what it cost the machine, and
that's where most unexplained slowness lives. Alongside the timings, each operator reports
what the operating system charged it.

Two of those readings change what you should do, so `explain(analyze=True)` prints them on
the operator's line, and only when they're present:

```text
aggregate    est≈1,000 actual=1,000  842.1ms (91%) cpu=11%  out=64KB  interp  PAGING(31,204 major faults)
```

`PAGING` means the kernel was fetching back memory the process already believed it held. It
comes first because it invalidates every other number on the line: the operator's time is
storage latency, and its low CPU number is threads blocked rather than work not done. Adding
parallelism here makes things strictly worse, because each extra worker faults in its own
working set and evicts the others. Lower `memory.max_memory_bytes` so the engine spills
deliberately instead, or give the process more memory.

`contended` means the scheduler repeatedly took cores away from the operator while it ran, so
something else on the machine wanted them. Batcher measures this per operator rather than from
the machine's load average, which is a one-minute average over the whole box and misses a short
query that lands inside someone else's burst. Treat the run's timing as a lower bound on the
plan's real speed.

The rest are in the JSON document rather than on the line, because they're inputs to a
diagnosis rather than verdicts:

| Field | What it tells you |
|---|---|
| `major_faults` | Pages fetched from disk. Any material count means the box is paging. |
| `minor_faults` | Pages committed without disk I/O, giving the *measured* working set, against which `peak_rss_bytes` is a high-water mark and the planner's estimate is a model. |
| `invol_ctx_switches` | Times the scheduler evicted the operator from a core. |
| `preemption_rate` | The same, per core-second, so it compares across operators of different widths and durations. |
| `vol_ctx_switches` | Times the operator blocked and yielded. High against low `cpu_util` means genuinely I/O- or lock-bound, rather than under-parallelized. |
| `io_read_bytes` | Bytes that actually reached the block device. A warm and a cold scan of the same file are identical in every other field and differ by two orders of magnitude in cost. |
| `io_write_bytes` | Bytes written to the device, spill included. |

Every one of these is `0` when the platform can't report it, and `0` means *not measured*
rather than *none*. On the streaming executor the per-operator counters are unmeasured by
design: its operators interleave, so no one of them owns a wall interval that
process-wide counters could honestly be attributed to.

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

- {doc}`Performance </user-guide/operate/performance>`: the tuning knobs behind these measurements.
- {doc}`Caching </user-guide/operate/caching>`: stop re-running the plan you just read.
- {doc}`Troubleshooting </user-guide/operate/troubleshooting>`: what to do about what you found.
- {doc}`Query lifecycle </deep-dives/query/query-lifecycle>`: the stages the plan passes
  through, which is what the tree is a picture of.
- {doc}`The plan IR </deep-dives/query/plan-ir>`: the JSON document the tree is printed from, and
  the contract the Rust engine reads it under.
- {doc}`Cost model </deep-dives/adaptive/cost-model>`: how an `est≈N` becomes a join order.
- {doc}`Adaptive re-optimization </deep-dives/adaptive/adaptive-reoptimization>`: why the same
  `explain()` says something different after a run.
- {doc}`Optimizing a slow query </tutorials/optimizing-a-slow-query>`: this checklist,
  walked end to end on a query that is actually slow.
- {doc}`Dataset API </api/relational/dataset>`: the `explain` and `stats` reference.
- {doc}`/cookbook/operations/inspecting_a_query`: reading a plan and timing a query, as a runnable script.
