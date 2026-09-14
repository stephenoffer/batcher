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
query plan (planned)                                               6 operators
──────────────────────────────────────────────────────────────────────────────
OPERATOR                              ESTIMATE  NOTES
sort  [revenue]                          est≈1  (default)
└─ aggregate  [by region · sum]          est≈1  (default)
   └─ hash_join  [inner on customer]     est≈2  (default)
      ├─ scan  [source 1]                est≈3  (exact)
      └─ filter  [status = paid]         est≈1  (default)
         └─ scan  [source 0]             est≈6  (exact)  pushed[status = paid]

decisions:
  - [kyber/selection] join build side: left≈1 right≈3 [default] → swap build→left + broadcast
```
:::

Read it inside out: the leaves run first, and the spine shows what feeds what. A `├─`
means the operator above it has another input below; a `└─` means this is the last one.
On a join that is the whole question, so it is drawn rather than left to be counted out of
an indent. Five things to look at.

The **bracketed description** after each operator says what *that* operator does: the join
type and its keys, the group keys and aggregates, the sort keys, the predicate, the source.
Without it a plan with four joins prints four identical `hash_join` lines, and "which join
is this one" is the first question anyone asks of a join tree.

The **tree shape** is the optimized plan, not the one you typed. `filter` sits directly
above the `orders` scan, below the join, so the predicate was pushed down. If you write a
filter after a join and it does *not* appear below the join here, something blocked the
pushdown (a UDF the optimizer cannot see through is the usual culprit), and you are
joining rows you are about to throw away.

The **estimate and its provenance** are the `est≈N (source)` column. `exact` means the
row count came from real metadata: a file footer, or an in-memory table. `default` means
the optimizer had nothing and used a heuristic. A plan full of `(default)` is a plan
making decisions in the dark.

The **pushed filter** is the `pushed[...]` note on a scan: the predicate the plan handed
that source to apply for itself, rather than reading the rows and filtering them after.
A scan with no note received nothing. See {doc}`pushdown` for which predicate shapes each
source can take.

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
query plan (planned)                                                3 operators
───────────────────────────────────────────────────────────────────────────────
OPERATOR                               ESTIMATE  NOTES
aggregate  [by customer · count_star]     est≈1  (default)
└─ filter  [status = paid]                est≈1  (default)
   └─ scan  [source 0]                    est≈6  (exact)  pushed[status = paid]

--- after one run
query plan (planned)                                                3 operators
───────────────────────────────────────────────────────────────────────────────
OPERATOR                               ESTIMATE  NOTES
aggregate  [by customer · count_star]     est≈3  (learned)
└─ filter  [status = paid]                est≈4  (default)
   └─ scan  [source 0]                    est≈6  (exact)  pushed[status = paid]

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
query plan (measured)                                                                6 operators  ·  2 rows  ·  44ms
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
OPERATOR                                ESTIMATE    ACTUAL   MISS   TIME     OP SHARE  NOTES
▶ sort  [revenue]                          est≈2  actual=2  exact   14µs  ▎░░░░░   3%  interp
▶ └─ aggregate  [by region · sum]          est≈2  actual=2  exact  316µs  ████░░  65%  interp
▶    └─ hash_join  [inner on customer]     est≈4  actual=4  exact   10µs  ░░░░░░   2%  interp
▶       ├─ filter  [status = paid]         est≈4  actual=4  exact  146µs  █▉░░░░  30%  interp
▶       │  └─ scan  [source 0]             est≈6  actual=6  exact    1µs  ░░░░░░  <1%  interp  pushed[status = paid]
        └─ scan  [source 1]                est≈3  actual=3  exact    1µs  ░░░░░░  <1%  interp

where the time went
  operators       485µs     1%  of 44ms
  elsewhere        44ms    99%  planning, optimization, admission, FFI crossing, result assembly

total: 44.13 ms, 2 rows out
bottleneck: aggregate (op 1), 64% of operator time — compute-bound (aggregate)
cpu utilization: 100% of cores (target >90%) — cores saturated
machine: GenuineIntel/16c/64GiB/l3=32MiB/nvme [a2f5aeb968ef]

decisions:
  - [kyber/selection] join build side: left≈4 right≈3 [default] → broadcast
  - [carbonite/admission] feasible
  - [carbonite/resources] memory pressure NORMAL, envelope 0% used at peak
```
:::

Every field on an operator line, and what it is telling you:

| Column | Reads as | What it means |
| --- | --- | --- |
| `▶` | present or absent | the operator is on the *critical path*: the hottest chain from the root down. Drawn only when the plan branches, because on a straight chain every operator is on it, and a legend below the table says so wherever it appears. |
| `OPERATOR` | a tree, then `[detail]` | the optimized plan. The spine says what feeds what; a `└─` is a last input. The bracket says what the operator does: join type and keys, group keys and aggregates, sort keys, predicate. |
| `ESTIMATE` | `est≈N` | the rows Kyber planned for, with its provenance in `NOTES` when the plan was not run. |
| `ACTUAL` | `actual=N` | the rows the operator really produced. |
| `MISS` | `exact`, `3.4x over`, `5000.0x under` | how far the estimate missed, and **which way**. `over` means the plan expected more rows than arrived. An estimate below one row is compared against one row rather than against itself: a row count is a count, so a selectivity that underflowed down a chain of predicates has no usable denominator, and dividing by it printed fifteen digits of an artifact. A genuine miss, however large, is still reported in full. |
| `TIME` | `268µs`, `54ms`, `1m03s` | wall time in the operator. Sub-millisecond work is reported in microseconds rather than rounded to `0.0ms`, which used to make the fastest steps look unmeasured. |
| `OP SHARE` | a bar and a percentage | the operator's share of **total operator time**, so the column ranks operators against each other. The wall clock's own division is the `where the time went` block. |
| `NOTES` | conditional clauses | strategy (`broadcast`), backend (`interp` / `jit`), `spill 2.0 GiB`, `rss+…`, `pushed[…]`, `PAGING(…)`, `contended(…)`. Each appears only when it has something to say. |

Below the table:

`where the time went` splits the wall clock between the operators and everything else —
planning, optimization, admission, the crossing into the Rust engine, and assembling the
Arrow result. On a small query that remainder is usually most of it, and it is the number
worth acting on. On a large one it should be a rounding error; if it is not, the plan is
not where your time is going.

`bottleneck` names the operator that owns the most operator time. `cpu utilization` grades
the run against core saturation and, when a budget is known, peak memory against it.
`machine` names the hardware, which is what makes two profiles comparable.

`what to look at` appears only when something is worth acting on: an estimate that missed
by 4x or more, an operator that spilled, or one the machine was paging against. Each entry
says what happened and what to do about it.

The estimation error is the one to read first. A `(0.01x)` on a join input means the
optimizer planned for 100 rows and got 10,000, and the fix for that is upstream (stale
statistics, a predicate on a column with no stats) rather than in the join. A `jit` that
says `interp` is correct, just slower than it could be: the compiler hit an expression or
a type it does not support and handed the batch back to the interpreter.

:::{warning}
On real data these numbers are wall-clock, so they move between runs. Compare shapes and
ratios, not milliseconds, and never gate a test on one of these timings.
:::

## Large plans

A generated plan — a `_sql` query over a wide star schema, a union of eighty partitions, a
pipeline built by a loop — routinely runs to hundreds of operators. Printing all of them
costs the reader the thing they came for, so past roughly two dozen operators
`explain(analyze=True)` changes shape in three ways.

It names the hot operators first, before the tree:

```text
hot operators (top 5 of 143 measured)
  aggregate (op 1)                200ms  ███▌░░  57%  100 rows
  hash_join (op 2)                 50ms  ▉░░░░░  14%  200 rows
  scan (op 4)                      30ms  ▋░░░░░   9%  10 rows
  scan (op 5)                      15ms  ▎░░░░░   4%  10 rows
  scan (op 6)                      10ms  ▎░░░░░   2%  10 rows
```

It marks the critical path with `▶`: the chain from the root that descends, at every
branch, into whichever side holds the most time beneath it. On a deep join tree that is the
answer to "which side is costing me", and no per-operator column can give it — both sides
look individually modest while one of them carries the run.

And it folds subtrees that cannot contain the answer:

```text
▶       │  ├─ scan              est≈10   actual=10   exact  3.8ms  ░░░░░░   1%  interp
        │  ├─ … 17 more
  17 operators folded: subtrees under 1% of operator time. explain(format="json") lists every one.
```

Indentation stops growing past ten levels, and a `⋯` in place of the outermost ancestor
bars says where it stopped. A long pipeline of chained `with_columns` and `filter` calls is
a *deep* plan rather than a wide one, and the spine costs three columns per level: left
unbounded it consumed the whole operator column, so the part of each row that survived
truncation was the indentation and the part discarded was the operator's name. The nearest
ancestors are the ones kept, because those are the branches a reader is still resolving.

A run of consecutive cold siblings collapses into one line rather than one line each,
because seventeen `… 1 operator folded` markers are exactly as long as the seventeen rows
they replaced. Nothing on the critical path is ever folded, no root is ever folded, and
nothing is folded at all without measurements to judge coldness by — a plan explained
without `analyze=True` prints in full, because there is no basis for calling any of it
cold. When you want every operator regardless, `format="json"` always carries all of them.

## stats(): the same measurements as a table

{py:meth}`stats() <batcher.Dataset.stats>` runs the query and returns the per-operator measurements as a `RunStats`
object rather than a rendered tree, which is the shape you want when a script is checking
a number rather than a human reading a plan.

```python
print(query.stats())
```

:::{dropdown} The per-operator table
```text
OP  KIND       ROWS IN  ROWS OUT  TIME  OP SHARE       OUT  BACKEND
───────────────────────────────────────────────────────────────────
 0  sort             2         2   4µs  ▎░░░░░   4%   28 B  interp
 1  aggregate        4         2  82µs  ████▋░  77%   28 B  interp
 2  hash_join        4         4   3µs  ▎░░░░░   3%   76 B  interp
 3  filter           6         4  18µs  █░░░░░  17%   84 B  interp
 4  scan             6         6   1µs  ░░░░░░  <1%  126 B  interp
 5  scan             3         3   1µs  ░░░░░░  <1%   33 B  interp
───────────────────────────────────────────────────────────────────
total: 2.92 ms, 2 rows out
bottleneck: aggregate (op 1), 77% of operator time — compute-bound (aggregate)
operators: 108µs of 2.9ms wall clock (4%); 2.8ms elsewhere (planning, optimization, admission, result assembly)
```
:::

That table is one run, so on a query this small the bottleneck line can name a
different operator each time; it earns its keep on data where one operator dominates.

The `operators:` line is the same accounting `explain(analyze=True)` prints, and on a query
this small it is the important one: 108 microseconds of operator work inside a 2.9
millisecond call means nothing in the table is what you are waiting for. Read it before you
read the table, and call `RunStats.wall_clock_summary()` directly when a script needs the
same split.

:::{note}
`stats()` raises {py:exc}`BackendError <batcher.BackendError>` on a `map_batches` / ML pipeline, which runs outside the
relational engine, so there is nothing to measure per operator. Use
`explain(analyze=True)` there.
:::

### What the run cost the machine

The per-operator table says where the time went. `RunStats.usage` says what the run
consumed while it was going: CPU across every worker thread, resident-set growth, page
faults, and bytes that actually reached a disk.

```python
usage = query.stats().usage
print(usage.cpu_ms >= 0.0, usage.cores_busy >= 0.0)
```

`cores_busy` is `cpu_ms / wall_ms`, the mean cores the run kept busy. On a sixteen-core
machine a value near 1 means the plan did not parallelize, and a high value alongside a
high `invol_ctx_switches` means it parallelized fine and then fought something else on the
box for the cores. Those look identical in wall time and have opposite fixes.

This is measured across the execution as a whole rather than per operator, and that is why
it is trustworthy everywhere. The engine's streaming executor interleaves its operators, so
attributing an OS counter to any one of them would be a fabricated number; the per-operator
`cpu_ms`, `peak_rss_bytes`, and `io_read_bytes` fields on `OpStat` are therefore zero on
that tier. Read a zero anywhere here as *unmeasured*, never as *none*.

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
measurements. {doc}`/user-guide/operate/tuning/performance` covers what that changes.

## What the operator cost the machine

Rows and milliseconds say what an operator did. They don't say what it cost the machine, and
that's where most unexplained slowness lives. Alongside the timings, each operator reports
what the operating system charged it.

Two of those readings change what you should do, so `explain(analyze=True)` prints them on
the operator's line, and only when they're present:

```text
▶ aggregate   est≈1,000  actual=1,000  exact  842ms  █████▌  91%  interp  PAGING(31,204 major faults)
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

- Read `where the time went` first. If the operators are a small share of the wall clock,
  nothing in the tree is what you are waiting for, and the fix is upstream of the engine:
  fewer, larger calls, or a cached plan.
- Read `what to look at`. It appears only when there is something to act on.
- Follow the `▶` marks. On a branching plan that chain is where the time is.
- Is the filter below the join? If not, the optimizer could not see through something.
- Does `MISS` read `exact` at every level? A `10.0x over` on a join input means the
  optimizer planned for ten times the rows that arrived, and the fix for that is upstream
  (stale statistics, a predicate on a column with no stats) rather than in the join.
- Did the build side get chosen the way you would have chosen it?
- Does the bottleneck operator say `spill`? Then it is memory, not CPU.
- Does a hot expression say `interp` where you expected `jit`?

Then, and only then, start changing the query.

## See also

- {doc}`Performance </user-guide/operate/tuning/performance>`: the tuning knobs behind these measurements.
- {doc}`Caching </user-guide/operate/tuning/caching>`: stop re-running the plan you just read.
- {doc}`Troubleshooting </user-guide/operate/running/troubleshooting>`: what to do about what you found.
- {doc}`Query lifecycle </architecture/deep-dives/query/query-lifecycle>`: the stages the plan passes
  through, which is what the tree is a picture of.
- {doc}`The plan IR </architecture/deep-dives/query/plan-ir>`: the JSON document the tree is printed from, and
  the contract the Rust engine reads it under.
- {doc}`Cost model </architecture/deep-dives/adaptive/cost-model>`: how an `est≈N` becomes a join order.
- {doc}`Adaptive re-optimization </architecture/deep-dives/adaptive/adaptive-reoptimization>`: why the same
  `explain()` says something different after a run.
- {doc}`Optimizing a slow query </getting-started/tutorials/foundations/optimizing-a-slow-query>`: this checklist,
  walked end to end on a query that is actually slow.
- {doc}`Dataset API </api/relational/dataset>`: the `explain` and `stats` reference.
- {doc}`/cookbook/operations/inspecting_a_query`: reading a plan and timing a query, as a runnable script.
