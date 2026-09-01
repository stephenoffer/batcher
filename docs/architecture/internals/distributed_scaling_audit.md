# Distributed scale-out audit — where linearity stops, and why

Measured 2026-09-01 on the project's own Ray cluster: 4 x 96-core / 192 GB workers plus a head
(384 schedulable CPUs, 960 GiB), Batcher at `perf/aggregate-and-dist-merge`. The corpus is 64 M
rows in 16 zstd Parquet files (~1 GB) on shared cluster storage, four columns, 200,000 distinct
group keys. Every figure below is a **warm** number — three runs of the same shape at the same
worker count, the last one reported — because the first run of any shape pays a fleet spawn and
that cost is measured separately rather than smeared across the ladder.

The bias of this pass is the same one `competitor_technique_review.md` adopted: prefer a number
that closes off a direction to a plausible technique that opens one.

## The corpora, so the numbers can be re-taken

Nothing below is reproducible without these, and neither corpus is committed — they are
generated, not fixtures. Both are plain Parquet (zstd), four columns, written to shared cluster
storage:

| | small | large |
|---|---|---|
| rows | 64 M (16 files x 4 M) | 512 M (64 files x 8 M) |
| on disk | ~1.0 GB | ~8.4 GB |
| `k` | `randint(0, 200_000)` | `randint(0, 5_000_000)` |
| `g` | `randint(0, 64)` | `randint(0, 64)` |
| `v` | `random()` (float64) | `random()` |
| `w` | `randint(0, 1_000_000)` | `randint(0, 100_000_000)` |
| seed | `default_rng(7)` | `default_rng(1000 + file_index)` |

The **file count is load-bearing** and is the variable the retraction below turns on:
`partition_descriptors` caps the map at the splits the source actually has, so a 64-file corpus
gives 64 map sources at *any* worker count above 16, and every measurement here that did not
explicitly lower `map_partition_multiplier` was taken at exactly 64 sources. A corpus with a
different file count is a different experiment, not a re-run of this one.

Lower cardinalities are derived rather than regenerated — `k % 100`, `k % 10_000`,
`k % 1_000_000` over the large corpus — which is what holds the input size and the source count
fixed while only the group count moves.

The cluster is four `96cpu-192gb` workers plus a head: 384 schedulable CPUs, 960 GiB, one
availability zone. Worker counts are always passed explicitly (`num_workers=`), because the
automatic fan-out was being rewritten by another session throughout this pass — see the
provenance note below.

## What the ladder actually shows

`group_by(k).agg(sum(v), count(v))`, warm, wall time and the two phases it decomposes into:

| workers | warm total | map barrier | reduce |
|---|---|---|---|
| 1 | 772 ms | 682 ms | 63 ms |
| 2 | 527 ms | 412 ms | 87 ms |
| 4 | 391 ms | 277 ms | 86 ms |
| 8 | 484 ms | 239 ms | 213 ms |
| 16 | 486 ms | 233 ms | 217 ms |
| 32 | 288 ms | 152 ms | 99 ms |
| 64 | 421 ms | 104 ms | 264 ms |

Read the two right-hand columns rather than the total. **The map stage scales**: 682 ms to
104 ms is 6.5x across a 64x fan-out, and it is sublinear mostly because 16 files cap how finely
the source can split. **The reduce anti-scales**: 63 ms at one worker to 264 ms at 64, and it
crosses over the map at eight workers. Past that point every worker added makes the query's
dominant phase slower, which is why the total flattens at ~2.9x and then wobbles.

Four shapes over the same corpus, warm, median of 5, **every timed run given a distinct
plan** so the result cache cannot serve it (see the note below). Reported after the Finding 2
fix, so this is the current state rather than the one the findings were found in:

| shape | 1 worker | 4 | 16 | 64 | speedup |
|---|---|---|---|---|---|
| `join` + `group_by` | 7,981 ms | 3,023 ms | 650 ms | 560 ms | **14.3x** |
| `sort` + `limit` | 248 ms | 127 ms | 115 ms | 76 ms | 3.3x |
| `group_by(200k)` | 746 ms | 397 ms | 405 ms | 274 ms | 2.7x |
| `distinct(200k)` | 421 ms | 351 ms | 408 ms | 299 ms | **1.1x** |

The join is what the distributed path is built for and it behaves like it. `distinct` is the
outlier and the reason is not its shuffle: at 64 workers it spends 119 ms in the map, 107 ms in
the reduce and **73 ms on the driver**, against 554/30/29 at one worker. Its scan parallelises
and everything else gets worse, so a shape whose single-worker time is already under half a
second has little left to win.

**A methodological trap worth recording, because it invalidated a whole ladder.** The first
`sort` ladder here read 31 ms at one worker rising to 70 ms at 64, which looks like a dramatic
anti-scaling result and is not a sort at all: `carbonite.cache` had served every repeat, because
a 1,000-row top-N result fits the result cache and an identical query is an identical cache key.
The aggregate ladders were unaffected — a 200,000-row result does not fit, and identical
(250 ms) and varied (257 ms) runs measured the same — but nothing about the timings said which
was which. Every ladder in this document now varies one literal per run.

## The same ladder at 8.4 GB, which is the one to read

Everything above runs on 64 M rows in 1 GB. At 64 workers that is ~16 MB of input each, so the
coordination floor dominates and every shape converges on ~300 ms whatever the fan-out — which
looks like a scaling wall and is mostly over-provisioning. Re-run on **512 M rows / 8.4 GB /
5 M distinct keys** (64 files, so the split count no longer caps the map either), warm, median
of 3:

| workers | total | map | reduce |
|---|---|---|---|
| 4 | 8,744 ms | 7,820 ms | 848 ms |
| 8 | 4,437 ms | 3,285 ms | 1,006 ms |
| 16 | 3,470 ms | 2,302 ms | 1,049 ms |
| 32 | 3,361 ms | 1,626 ms | 1,588 ms |
| 64 | 3,551 ms | 1,131 ms | **2,461 ms** |

**The map scales and the reduce does not.** 7,820 ms to 1,131 ms is 6.9x across a 16x fan-out,
which is the scan and partial-aggregate path behaving as intended. Over the same range the
reduce gets **2.9x worse**, overtakes the map between 32 and 64 workers, and turns the total
back up. Finding 2's fix does not reach this: at 5 M groups `_busy_floor` computes
`min(64, 5e6 // 50,000) = 64`, so the floor still hands the exchange one reducer per worker and
`mappers x reducers` is 4,096 streams again.

Swept directly at 64 workers, holding everything else fixed:

| reducers | median | vs. current |
|---|---|---|
| 64 (what the engine picks) | 3,784 ms | — |
| 32 | 3,093 ms | 1.22x |
| 16 | **2,996 ms** | **1.26x** |
| 8 | 3,048 ms | 1.24x |
| 4 | 3,216 ms | 1.18x |

A clear interior optimum with a broad basin from 8 to 32, and the engine's own choice is the
worst point on the curve.

The same sweep at **32** workers, same corpus, separates the two explanations:

| reducers | w=32 | w=64 |
|---|---|---|
| one per worker (the engine's choice) | 3,883 ms | 3,784 ms |
| 16 | **3,687 ms** | **2,996 ms** |
| 8 | 3,737 ms | 3,048 ms |
| penalty for one-per-worker | 1.05x | **1.26x** |

Two things follow, and both are stronger than the 64-worker sweep alone supports. **The optimum
is the same 16 at both fan-outs**, so it is a property of the data rather than of the cluster —
the reducer count should stop following the worker count once the data has enough reducers.
And **the penalty for exceeding it grows with the fan-out**, 1.05x to 1.26x for a doubling,
which is the `mappers x reducers` stream count showing up as a cost: 1,024 streams at 32
workers against 4,096 at 64.

### Retracted: the cardinality sweep below was an ordering artefact

The sweep that follows, and the "eight is the optimum at every cardinality" conclusion drawn
from it, **do not reproduce and should not be relied on.** They are kept because the way they
failed is the useful part.

| groups | r=2 | r=4 | r=8 | r=16 | r=32 | r=64 |
|---|---|---|---|---|---|---|
| 100 | 406 | 415 | 385 | 408 | 414 | 424 |
| 10,000 | 622 | 665 | 595 | 630 | 624 | 684 |
| 1,000,000 | 1,751 | 1,778 | 1,706 | 1,819 | 1,793 | 1,754 |
| 5,000,000 | 3,402 | 3,117 | 2,941 | 2,984 | 3,143 | 3,480 |

Every one of those rows ran its caps **in a fixed order**, `[2, 4, 8, 16, 32, 64]`, three timed
reps each. Anything that drifts over the life of the process — fleet warmth, page cache, a
co-tenant arriving — therefore maps onto cap position rather than onto the cap, and an interior
minimum appears where the drift happened to trough. Re-measured **round-robin** (one rep of each
cap per round, five rounds, every cap warmed first), 100 groups, same corpus and fan-out:

| reducers | median | min | max |
|---|---|---|---|
| 1 | **407 ms** | 385 | 443 |
| 2 | 418 ms | 387 | 468 |
| 8 | 487 ms | 458 | 505 |
| 16 | 709 ms | 587 | 790 |

Monotone: at low cardinality fewer reducers is strictly better, and `r=8` — the claimed optimum
— is 20% *worse* than `r=1`, not 5% better. A separate fixed-order run had already read `r=8` at
480 ms against the sweep's 385 ms for the same configuration, which is the contradiction that
prompted the re-measurement.

What survives, and what does not:

- **The Finding 2 fix is validated by this.** `_busy_floor` gives 1 reducer at these
  cardinalities, and 1 is the measured optimum. The engine's own choice measured 385 ms against
  `r=8`'s 487 ms.
- **The large-corpus ladder survives.** Its phases are measured *inside* each query, so drift
  moves the map and the reduce together — and there the map fell 6.9x while the reduce rose
  2.9x over the same runs, which no ordering artefact produces.
- **The high-cardinality end has a genuine interior optimum**, and reporting it as monotone
  was a second error in this section — made by extrapolating from a round-robin run that
  started at `r=8` and never looked below it. Round-robin at 5 M groups, the upper arm and the
  lower arm (separate runs, so read each arm's ordering rather than across them):

  | reducers | 1 | 2 | 4 | **8** | 32 | 64 |
  |---|---|---|---|---|---|---|
  | median | 3,979 | 3,586 | 3,231 | **3,040** | 4,948* | 5,455* |

  (*upper arm, taken while the cluster was busier; `r=8` read 4,306 ms in that run.) The curve
  falls to 8 and rises on both sides, so the shape is U-shaped here and monotone-increasing at
  100 groups where the minimum is `r=1`. That is a coherent picture rather than a puzzling one:
  with almost no partial state there is nothing to parallelise and one reducer wins, and with a
  lot of it the merge is worth spreading until the `mappers x reducers` stream count starts
  charging more than the parallelism returns.

  The engine picks 64 here, which is the worst end of the measured range — 1.27x off `r=8`
  within the run where both were measured. That win is real and unclaimed.
- **The falsified fan-in mechanism stays falsified**, now for a second reason: it was fitted to
  a table that turns out to be an artefact.

### The rule that survived a prediction and then died — retracted

Two interleaved optima, `r=1` at 100 groups and `r=8` at 5 M, are both fitted by bounding a
reducer's share of the partial state at about **625,000 groups** — the
`ceil(rows / target_rows_per_task)` shape `aggregate_reducer_count` already uses, with a target
~6.4x smaller than the configured 4 M. Used to predict a third cardinality it had not seen, it
held: at 1 M groups it requires 2, and round-robin measurement put the optimum at 1-2
(`r=1` 1,752 ms, `r=2` 1,763 ms, `r=4` 1,793 ms, `r=16` 1,919 ms). It also survived two
invariance checks — the optimum stayed at 8 across `shuffle_fan_in` 8 and 16, and across 16 and
64 workers (`r=2` 3,583 ms, `r=8` 3,258 ms, `r=16` 3,300 ms at 16 workers), which ruled out its
being a per-node or per-core quantity in disguise.

**It is still wrong, and one more measurement showed why.** Every run above was on the 64-file
corpus, where `partition_descriptors` caps the map at 64 sources whatever the worker count — so
`sources` was 64 in all of them and could not be attributed. Setting
`distributed.map_partition_multiplier = 1` at 16 workers gives 16 sources instead, holding data,
cardinality and fan-out fixed:

| reducers | 2 | 8 | **16** |
|---|---|---|---|
| median | 3,015 ms | 2,904 ms | **2,727 ms** |

The optimum moves from 8 to 16. Quartering the sources doubles it, which is `r*` proportional to
`1 / sqrt(sources)` — the shape a cost of `merge_work / r + stream_cost x sources x r` has, and
the shape suggested by the `mappers x reducers` stream count all along. **625,000 was never a
groups-per-reducer figure.** It was a groups-per-reducer figure *at 64 sources*, and it would be
wrong on any corpus that splits differently — which is most of them.

What this costs and what it buys:

- **The shippable one-constant change is withdrawn.** `_MIN_GROUPS_PER_REDUCER = 625_000` fits
  four measurements and is confounded with a fifth variable that was never moved. Shipping it
  would have put a corpus-shape artefact into the sizing of every aggregate.
- **The same caveat attaches to the 50,000 already shipped** in Finding 2, but the change is
  safe under it for a structural reason rather than a lucky one. `_busy_floor` returns
  `min(floor, rows // 50_000)`, which is `<= floor` by construction, and it enters the same
  `max(...)` the memory rule does — so the shipped count is **never higher** than the count
  before it, at any cardinality or floor. Checked exhaustively over the cross-product of eight
  cardinalities (4 to 50 M groups) and six floors (1 to 256): no configuration asks for more
  reducers than the old rule did. Since fewer reducers measured faster or level in every
  interleaved comparison taken, a confounded constant here can cost an opportunity and cannot
  cost a regression. The number is still not derived and should not be quoted as one; what it
  is, is monotone in the safe direction.

  The 16-source run above is a good illustration of both halves at once: there
  `_busy_floor(16, 5e6) = min(16, 100) = 16`, which is exactly the measured optimum — right,
  and right by accident, since nothing in the rule knows the source count that made 16 correct.
- **The direction is now well established across every design tried**: one reducer per worker
  was worse than a data-derived count in every cardinality, fan-in, worker count and source
  count measured. What is not established is the count, and the missing term is `sources`, which
  neither `aggregate_reducer_count` nor `target_rows_per_task` currently references at all.

**No formula is shipped here.** Two were proposed in this section and both are now retracted —
the first by an ordering artefact, the second by a confound with the source count that four
prior measurements and one successful out-of-sample prediction all failed to expose. The
surviving statement is a direction, not a number, and the next person to work on this should
start by putting `sources` into the model rather than by re-fitting a constant.

## Finding 1 — a warm fleet reserves the whole cluster, against every other process

The highest-impact result in this pass, and the one that is a *cluster* property rather than a
query property.

`_even_cpu_share` divides the cluster's **nameplate** cores by the worker count, so the fleet a
query reserves is `workers x (total / workers)` — the entire cluster, by construction, at every
fan-out. Two Batcher processes therefore each conclude they own the machine. Reproduced with two
drivers on the idle fleet above, one at `num_workers=64` and one at 32:

```text
placement group did not form within the timeout; falling back to default scheduling
  workers=64 strategy=SPREAD reason="the cluster is short of free capacity: 64 outstanding at
  6 CPU, 0.0 GB each, and 1 candidate node(s) have 95 CPU free between them"
shuffle fleet came up narrower than requested  placed=60 requested=64 waited_s=120.0
```

with `ray status` reading `360.0/384.0 CPU (0.0 used of 4.0 reserved in placement groups)` —
**360 of 384 cores reserved and none of them working**. Both queries stalled for the full
`placement_timeout_s` (120 s), then fell back to default scheduling and ran. The second tenant
produced no output at all for the first 100 seconds of a workload that takes about three.

A warm fleet then *keeps* that reservation for `session_fleet_idle_s` (30 s) after its query
ends. `yield_session_fleet` exists to hand it back, and it is the right mechanism, but it is
reachable only from within the process that holds the fleet: `_SESSION` is a module global, so a
second Batcher driver has no way to ask for it and no way to be noticed asking.

**What was built, and is not committed.** The explicit-`num_workers` path should thin its grant
against free capacity through `_placeable_grant`, which the automatic fan-out path has always
done and which the explicit path — the one every benchmark and integration test in this repo
takes — never did. It is a documented no-op on an idle cluster, and it closes the case where the
co-tenant is holding *part* of each node. The change is one `elif` in `execute_distributed` and
two composition tests in `tests/unit/test_placeable_grant.py`; both were left in the working tree
rather than committed, because another session held 87 staged and further unstaged lines in
`dist/executor.py` (a rewrite of `_numa_sliced` and the fan-out constants) for the whole of this
pass, and `git commit` bounds paths rather than hunks — committing the file would have carried
their unfinished work under this change's message.

**What did not, and why.** It does not fix the Batcher-against-Batcher case above, and it cannot:
thinning preserves the worker count, so `workers x grant` is still the whole cluster whenever
`_even_cpu_share` sized it. Fixing that needs one of two things, and both are policy decisions
rather than repairs:

- sizing a fleet to a *share* of the cluster when another tenant is present, which needs a
  cross-process notion of "another tenant" that does not exist today; or
- releasing an idle warm fleet on another process's unmet demand, which needs the demand signal
  to cross the process boundary.

A third, smaller change was built and **reverted**: making `_placeable_grant` return the thinnest
grant instead of the nameplate one when nothing tiles. It converts the 120 s stall into an
immediate narrow run, and `tests/unit/test_placeable_grant.py::test_a_genuinely_full_cluster_keeps_the_grant`
already pins the opposite, for a measured reason — a one-core fleet gets cached and the process
stays on one-core workers for the rest of its life (`_FLEET_THINNESS_TOLERANCE` records TPC-H
sf10 going from 27 s to over 20 minutes that way). The `_fleet_is_too_thin` respawn added later
weakens that objection but does not obviously retire it, and re-deciding it wants the
measurement, not an argument.

## Finding 2 — the reducer floor made the exchange O(workers squared) on a small aggregate — FIXED

`aggregate_reducer_count` sizes an aggregate's reduce by its **learned group count**, which is
the right basis: an aggregate exchanges partial state, not rows. For this corpus that is
`ceil(200,000 / target_rows_per_task)` = 1 reducer. The count was then floored at the worker
count, because a bucket is reduced by exactly one worker and fewer buckets than workers idles
the rest — measured, on a 9-node cluster, at 0.65 s / 1.05 s / 4.47 s for 2 / 4 / 8 workers.

The floor won here, so a 64-worker run got **64 reducers against 64 map sources**: 4,096
exchange streams carrying about 5 MB of partial state, roughly 1 KB per stream. The combiner
tree's cost tracked that product rather than the data — 100 ms at 32 workers against 291 ms at
64, for the same input.

Both rules are individually well-founded and they disagreed at low cardinality. The fix bounds
the *floor* by whether the groups can keep those workers busy (`_busy_floor`, at 50,000 groups
per reducer), leaving the high-cardinality rule and the case the floor was added for untouched.
Measured warm, median of seven, every case checked against DuckDB:

| groups | workers | before | after | speedup |
|---|---|---|---|---|
| 64 | 64 | 302 ms | 91 ms (1 reducer) | 3.30x |
| 200,000 | 8 | 461 ms | 416 ms (4) | 1.11x |
| 200,000 | 16 | 442 ms | 389 ms (4) | 1.14x |
| 200,000 | 64 | 348 ms | 249 ms (4) | 1.40x |
| 1,000,000 | 64 | 880 ms | 434 ms (20) | 2.03x |

Note what the 64-group row is. `aggregate_reducer_count`'s own docstring already said a
60M-row-to-4-group aggregate "needs one" reducer, and the module's test asserted exactly that —
but the test passed the default `floor=1` while every real caller passes `floor=workers`, so
the documented behaviour was never exercised where it applies. The claim was true of the
function and false of the engine.

The end-to-end effect on the ladder this document opens with, same corpus and cluster: 64
workers went from 421 ms to 274 ms, and the reduce phase from 264 ms to 136 ms, so the curve is
now monotone across the top end instead of turning back up past eight workers. It is still not
linear — the map stage is capped by a 16-file source — but the reduce no longer *anti*-scales.

## Finding 3 — the combiner tree builds an O(reducers x sources) structure on the driver

`_tree_reduce` materializes `frontier` and `fallbacks` as dense dicts of one entry per
`(reducer, source)` pair before it launches anything, and each entry constructs a `ShuffleTicket`
(a frozen dataclass whose `__post_init__` formats its wire string). Driver-side cost, measured
standalone:

| reducers x sources | pairs | build |
|---|---|---|
| 16 x 32 | 512 | 0.9 ms |
| 64 x 128 | 8,192 | 14.2 ms |
| 128 x 256 | 32,768 | 65.6 ms |
| 256 x 512 | 131,072 | 335.8 ms |

Ticket construction is 46 ms of that 65.6 ms; the tuples and the empty fallback lists are the
rest. At the fan-outs reachable today this is tens of milliseconds and not the reason the reduce
anti-scaled — Finding 2 was. It is on the list because it is *quadratic in the fan-out* and it is
control-plane work, which `.claude/rules/architecture.md` reserves for the data plane.

**Fixing Finding 2 shrank this one rather than exposing it**, which is worth recording because
the opposite was expected. The frontier is `reducers x sources`, and bounding the reducer count
by the work available took the 64-worker aggregate from `64 x 64` pairs to `4 x 64` — so the
term this section is about is now a few hundred objects, not eight thousand. It still binds for
a *raw-row* shuffle (join, sort, window), where `row_shuffle_reducer_count` may only raise the
count above one bucket per worker and the sources scale with the fleet. That is where to measure
it next, not on an aggregate.

The cheap half (not allocating `reducers x sources` empty lists when replication is off) is worth
about 2 ms of the 65 and was judged not worth the churn. The half that matters is shipping the
*rule* rather than the enumeration: a combine task for bucket `r` over sources `[s0..s7]` at stage
`st` needs exactly the tickets `(plan, st, si, r)`, so sending `(st, r, [s0..s7])` and letting the
worker build its own eight would move the whole quadratic term off the driver and shrink the
pickle from 131,072 ticket objects to a list of integers. That is a change to `combine_publish`'s
worker protocol, shared with the recovery and replication paths that index replicas positionally,
and it was not landed blind on a cluster too busy to measure it on.

## What these numbers were measured against

Every figure in this document was taken from a **working tree, not from `HEAD`**, and this
repository is written by several agents at once. That is worth stating precisely, because it
bounds what the numbers mean.

The files the aggregate path reads — `adaptive_sizing/sizing.py`, `flight_aggregate.py`,
`executors/ray_runtime/reduce.py` — were clean throughout, so the reduce and reducer-count
results are measurements of committed code plus this pass's own changes. The files that decide
**worker sizing and fleet lifetime** were not: `dist/executor.py` carried another session's
rewrite of `_numa_sliced` and the fan-out constants for the whole pass, and `fleet/_fleet.py`
carried 107 staged lines. So the *absolute* wall times here were produced under a fan-out policy
that is not the one at `HEAD`, and they should be re-taken before anyone quotes them as
Batcher's numbers. The *relative* comparisons are unaffected: every A/B in this document varied
one thing inside a single process against everything else held fixed.

One measurement had to be discarded outright. A join ladder on the 8.4 GB corpus produced
`BroadcastOutputTooLarge: broadcast probe output reached 0.0 GiB on this node, over the 0.0 GiB
bound` 56 times, which reads as two defects at once — a bound small enough that any output
trips it, and a message that renders both sides as `0.0` and so says nothing. Neither is a
defect in Batcher: `BroadcastOutputTooLarge` does not exist at `HEAD`. It is 242 uncommitted
lines of another session's in-flight broadcast rewrite, and the join ladder was measuring that
rewrite rather than the engine. The numbers were dropped and the observation is recorded here
only as the reason they were.

The general form is worth keeping: **on a shared tree, a surprising measurement is a reason to
check `git status` on the files the path reads before it is a reason to open the code.**

## Concurrent queries in one process

The rest of this document measures one query at a time. `execution.max_concurrent_queries`
defaults to `0`, so `ConcurrencyLimiter` is off, and `carbonite/policies/concurrency.py` records
a *single-node* collapse behind it — 124 QPS to 88 going from 1 client to 16. The distributed
path had not been measured. Eight threads in one process, each running the 64 M-row aggregate
at `num_workers=8` over the shared session fleet:

| clients | QPS | p50 | p95 |
|---|---|---|---|
| 1 | 1.39 | 441 ms | 442 ms |
| 2 | 3.19 | 624 ms | 652 ms |
| 4 | 3.94 | 1,001 ms | 1,053 ms |
| 8 | 4.13 | 1,940 ms | 1,974 ms |

It **saturates rather than collapsing** — throughput flattens at ~4 QPS from four clients on,
while p50 grows 4.4x. No errors at any point.

**Most of that is not a defect, and the test design is why.** Pinning `num_workers=8` fixes the
fleet at eight actors however many clients arrive, so eight concurrent queries are eight times
oversubscribed on the same eight workers while the cluster's other 376 cores sit idle. A fair
reading is that this measures fleet sharing, not the engine's concurrency ceiling.

What it does isolate is a real serialization point, and it is the one `_session_fleet_alive`
was already suspected of being. `_acquire_session_fleet` pings every actor for liveness
**inside `_SESSION_LOCK`**, so arriving queries queue behind each other's pings. Attributed:

| | 1 client | 8 clients |
|---|---|---|
| query p50 | 440 ms | 1,708 ms |
| `acquire_fleet` p50 | 2.6 ms | **49.2 ms** (max 380.5) |
| liveness ping p50 | 2.6 ms | 15.1 ms |

`acquire_fleet` grows 19x and its tail reaches 380 ms. But it is 49 ms of a 1,708 ms query —
**about 3% of the added latency, not the cause of it**, and saying otherwise would be reading a
confirmed prediction as a confirmed explanation. The reason it is still worth recording is that
its cost is `O(workers x concurrent queries)` on both axes: the ping was 2.6 ms across 8 actors
here and 8.7 ms across 64 in the single-client measurement above, and only the first factor was
being paid then. A wider fleet under real concurrency is where this stops being 3%.

The fix is the one already described and not made — cache the liveness verdict for a short TTL,
so back-to-back queries skip the fan-out — and it remains blocked on `dist/fleet/_fleet.py`,
which carried another session's staged work for this entire pass.

## Ruled out

Kept because the ratio of already-built to genuinely-missing is the most useful thing this
kind of pass produces, and a candidate that was measured and lost costs the next reader
nothing to skip.

**Batching the metrics drain's `ray.get`.** `metering.py::drain_worker_metrics` reads its
per-worker documents with `for ref in pending: ray.get(ref)` — one blocking call per worker,
where `shuffle_replication.py` settles the same shape concurrently with a `ray.wait` first.
It reads like an obvious O(workers) round-trip bug, and `drain_metrics` does cost 5.6 ms at 8
workers rising to 15.0 ms at 64, which fits. Measured directly against 64 and 128 same-sized
remote tasks, warm: serial 7.3 ms against wait-then-get 8.7 ms at 128 refs — no difference.
The first 64-ref reading did show 47 ms against 8 ms, which is what the change would have been
justified on; re-running it warm showed that figure was scheduler cold-start, not the pattern.
Refs that all become ready at about the same time cost the same either way, and these do,
because every worker drains after the same barrier. The cost is the 64 actor RPCs themselves,
which is what closing the Core-to-Kyber loop on a 64-worker fleet is worth.

**Skipping the locality probe for a small reduce.** `_locality_reducer_hosts` asks all
`workers` mappers for their published bucket sizes so reducers can be placed near their data —
64 actor RPCs, and after Finding 2 it is spending them to place 4 reducers. Disabling it
measured 305 ms against 322 ms, a 5% win that looked like a disproportionate probe. It did not
reproduce: interleaved off/on/off/on runs read 300.3 / 304.1 / 303.7 / 304.3 ms, so the gap is
noise and the probe's RPCs overlap with work that is happening anyway. The first reading was
one unreplicated 5% difference, which is not enough to change scheduling code on.

**The remaining reduce time is not obviously coordination.** After Finding 2 the 64-worker
aggregate's reduce is ~136 ms for a two-level combiner tree over 4 buckets and 64 sources,
moving about 307 MB. That is roughly 2.3 GB/s aggregate through the tree, so it is closer to
network-bound than to launch-bound, and shaving driver round trips would not obviously move it.

## What landed

- The shuffle reduce's submit-ahead window **slides** instead of stepping. It was a chunked
  barrier — launch `window`, `ray.get` all of them, launch the next `window` — which bounds the
  scheduler queue just as well and serializes the stage behind its slowest task once per chunk.
  A 64-worker aggregate over 128 map partitions runs 1,024 combine tasks through a 256-deep
  window, so that was four barriers where one slow bucket held 255 actors idle. `map_barrier`
  had filled from a `ray.wait` on one completion since it was written, and this function's own
  docstring already called itself that barrier's reduce-side twin; it was the twin in everything
  except the pipelining. Results are still returned in submission order, so nothing above it can
  see which shape it is.
- The aggregate reduce's worker floor is bounded by the work available (Finding 2), which is
  the largest single win in this pass and the one that removed the anti-scaling.

The explicit-`num_workers` thinning (Finding 1) is written and green but uncommitted, for the
file-contention reason recorded above.

## What this pass deliberately did not do

- No change to the combiner-tree wire shape (Finding 3). It wants a measurement on a raw-row
  shuffle, which is where it still binds.
- No cross-process fleet arbitration (Finding 1). It is the right answer and it is a feature.
- No Rust data-plane work. Everything above is control plane.
