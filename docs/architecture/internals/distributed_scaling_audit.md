# Distributed scale-out audit — where linearity stops, and why

Measured 2026-09-01 on the project's own Ray cluster: 4 x 96-core / 192 GB workers plus a head
(384 schedulable CPUs, 960 GiB), Batcher at `perf/aggregate-and-dist-merge`. The corpus is 64 M
rows in 16 zstd Parquet files (~1 GB) on shared cluster storage, four columns, 200,000 distinct
group keys. Every figure below is a **warm** number — three runs of the same shape at the same
worker count, the last one reported — because the first run of any shape pays a fleet spawn and
that cost is measured separately rather than smeared across the ladder.

The bias of this pass is the same one `competitor_technique_review.md` adopted: prefer a number
that closes off a direction to a plausible technique that opens one.

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

A `sort(w).limit(1000)` ladder over the same corpus flattens the same way, at 16 workers.
Those figures are **cold** — 6,817 ms at one worker to 2,020 ms at 32, 3.4x — because each rung
asked for a different worker count and a fleet that is too narrow for the request is torn down
and respawned rather than grown. They are quoted for the shape of the curve, not against the
warm table above.

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
