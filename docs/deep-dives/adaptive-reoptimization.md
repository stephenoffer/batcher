# Adaptive re-optimization

Every cost-based optimizer runs on estimates, and estimates are wrong. A cardinality
error compounds multiplicatively up a join tree, so a 10× miss at the leaves becomes a
1000× miss at the root, and the plan you get is not merely suboptimal. It is planned for
a different query.

| System | When it re-plans |
|---|---|
| DuckDB | never — it optimizes once and lives with it |
| Spark AQE | at stage boundaries, which means only where a shuffle already forced a materialization |
| Batcher | at every pipeline breaker, on measured cardinalities |

Batcher's answer: execute the plan one *pipeline breaker* at a time. A breaker has to
materialize its input anyway, so at that point the engine is not estimating the size of
what it just processed. It counted it. Splice the measured result back in as a source
with an exact row count, and re-plan the rest.

## The loop

It lives in `python/batcher/api/adaptive.py`, in the `api` layer, and it is entirely
Python. There is no Rust component; Rust returns per-operator metrics, and the control
plane does the segmenting.

Kyber optimizes the logical plan once, up front. Then:

1. `_lowest_breaker(plan)` finds a breaker whose inputs are all breaker-free: the first
   thing that must materialize.
2. `_estimate_rows` asks Kyber's `CardinalityEstimator` how big it thinks that stage's
   output will be. This is the prediction under test.
3. `_run_stage` runs it through the full Kyber → Carbonite → Core sequence and gets back a
   table.
4. `_stage_row_count` reads `num_rows`. That is the measurement.
5. The stage node is replaced with a `Scan` over the materialized result
   (`_replace(plan, target, Scan(sid, schema))`), and the loop repeats.

```text
   the optimized logical plan
            │
   ┌───────►│
   │        ▼
   │   _lowest_breaker(plan)        a breaker whose inputs are all breaker-free
   │        │
   │        ▼
   │   _estimate_rows(stage)        Kyber's PREDICTION — the thing under test
   │        │
   │        ▼
   │   _run_stage(stage)            Kyber → Carbonite → Core, one full round
   │        │
   │        ▼
   │   _stage_row_count(table)      the MEASUREMENT. counted, not guessed.
   │        │
   │        ├─── inside the band? ──────►  break: run the whole residual plan in one shot
   │        │      q-error ≤ 1 + optimizer.reoptimize_error   (default 2.0 → a 3× band)
   │        │
   │        ▼  outside the band
   │   replace the stage node with Scan(the materialized result)
   │        │
   └────────┘
```

:::{important}
Step 5 is what makes this work. The spliced source is an `InMemorySource` with a known row
count, so its cardinality carries `Provenance.EXACT`. The next iteration is not merely
re-optimized; it is optimized on a plan where one subtree's size is *known*. Join order,
build-side choice, and broadcast eligibility above that point are decided against a fact rather
than a guess.
:::

The breakers are the module constant `_BREAKERS = (Aggregate, Sort, Distinct, Window,
Limit, Join, Union)`. A `Filter → Project → MapBatches` chain never segments, because it
streams; there is nothing to materialize and nothing to count.

## The trigger, and why its polarity is backwards from what you'd expect

`_estimate_accurate` compares estimate against actual as a *symmetric q-error*:

```text
max(actual/estimate, estimate/actual) <= 1.0 + optimizer.reoptimize_error
```

with `reoptimize_error` defaulting to `2.0`, so the tolerance band is a 3× q-error in
either direction. (Relative error was tried first and abandoned: `|actual − est| / est` is
bounded by 1 for any over-estimate, so it called every over-estimate accurate.)

:::{note}
Staging is not *triggered* by a bad estimate. Staging is the default, and a *good* estimate
stops it: if a stage's measured rows land inside the band, the loop breaks and runs the whole
residual plan in one shot. Inaccuracy is what keeps the loop paying for another round. That is
backwards from what most readers of the code expect on their first pass.
:::

That inverts the usual framing but it is the right economics. Each stage costs roughly
20–40 ms of control plane. If the estimator is already tracking reality, buying another
measurement is pure overhead. If it is not, every additional measurement is worth more
than it costs.

The consequence to be honest about: on a query whose estimates are accurate, the "adaptive
loop" adapts at exactly one breaker and then behaves like a static optimizer. The claim
"re-plans the rest of the query at every breaker" is true only while estimates keep
missing, which, to be fair, is the case the mechanism exists for.

## When it turns on

`adaptive="auto"` is the default on `collect()`. `resolve_adaptive` turns it on when:

- the plan requires staging on the distributed path (a 3+-table star join, which the
  one-shot dispatcher cannot handle); or
- the plan has a join, its total scan rows clear `_ADAPTIVE_MIN_INPUT_ROWS` (20M, a
  hard-coded module constant, not a config knob), and some join operand is non-streamable
  with a merely-default-provenance estimate; or
- the `MetadataHub` says it has helped for this plan signature before
  (`learned_adaptive_helps`: the estimate flipped outside the band on ≥25% of at least 3
  prior runs).

So on a small query, adaptive is off and stays off. That floor exists because a
sub-second query cannot afford a 40 ms staging round-trip to learn something it could have
guessed. Below 20M rows you can still force it with `adaptive=True`.

## Two feedback loops, one measurement layer

The adaptive loop's measurement is just a row count. It is not the only thing Core
measures, and the second loop is the more consequential one over time.

::::{tab-set}
:::{tab-item} Within-query (this page)
```text
measurement:  num_rows of the stage just materialized
consumer:     the next iteration of the staging loop
effect:       the residual plan is re-optimized against an EXACT row count
lifetime:     one collect()
```
:::

:::{tab-item} Across-query (the metadata loop)
```text
measurement:  ExecMetrics { ops: Vec<OpMetric> } from execute_plan_metered
              rows_in, rows_build, rows_out, elapsed_ns, cpu_ns, threads,
              peak_bytes, result_bytes, spilled, spill_bytes, peak_rss_bytes,
              backend ("interp" | "jit" | "interp+jit")
consumer:     the MetadataHub, via core/executor.py::_record_op_feedback
effect:       per-signature cardinality correction, calibrated cost coefficients,
              memory sized from measured peaks
lifetime:     the process, or forever with a durable backend
```
:::
::::

They share the measurement plumbing and nothing else. The across-query half is what makes plans
improve the more a query shape is run; see [Learned metadata](learned-metadata.md).

## Watching it work

`explain(analyze=True)` runs the query and renders estimate against actual per operator.

```python
import batcher as bt

rows = 4000
ds = bt.from_pydict({"g": [i % 7 for i in range(rows)], "x": [float(i) for i in range(rows)]})
q = ds.filter(bt.col("x") > 100).group_by("g").agg(s=bt.sum("x"))
print(q.explain(analyze=True))
```

:::{dropdown} The `analyze=True` output, estimate against actual
```text
aggregate                       est≈7 actual=7 (1.0x)  0.2ms (1%)  cpu=100%  out=112B  interp
  filter                        est≈3,500 actual=3,899 (1.1x)  1.1ms (4%)  cpu=100%  out=0B  jit
    scan                        est≈4,000 actual=4,000 (1.0x)  0.0ms (0%)  cpu=100%  out=62KB  interp
```
:::

The `(1.1x)` on the filter is the q-error for that operator. The filter's estimate came
from the Selinger default for a range predicate (a third of rows), which happened to land
close. A `LIKE '%x%'` or a correlated conjunction is where you see the number blow out,
and that is exactly what gets recorded and corrected next run.

## Limits worth stating

Re-optimization is *between* stages, never mid-operator. A stage runs to completion. There
is no runtime plan switch inside a hash join, no operator-level abort, no bail-out halfway
through a sort. If a join's build side turns out to be 100× the estimate, you find out when
the build finishes, not while it is running.

Only the seven breaker types segment. A badly mis-estimated selective filter is measured
only when it feeds a breaker, which for a pure `scan → filter → collect` is never.

And the granularity has a floor: staging materializes. A plan with many small breakers
pays the control-plane round-trip at each one, which is why the accuracy early-exit exists
and why the whole mechanism is gated behind a 20M-row input floor.

## Code map

| Concern | File |
|---|---|
| The loop, segmentation, splicing | `python/batcher/api/adaptive.py` |
| Breaker-free test | `python/batcher/plan/logical/transforms.py::is_streamable` |
| Per-operator metrics (Rust) | `crates/bc-interp/src/metrics.rs` |
| Metric → feedback transcription | `python/batcher/core/executor.py` |
| The `reoptimize_error` knob | `python/batcher/config/config.py::OptimizerConfig` |

## See also

:::{seealso}
- [Architecture](../architecture/index.md): the contract loop this closes — Core measures, Kyber decides
- [Kyber optimizer](../internals/kyber.md): the pass pipeline that runs at each stage
- `docs/internals/mathematical_foundations.md` (in the repo, not a site page): the regret and stability arguments
- [Adaptive execution](../getting-started/concepts/adaptive.md): the same idea, without the code
- [Optimizing a slow query](../tutorials/optimizing-a-slow-query.md): using this in anger
- [Reading a plan](../user-guide/explain-plans.md): the `analyze=True` output above
- [TPC-H benchmarks](../benchmarks/tpch.md): the join shapes where re-planning pays
- [Cardinality estimation](cardinality-estimation.md): where the estimate under test comes from
- [Learned metadata](learned-metadata.md): the across-query half of the loop
- [Cost model](cost-model.md): what a corrected cardinality feeds into
:::
