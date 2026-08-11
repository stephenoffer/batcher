# Adaptive re-optimization

This page describes how Batcher re-plans a query while it runs, using row counts it measured rather than row counts it guessed.

Every cost-based optimizer plans on estimates, and estimates are wrong. A cardinality error compounds multiplicatively up a join tree, so a 10x miss at the leaves becomes a 1000x miss at the root. The plan you get isn't merely suboptimal. It's a plan for a different query.

Batcher's answer is to execute the plan one *pipeline breaker* at a time. A breaker has to materialize its input anyway, so at that point the engine isn't estimating the size of what it just processed. It counted it. Splicing the measured result back in as a source with an exact row count lets the optimizer re-plan the rest of the query against a fact.

## How it compares to other engines

Re-planning mid-query isn't unique to Batcher. What is different is where the loop can run and what it keeps.

![A capability matrix comparing DuckDB, Spark AQE, and Batcher on three properties: re-planning inside one query, running on a single node, and carrying what was learned into the next run. DuckDB optimizes once and keeps no cross-run state. Spark AQE re-plans at stage boundaries but needs shuffle stages and keeps no cross-run state. Batcher re-plans at the same stage-boundary granularity, runs the same loop on a single node, and carries sketches, calibrated costs, and a bandit into the next run.](/_static/diagrams/adaptive_positioning.svg)

| System | When it re-plans |
|---|---|
| DuckDB | Never. It optimizes once and runs that plan. |
| Spark AQE | At stage boundaries, meaning only where a shuffle already forced a materialization. |
| Batcher | At pipeline breakers, the same granularity as Spark AQE, and on a single node as well as on a cluster. |

Batcher's loop is stage-boundary adaptation. It isn't finer-grained than Spark AQE, and the module that implements it says so in its own first line. What Batcher adds over Spark is that the loop runs single-node too, and that it sits alongside a sketch-backed cross-query learned-stats loop that neither DuckDB nor Spark has. See {doc}`Learned metadata </architecture/deep-dives/adaptive/learned-metadata>` for that second half.

## Pipeline breakers

A breaker is an operator that can't emit its first output row until it has consumed its whole input. `plan_surgery.py` names the seven of them in the module constant `BREAKERS`: `Aggregate`, `Sort`, `Distinct`, `Window`, `Limit`, `Join`, and `Union`. Everything else streams. A `Filter -> Project -> MapBatches` chain never segments, because there's nothing to materialize and nothing to count.

Breakers are where the pipeline already stops, which is what makes them free places to measure.

![A streaming Scan-Filter-Project pipeline feeding two pipeline breakers: the HashJoin build, then the Aggregate.](/_static/diagrams/pipeline_breakers.svg)

## The loop

The loop lives in `python/batcher/api/adaptive/`, in the `api` layer, and it's entirely Python. There's no Rust component. Rust returns per-operator metrics and the control plane does the segmenting. The package splits three ways: `staging.py` runs the loop, `gating.py` decides whether to be adaptive and whether an estimate held, and `plan_surgery.py` walks and rewrites the plan tree.

Kyber optimizes the logical plan once, up front. Then each round does the following:

1. `plan_surgery.lowest_breaker(plan)` finds a breaker whose inputs are all breaker-free, which is the first thing that must materialize.
1. `gating._estimate_rows` asks Kyber's `CardinalityEstimator` how big it thinks that stage's output will be. This is the prediction under test.
1. `staging._run_stage` runs the stage through the full Kyber, Carbonite, Core sequence and gets back a table or a partitioned source.
1. The stage's result carries its exact output row count. That's the measurement.
1. `plan_surgery.replace` swaps the stage node for a `Scan` over the materialized result, and the loop repeats.

```text
   the optimized logical plan
            |
   +------->|
   |        v
   |   lowest_breaker(plan)         a breaker whose inputs are all breaker-free
   |        |
   |        v
   |   _estimate_rows(stage)        Kyber's PREDICTION, the thing under test
   |        |
   |        v
   |   _run_stage(stage)            Kyber -> Carbonite -> Core, one full round
   |        |
   |        v
   |   result.num_rows              the MEASUREMENT. counted, not guessed.
   |        |
   |        +--- inside the band? -----> break: run the whole residual plan in one shot
   |        |      q-error <= 1 + optimizer.reoptimize_error   (default 2.0, a 3x band)
   |        |
   |        v  outside the band
   |   replace the stage node with Scan(the materialized result)
   |        |
   +--------+
```

:::{important}
Step 5 is what makes this work. A collected table is spliced back as an {py:class}`InMemorySource <batcher.io.InMemorySource>` with a known row count, so its cardinality carries `Provenance.EXACT`. The next iteration isn't merely re-optimized. It's optimized on a plan where one subtree's size is *known*. Join order, build-side choice, and broadcast eligibility above that point are decided against a fact rather than a guess.
:::

On the distributed path a stage can stay partitioned on disk or on the Flight fleet instead of collecting to the driver, and its `row_count` feeds the next round the same way. A large multi-stage query never funnels every breaker's output through driver memory.

## The trigger, and why its polarity runs backwards

`gating._estimate_accurate` compares estimate against actual as a *symmetric q-error*:

```text
max(actual/estimate, estimate/actual) <= 1.0 + optimizer.reoptimize_error
```

`reoptimize_error` defaults to `2.0`, so the tolerance band is a 3x q-error in either direction. Relative error was tried first and abandoned. `|actual - est| / est` is bounded by 1 for any over-estimate, so it called every over-estimate accurate, and an over-estimate is exactly what the loop exists to catch.

:::{note}
Staging isn't *triggered* by a bad estimate. Staging is the default, and a *good* estimate stops it. If a stage's measured rows land inside the band, the loop breaks and runs the whole residual plan in one shot. Inaccuracy is what keeps the loop paying for another round. That runs backwards from what most readers of the code expect on a first pass.
:::

The economics justify the inversion. Each stage costs roughly 20 to 40 ms of control plane. If the estimator is already tracking reality, buying another measurement is pure overhead. If it isn't, every additional measurement is worth more than it costs.

On a query whose estimates are accurate, the loop adapts at exactly one breaker and then stops paying for measurements it does not need. Re-planning at every breaker happens while estimates keep missing, which is the case the mechanism exists for.

The early exit has one guard. A residual plan that still has no one-shot distributed path, such as a 4-table bushy join, keeps staging regardless of accuracy, because the dispatcher would otherwise refuse it. The shortcut may skip re-optimization, never the staging a plan structurally requires.

## When it turns on

`adaptive="auto"` is the default on {py:meth}`collect() <batcher.Dataset.collect>`. `gating.resolve_adaptive` asks measured history first and falls back to a structural heuristic:

- The plan requires staging on the distributed path, such as a 3-or-more-table star join that the one-shot dispatcher can't route at all. There staging isn't an optimization, it's the only distributed path, and it always wins.
- Otherwise the `MetadataHub` decides, if it has measured this plan signature on both routes. `learned_adaptive_route` is a two-arm bandit over `staged` and `one_shot`, keyed by plan signature and rewarded with the whole query's wall time. Staging only re-plans equivalent algebra, so both arms return the identical relation.
- With no history, the structural heuristic decides: the plan has a join, its total scan rows clear `_ADAPTIVE_MIN_ROWS_PER_STAGE` (5,000,000, a hard-coded module constant in `gating.py` rather than a config knob) multiplied by the number of pipeline breakers the loop would cut at, and some join operand is both non-streamable and sized by a merely-default-provenance estimate.

That floor is charged per cut rather than per query, because that is what staging costs. One breaker-produced operand is one materialization, one re-plan, and the fusion given up at one boundary; a snowflake pays that six times over. A single flat number has to be set for the worst shape it will meet, and the flat 20,000,000 this replaced was: it kept the loop away from the many-join shapes that measurably lost, at the price of never reaching the cheap two-breaker shapes at all. Per-stage, the same arithmetic lands at 20M for a four-cut plan, 10M for a two-cut one, and 30M for a six-cut one.

The bandit matters more than it sounds, because staging is not the planning round-trip the structural gate was priced against. The loop runs one breaker per stage, so it materializes every join separately and gives up both operator fusion and the streaming executor's width. On TPC-H sf10 with warm statistics that costs a multiple of the whole query: q8 887 ms staged against 142 ms one-shot, q17 476 against 105, q2 205 against 32, and q5 running at 1.9x parallelism on a 96-core machine where the one-shot plan reaches 22.6x. The structural heuristic fires on nearly every multi-join query at that scale, and nothing used to measure whether it paid.

It is a bandit rather than a rule because which route wins is not a constant of the plan. Staging is the only distributed route for some shapes; it is what earns the statistics a cold shape has not learned yet; and the cost of a mis-estimated plan grows with the data. Exploration is bounded at roughly one run of the losing arm per signature, and the arms are re-explored as their measurements age (the discounted-UCB horizon in `bandit.py`).

So on a small single-node query, adaptive is off and stays off. That floor exists because a sub-second query can't afford a staging round-trip to learn something it could have guessed. The gate reads exact source row counts, which separates scales cleanly: TPC-H sf1 is roughly 9M rows and stays off, sf10 is roughly 90M and turns on. Below its own floor you can still force it with `adaptive=True`.

## Two feedback loops, one measurement layer

The adaptive loop's measurement is a row count. It isn't the only thing Core measures, and the second loop is the more consequential one over time.

::::{tab-set}
:::{tab-item} Within-query (this page)
```text
measurement:  the exact output rows of the stage just materialized
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
lifetime:     the process, or longer with a durable backend
```
:::
::::

They share the measurement plumbing and nothing else. The across-query half is what makes plans improve the more a query shape runs. Each run also folds its own outcome back in: `gating.record_adaptive_route` records the query's wall time against the route it took, which is what `learned_adaptive_route` ranks on the next run.

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
aggregate                       est~7 actual=7 (1.0x)  0.2ms (1%)  cpu=100%  out=112B  interp
  filter                        est~3,500 actual=3,899 (1.1x)  1.1ms (4%)  cpu=100%  out=0B  jit
    scan                        est~4,000 actual=4,000 (1.0x)  0.0ms (0%)  cpu=100%  out=62KB  interp
```
:::

The `(1.1x)` on the filter is the q-error for that operator. The filter's estimate came from the Selinger default for a range predicate, a third of rows, which happened to land close. A `LIKE '%x%'` or a correlated conjunction is where you see the number blow out, and that's exactly what gets recorded and corrected next run.

## Requirements and limitations

Re-optimization happens *between* stages, never mid-operator. A stage runs to completion. There's no runtime plan switch inside a hash join, no operator-level abort, and no bail-out halfway through a sort. If a join's build side turns out to be 100x the estimate, you find out when the build finishes, not while it's running.

Only the seven breaker types segment. A badly mis-estimated selective filter is measured only when it feeds a breaker, which for a pure `scan -> filter -> collect` is never.

Granularity has a floor, because staging materializes. A plan with many small breakers pays the control-plane round-trip at each one, which is why the accuracy early-exit exists and why the structural heuristic won't turn adaptivity on below 20M input rows. The distributed-staging and learned-history paths ignore that floor, since one is a correctness requirement and the other has measured evidence for the shape.

A stage carrying `map_batches` is opaque to the IR, so the whole-plan Kyber optimize is skipped for a UDF plan and each stage is optimized on its own instead.

## Code map

Each concern below maps to the one file that owns it, so you can read the mechanism
this page describes in the source:

| Concern | File |
|---|---|
| The stage loop, splicing, intermediate cleanup | `python/batcher/api/adaptive/staging.py` |
| The on/off gate and the q-error test | `python/batcher/api/adaptive/gating.py` |
| Breaker set, plan walk, subtree replacement | `python/batcher/api/adaptive/plan_surgery.py` |
| Breaker-free test | `python/batcher/plan/logical/transforms.py::is_streamable` |
| Learned adaptive router | `python/batcher/kyber/learned_tuning/bandit.py::learned_adaptive_route` |
| Per-operator metrics (Rust) | `crates/bc-interp/src/metrics.rs` |
| Metric to feedback transcription | `python/batcher/core/executor.py::_record_op_feedback` |
| The `reoptimize_error` knob | `python/batcher/config/config.py::OptimizerConfig` |

## See also

- {doc}`Architecture </architecture/index>`: the contract loop this closes, where Core measures and Kyber decides.
- {doc}`Kyber optimizer </architecture/internals/kyber>`: the pass pipeline that runs at each stage.
- `docs/architecture/internals/mathematical_foundations.md` (in the repo, not a site page): the regret and stability arguments.
- {doc}`Adaptive execution </getting-started/concepts/adaptive>`: the same idea, without the code.
- {doc}`Optimizing a slow query </tutorials/foundations/optimizing-a-slow-query>`: using this in anger.
- {doc}`Reading a plan </user-guide/operate/tuning/explain-plans>`: the `analyze=True` output above.
- {doc}`TPC-H benchmarks </benchmarks/results/tpch>`: the join shapes where re-planning pays.
- {doc}`Cardinality estimation </architecture/deep-dives/adaptive/cardinality-estimation>`: where the estimate under test comes from.
- {doc}`Learned metadata </architecture/deep-dives/adaptive/learned-metadata>`: the across-query half of the loop.
- {doc}`Cost model </architecture/deep-dives/adaptive/cost-model>`: what a corrected cardinality feeds into.
