# Adaptive re-optimization

This page describes how Batcher re-plans a query while it runs, using row counts it measured rather than row counts it guessed.

Every cost-based optimizer plans on estimates, and a cardinality error compounds multiplicatively up a join tree: a 10x miss at the leaves becomes a 1000x miss at the root. Batcher's answer is to execute the plan one *pipeline breaker* at a time. A breaker has to materialize its input anyway, so at that point the engine counted what it just produced. Splicing the measured result back in as a source with an exact row count lets the optimizer re-plan the rest of the query against a fact.

From the user's side it's one argument, and the answer never depends on it:

```python
import batcher as bt

orders = bt.from_pydict({"cust": [1, 2, 2, 3], "amt": [10.0, 20.0, 5.0, 7.5]})
custs = bt.from_pydict({"cust": [1, 2, 3], "region": ["eu", "us", "us"]})
q = orders.join(custs, on="cust").group_by("region").agg(total=bt.sum("amt")).sort("region")

print(q.collect(adaptive=True).to_pydict())   # {'region': ['eu', 'us'], 'total': [10.0, 32.5]}
print(q.collect(adaptive=False).to_pydict())  # {'region': ['eu', 'us'], 'total': [10.0, 32.5]}
```

`adaptive="auto"`, the default, decides per query whether staging is worth paying for. `True` forces it and `False` turns it off.

## How it compares to other engines

Re-planning mid-query isn't unique to Batcher. What differs is the process the loop runs in and what it keeps.

![A capability matrix comparing DuckDB, Spark AQE, and Batcher on three properties: re-planning inside one query, running inside the Python process, and carrying what was learned into the next run. DuckDB runs embedded in-process but optimizes once and keeps no cross-run state. Spark AQE, on by default since Spark 3.2, re-plans at stage boundaries, but from a JVM beside Python, and keeps no cross-run state. Batcher re-plans at the same stage-boundary granularity, runs the loop inside the Python process, and carries sketches, calibrated costs, and a bandit into the next run.](/_static/diagrams/adaptive_positioning.svg)

| System | When it re-plans |
|---|---|
| DuckDB | Never. It optimizes once and runs that plan. |
| Spark AQE | At stage boundaries, where a shuffle exchange already forced a materialization. On by default since Spark 3.2, in local mode as well as on a cluster. |
| Batcher | At pipeline breakers, the same granularity as Spark AQE, inside the Python process rather than in a JVM beside it. |

This is stage-boundary adaptation, the same grain as Spark AQE. What Batcher adds is that the loop sits alongside a sketch-backed cross-query learned-stats loop. See {doc}`Learned metadata </architecture/deep-dives/adaptive/learned-metadata>` for that second half.

## Pipeline breakers

A breaker is an operator the loop may cut the plan at. `plan_surgery.py` names them in `BREAKERS`: `Aggregate`, `Sort`, `Distinct`, `Window`, `Limit`, `Join`, and `Union`. Most block until they have consumed their whole input. `Limit` and `Union` don't, and are on the list as places the loop may materialize and count. A `Filter -> Project -> MapBatches` chain never segments, because there's nothing to materialize.

![A streaming Scan-Filter-Project pipeline feeding two pipeline breakers: the HashJoin build, then the Aggregate.](/_static/diagrams/pipeline_breakers.svg)

## The loop

The loop lives in [`python/batcher/api/adaptive/`](https://github.com/stephenoffer/batcher/tree/main/python/batcher/api/adaptive) and is entirely Python: Rust returns per-operator metrics and the control plane does the segmenting. Kyber optimizes the logical plan once, up front. Then each round does the following:

1. `plan_surgery.lowest_breaker(plan)` finds a breaker whose inputs are all breaker-free.
1. `gating._estimate_rows` asks Kyber's `CardinalityEstimator` for that stage's output size. This is the prediction under test.
1. `staging._run_stage` runs the stage through the full Kyber, Carbonite, Core sequence.
1. The stage's result carries its exact output row count. That's the measurement.
1. `plan_surgery.replace` swaps the stage node for a `Scan` over the materialized result, and the loop repeats.

:::{important}
Step 5 is what makes this work. The spliced result carries `Provenance.EXACT`, so join order, build side, and broadcast eligibility above that point are decided against a fact rather than a guess.
:::

On the distributed path a stage can stay partitioned on disk or on the Flight fleet instead of collecting to the driver, and its `row_count` feeds the next round the same way.

![One turn of the loop, at a single pipeline breaker. The query is planned with a join operand sized by a guess, carrying Provenance.DEFAULT so the optimizer knows it is guessing that one; the loop cuts at the lowest breaker whose inputs all stream, executes it, and takes an exact row count measured on the materialized result. The two outcomes go opposite ways. If the estimate held, meaning a symmetric q-error inside a 3x band against optimizer.reoptimize_error of 2.0, the loop stops cutting and finishes the rest in one shot. If it missed by more, the residual plan is re-planned on the size just measured, and build side, broadcast and join order are then chosen on rows rather than on a guess. The splice is a Scan over the stage's result, so the next stage's estimator reads an exact size rather than one more inherited guess. This is the same mechanism and the same granularity as Spark AQE, and nothing re-plans inside a stage. A breaker whose output size is already known exactly is not cut at all: across the 22 TPC-H shapes, 17 of 51 ran inline, fused into the subplan above them.](/_static/diagrams/reopt_at_breaker.svg)

## The trigger: a good estimate stops the loop

`gating._estimate_accurate` compares estimate against actual as a symmetric q-error:

```text
max(actual/estimate, estimate/actual) <= 1.0 + optimizer.reoptimize_error
```

`reoptimize_error` defaults to `2.0`, a 3x band in either direction. Staging is the default once the loop engages, and a *good* estimate stops it: if a stage lands inside the band, the residual plan runs in one shot. Each stage costs a materialization and the fusion given up at that boundary, so buying another measurement only pays while estimates keep missing. A residual plan with no one-shot distributed path, such as a 4-table bushy join, keeps staging regardless.

The band is an ordinary config field:

```python
import batcher as bt

print(bt.active_config().optimizer.reoptimize_error)  # 2.0

strict = bt.Config().replace(optimizer=bt.OptimizerConfig(reoptimize_error=1.0))
with bt.config_context(strict):
    print(bt.active_config().optimizer.reoptimize_error)  # 1.0
```

## When it turns on

`gating.resolve_adaptive` resolves `adaptive="auto"`. An explicit `True` or `False` wins outright, and the rest is asked in a fixed order.

![When the within-query adaptive loop engages under adaptive equals auto. Two things skip the ladder: an explicit adaptive=True or False wins outright, and a distributed plan the one-shot dispatcher cannot route is staged whatever its size, because there staging is the only execution path rather than an optimization. Everything else passes three gates in order. Is there a join, since with no join there is nothing to re-decide. Does it clear the floor, which is charged per breaker and not per query. Is a join operand unsized, meaning breaker-produced and still a guess. Any no lands in the same place: plan once, run once, with no staging, no per-stage cut and no re-plan, which is where the great majority of queries land and is the cheaper path for them. A yes makes staging a candidate: stage, measure and re-plan, one breaker per stage, taken once the route bandit has measured it faster for this plan signature. Cold, the bandit runs one-shot first. The floor itself is 5,000,000 rows OR about 320 MB, times the pipeline breakers the loop would cut at, so two breakers need 10,000,000 rows, four need 20,000,000 and six need 30,000,000. The flat 20,000,000-row whole-query gate is retired.](/_static/diagrams/adaptive_gating.svg)

- **Distributed shapes that require staging.** `dist.requires_staging` names a join whose operand already spans two sources (any star or snowflake over three tables or more) and a breaker beneath a breaker, such as `limit(100).group_by(k).agg(...)`. Those stage at any size, because staging is the only correct way they run.
- **A join and a per-stage floor.** Everything else needs a join, and its scan input must reach 5,000,000 rows or roughly 320 MB *per pipeline breaker the loop would cut at*: 10M rows for two breakers, 20M for four, 30M for six. The byte floor is the row floor times the 64-byte `optimizer.row_bytes`, and the two are OR'd so wide rows such as decoded images qualify on bytes.
- **An unsized operand.** Some join operand must be breaker-produced and genuinely unknown. `kyber.estimate_is_reliable` counts an operand whose signature has a history of in-band estimates as already sized.
- **The route bandit.** `learned_adaptive_route` is a two-arm bandit over `staged` and `one_shot`, keyed by plan signature and rewarded with wall time. Cold, it runs one-shot first and explores staging afterwards. Both arms return the identical relation.

Below the floor you can still force the loop with `adaptive=True`. The breaker count is an upper bound, because a breaker whose output is already exact isn't cut: across the TPC-H shapes, 17 of 51 breakers ran inline.

## Two feedback loops, one measurement layer

The adaptive loop's measurement is a row count. Core also records a richer set per operator, which the across-query loop consumes.

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

Each run also folds its own outcome back in: `gating.record_adaptive_route` records the query's wall time against the route it took, which is what `learned_adaptive_route` ranks on the next run.

## Watching it work

`explain(analyze=True)` runs the query and renders estimate against actual per operator. The `MISS` column is each operator's q-error.

```python
import batcher as bt

rows = 4000
ds = bt.from_pydict({"g": [i % 7 for i in range(rows)], "x": [float(i) for i in range(rows)]})
q = ds.filter(bt.col("x") > 100).group_by("g").agg(s=bt.sum("x"))
print(q.explain(analyze=True))
```

:::{dropdown} The `analyze=True` output, estimate against actual
```text
query plan (measured)                                                     3 operators  ·  7 rows  ·  91ms
─────────────────────────────────────────────────────────────────────────────────────────────────────────
OPERATOR                  ESTIMATE        ACTUAL        MISS   TIME     OP SHARE  NOTES
aggregate  [by g · sum]      est≈7      actual=7       exact  279µs  ████▌░  73%  interp
└─ filter  [x > 100]     est≈1,333  actual=3,899  2.9x under  102µs  █▋░░░░  27%  interp
   └─ scan  [source 0]   est≈4,000  actual=4,000       exact    1µs  ░░░░░░  <1%  interp  pushed[x > 100]
```

The filter's cold estimate is the one-third range default, a 2.9x miss, which sits inside the default 3x band. That miss is recorded against the filter's signature and corrects the next run's estimate. The full output continues with where the time went, the bottleneck operator, and the decisions Kyber and Carbonite made. Timings move from run to run.
:::

## Practical limits

- Re-optimization happens *between* stages, never inside an operator. A stage runs to completion.
- Only the seven breaker types segment, so a pure `scan -> filter -> collect` is never measured mid-query.
- A plan with no join never engages the loop on one node, at any size. Re-planning buys a join order, build side, or broadcast decision, and such a plan has none to make.
- A stage carrying `map_batches` is opaque to the IR, so each stage of a UDF plan is optimized on its own.

:::{dropdown} Code map
| Concern | File |
|---|---|
| The stage loop, splicing, intermediate cleanup | [`python/batcher/api/adaptive/staging.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/api/adaptive/staging.py) |
| The on/off gate and the q-error test | [`python/batcher/api/adaptive/gating.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/api/adaptive/gating.py) |
| Breaker set, plan walk, subtree replacement | [`python/batcher/api/adaptive/plan_surgery.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/api/adaptive/plan_surgery.py) |
| Breaker-free test | `python/batcher/plan/logical/transforms.py::is_streamable` |
| Learned adaptive router | `python/batcher/kyber/learned_tuning/bandit.py::learned_adaptive_route` |
| Per-operator metrics (Rust) | [`crates/bc-interp/src/metrics.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/metrics.rs) |
| Metric to feedback transcription | `python/batcher/core/executor.py::_record_op_feedback` |
| The `reoptimize_error` knob | `python/batcher/config/config.py::OptimizerConfig` |
:::

## See also

- {doc}`Architecture </architecture/index>`: the contract loop this closes, where Core measures and Kyber decides.
- {doc}`Kyber optimizer </architecture/internals/kyber>`: the pass pipeline that runs at each stage.
- {doc}`Adaptive execution </getting-started/concepts/adaptive>`: the same idea, without the code.
- {doc}`Optimizing a slow query </getting-started/tutorials/foundations/optimizing-a-slow-query>`: using this in anger.
- {doc}`Reading a plan </user-guide/operate/tuning/explain-plans>`: the `analyze=True` output above.
- {doc}`TPC-H benchmarks </benchmarks/results/tpch>`: the join shapes where re-planning pays.
- {doc}`Cardinality estimation </architecture/deep-dives/adaptive/cardinality-estimation>`: where the estimate under test comes from.
- {doc}`Learned metadata </architecture/deep-dives/adaptive/learned-metadata>`: the across-query half of the loop.
- {doc}`Cost model </architecture/deep-dives/adaptive/cost-model>`: what a corrected cardinality feeds into.
