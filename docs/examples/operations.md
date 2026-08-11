# Operating the engine

This page covers the scripts that read plans, measure queries, configure the engine, and
handle its failures.

## Read the plan before you guess

The plan is the ground truth about what will run, and reading it costs nothing because
`explain` never executes. Two things to look for: whether the filter sits directly on the
scan, and whether the projection narrowed before the join.

```python
import batcher as bt
from batcher import col

orders = bt.from_pydict({"id": [1, 2, 3], "total": [10.0, 90.0, 400.0]})

query = orders.filter(col("total") > 50).select("id")
plan = query.explain()

assert "filter" in plan.lower()
assert "scan" in plan.lower()
assert query.count() == 2
```

`profile` is the executed counterpart and reports per-operator timings, which is the only way
to know whether the join or the scan is the problem. Guessing from the plan shape is how
people end up optimizing the cheap half.

## Measure honestly

Four rules, and the last one matters most. Warm the cache, run more than once, verify the
result before timing it, and report the distribution rather than the best number. A minimum
is not a measurement, it is the luckiest sample.

The scripts here also pin the invariants that a performance change must not break. Morsel
size, partition count and spill are scheduling choices, so sweeping each and asserting the
result is unchanged is the cheapest test that an operator is not accidentally order- or
batch-dependent.

## Errors are typed

Every failure mode has its own class, so a caller can distinguish "you wrote the query wrong"
from "the data is not what you promised" from "the file is not there". Catching bare
`Exception` throws that away.

```python
import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError

orders = bt.from_pydict({"id": [1, 2, 3]})

try:
    orders.select("nope")
except ColumnNotFoundError as error:
    # The message names the columns that do exist, so it is also the fix.
    assert "id" in str(error)

assert issubclass(ColumnNotFoundError, PlanError)
```

## Configuration has a scope

A global `set_option` outlives the function that made it. `option_context` is the same
setting with a lifetime, and it is what library code should use, because the caller's
configuration is not yours to keep.

## Every script on this page

The table below lists the operations and performance scripts in path order.

<!-- library-table: operations,perf -->
| Script | Shows |
| --- | --- |
| `examples/operations/comparing_two_runs.py` | Diffing the output of two pipeline versions |
| `examples/operations/configuration.py` | Configuring the engine: options, scoped overrides, and profiles |
| `examples/operations/configuration_scopes.py` | Setting engine options, and keeping the change scoped |
| `examples/operations/dataset_identity.py` | Comparing two Datasets, and what "equal" means for a lazy plan |
| `examples/operations/debugging_a_wrong_result.py` | Shrinking a wrong result down to the operator that caused it |
| `examples/operations/environment.py` | What is installed, what the engine sees, and what to paste into a bug report |
| `examples/operations/environment_and_hardware.py` | What the engine can see about the machine it is on |
| `examples/operations/error_handling.py` | The exception hierarchy: catching the failure you meant to catch |
| `examples/operations/error_types.py` | The typed exceptions, and catching the right one |
| `examples/operations/explain_modes.py` | Reading a plan at different levels of detail |
| `examples/operations/gpu_cloud.py` | Checking a GPU-cloud node before you trust its throughput |
| `examples/operations/inspecting_a_query.py` | Reading a plan, timing a query, and checking what the engine actually ran |
| `examples/operations/lineage.py` | Tracing where a column came from |
| `examples/operations/memory_and_caching.py` | Bounded memory: caching a reused branch and spilling under a tight budget |
| `examples/operations/memory_pricing.py` | Memory: what a result costs to hold, and when to hold it |
| `examples/operations/observability.py` | Watching a query run: verbosity, logging, and execution statistics |
| `examples/operations/observability_events.py` | Watching a query run: the progress reporter and the activity store |
| `examples/operations/plan_stability.py` | Checking that a refactor did not change the plan |
| `examples/operations/profiling_a_query.py` | Measuring where a query spends its time |
| `examples/operations/query_metadata_feedback.py` | The feedback loop: what the executor measured, and what the optimizer does with it |
| `examples/operations/reading_a_plan.py` | Reading `explain` output, and what the optimizer did to your query |
| `examples/operations/release_check.py` | The suite in miniature: one script touching every subsystem |
| `examples/operations/reproducible_runs.py` | Making a run reproducible, and finding out which parts are not |
| `examples/operations/session_and_versions.py` | What is running: engine version, build profile, and an isolated session |
| `examples/operations/streaming_basics.py` | Batch as the bounded case of streaming: the same operators, incrementally |
| `examples/operations/typed_error_recovery.py` | Recovering from a failure without swallowing it |
| `examples/perf/adaptive_reoptimization.py` | Adaptive re-optimization: re-planning on measured cardinalities |
| `examples/perf/aggregation_strategies.py` | Three ways to compute the same summary, and what each costs |
| `examples/perf/batch_size_and_morsels.py` | Morsel size: the scheduling knob that must never change an answer |
| `examples/perf/cache_versus_recompute.py` | When caching pays, and when it is just memory you gave away |
| `examples/perf/caching_a_reused_result.py` | Caching an intermediate that several branches read |
| `examples/perf/join_side_and_order.py` | Which side of a join is built, and why it matters |
| `examples/perf/measuring_honestly.py` | How to time a query without fooling yourself |
| `examples/perf/predicate_selectivity.py` | Selectivity: how much a predicate actually removes, and why it matters |
| `examples/perf/pushdown_and_projection.py` | Two optimizations you can see: reading fewer columns and fewer rows |
| `examples/perf/repartitioning.py` | Repartitioning: changing the parallelism without changing the data |
| `examples/perf/scan_vs_compute_bound.py` | Telling a scan-bound query from a compute-bound one |
| `examples/perf/spilling_under_a_budget.py` | Running a query that does not fit in memory |
| `examples/perf/streaming_versus_collect.py` | When to stream and when to collect |
| `examples/perf/wide_versus_narrow_tables.py` | What column count costs, and why a projection is the first optimization |
<!-- /library-table -->
