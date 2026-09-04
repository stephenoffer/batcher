# Prometheus and Grafana

This page describes how to scrape Batcher's metrics and how to install the dashboard that
ships with it.

## What Batcher exposes

Batcher folds every query into a fixed set of counters, gauges, and one latency histogram.
The set is bounded, so the cost does not grow with the number of queries you run.

Collection is opt-in. Attaching any sink to the event bus tells the engine to assemble a
per-query profile, which is a real cost on the small-query path, so a process that exports
no metrics does not pay for one.

```python
from batcher.observe import prometheus_text, start_metrics

start_metrics()
text = prometheus_text()
assert "batcher_queries_total" in text
```

`prometheus_text` renders the current values in the Prometheus text exposition format.
`metrics_snapshot` returns the same numbers as nested plain data if you are exporting
somewhere else.

## Serving the endpoint

The built-in dashboard server exposes the same numbers at `/metrics`:

```python
# docs: skip
import batcher as bt

bt.start_ui(port=8265)
```

Point a scrape config at that port:

```yaml
scrape_configs:
  - job_name: batcher
    static_configs:
      - targets: ["batcher-host:8265"]
```

On a Ray cluster, every worker process that executes a query exposes its own counters.
Scrape them the way you already discover Ray workers, and leave the dashboard's instance
picker on `All`. Every panel aggregates across instances, so one dashboard reads a single
process and a whole cluster without editing a query.

## The metric vocabulary

The families group by the question they answer.

| Group | Families | Answers |
|---|---|---|
| Throughput | `batcher_queries_total`, `batcher_queries_failed_total`, `batcher_queries_active`, `batcher_query_duration_seconds` | How much work, how fast, how much of it failed |
| Data volume | `batcher_rows_scanned_total`, `batcher_rows_out_total`, `batcher_bytes_scanned_total`, `batcher_bytes_written_total` | How much data moved, in and out |
| Memory pressure | `batcher_spills_total`, `batcher_spill_bytes_total`, `batcher_out_of_core_phases_total`, `batcher_peak_rss_bytes` | Whether the cluster is under-provisioned for memory |
| Contention | `batcher_cores_busy`, `batcher_cpu_seconds_total`, `batcher_involuntary_context_switches_total`, `batcher_major_page_faults_total` | Whether the machine, rather than the plan, is the problem |
| Distribution | `batcher_partitions_done_total` | Progress through a distributed run |
| Rejects | `batcher_malformed_rows_total`, `batcher_skipped_total`, `batcher_dq_violations_total` | Rows and files a query silently dropped |
| Accelerators | `batcher_node_throttled_devices`, `batcher_node_faulted_devices`, `batcher_node_nvlink_down_devices` | Whether a slow GPU stage is a plan problem or a hardware one |
| Resource levels | `batcher_memory_*`, `batcher_spill_*`, `batcher_admission_*`, `batcher_shuffle_*`, `batcher_result_cache_*` | What the engine is *holding* right now, rather than what it has done |

### Units

Every duration is exported in **seconds** and named for it, which is what the Prometheus
naming conventions ask for: `batcher_query_duration_seconds`, `batcher_cpu_seconds_total`,
`batcher_operator_elapsed_seconds_total`. Bytes are bytes. Nothing needs a scale factor in
the query that reads it, and `histogram_quantile` returns a figure a panel can label
directly.

`metrics_snapshot()` reports the same durations in **milliseconds**, under keys like
`cpu.time_ms_total`. That is deliberate: it is a plain Python dict with its own documented
shape, and the base-unit convention being followed here is Prometheus's rather than a
general rule about how Batcher names things.

## Resource levels

The families above are counters: they say what the process *did*. The `batcher_memory_*`,
`batcher_spill_*`, `batcher_admission_*`, `batcher_shuffle_*` and `batcher_result_cache_*`
families are the other half of every capacity question. They say what it is *holding* right
now: the buffer pool's envelope and high-water mark, the spill store's per-tier bytes and
free disk, the admission limiter's queue depth, the shuffle session's credit window, and the
result cache's hit rate.

```text
# HELP batcher_memory_engine_pool_used_bytes Current engine pool used bytes of the buffer-pool (level, not a total)
# TYPE batcher_memory_engine_pool_used_bytes gauge
batcher_memory_engine_pool_used_bytes 0
```

These are **gauges**, so each reading replaces the last. Differencing successive scrapes of
`used_bytes` gives noise rather than a rate, which is the opposite of how you read the
counter families. A job that ran 400 queries with the pool at 30% and one that ran 400 with
it at 99% and 60 GB spilled are identical by every counter and completely different
operationally.

They are flattened generically from each resource's own statistics, so a resource that grows
a field starts being exported without a change to the exporter. Two consequences worth
knowing:

- The `HELP` text is derived from the group and the field name rather than hand-written, so
  it describes the series without being able to fall behind it.
- A single group is capped at 128 series so that a per-partition or per-channel reading
  cannot become unbounded label cardinality. `batcher_resource_series_dropped` reports how
  many the cap removed, and is exported as `0` when it removed none, so there is a baseline
  to alert against.

The section is empty until a query has completed under a resource manager.

## The shipped dashboard

`tools/grafana/batcher-overview.json` is a Grafana dashboard covering every family above.
Import it through **Dashboards > New > Import**, upload the JSON, and pick your Prometheus
data source.

The dashboard has a job picker and an instance picker, both defaulting to `All`. Three
panels are worth knowing about before you need them:

**Cores kept busy** separates a slow query from an unparallelized one. Both look identical
in a latency panel. A value near 1 on a many-core machine means the query ran on one core.

**Bytes spilled** sits beside the spill counter because they answer different questions. A
1 GB spill and a 100 GB spill increment the counter identically.

**Mean file size committed** is the small-files panel. A falling line means a lakehouse
table is accumulating fragments that cost every later reader, long before a query is slow
enough for anyone to investigate.

`tests/integration/test_grafana_dashboard.py` holds the dashboard to the exporter in both
directions: a panel may not name a metric that does not exist, and a metric the engine
pays to produce may not go unshown.

## Requirements and limitations

Counters are cumulative from the point collection starts, so a backend computes rates by
differencing, as with any counter-based system. `reset_metrics` exists for tests and for a
long-lived service that would rather report per-interval numbers itself.

The accelerator gauges read the local node's devices. On a cluster they are meaningful
only when summed across instances, which is what the dashboard does.

## See also

- {doc}`/integrations/observability/lineage`: run events for a governance catalog.
- {doc}`Configuration </configuration/index>`: the full `observability` block.
