# Prometheus and Grafana

This page describes how to scrape Batcher's metrics into Prometheus and how to install the Grafana dashboard that ships with the engine.

## What Batcher exposes

Batcher folds every query into a fixed set of counters, gauges, and one latency histogram. The set is bounded, so the exporter's cost doesn't grow with the number of queries you run.

Collection is opt-in. Attaching a sink to the event bus tells the engine to assemble a per-query profile, and that profile is a real cost on the small-query path. A process that exports no metrics doesn't pay it.

```python
from batcher.observe import prometheus_text, start_metrics

start_metrics()
text = prometheus_text()
assert "batcher_queries_total" in text
```

{py:func}`~batcher.observe.prometheus_text` renders the current values in the Prometheus text exposition format. {py:func}`~batcher.observe.metrics_snapshot` returns the same numbers as nested plain data, for exporting somewhere other than Prometheus. {py:func}`~batcher.observe.stop_metrics` detaches the sink again.

## Serve the endpoint

The built-in dashboard server exposes the same numbers at `/metrics`, and as JSON at `/api/metrics`. It binds to `127.0.0.1` on port 4040 by default, so pass a host a scraper can reach:

```python
# docs: skip
import batcher as bt

bt.start_ui(host="0.0.0.0", port=8265)
```

Then point a scrape config at that port:

```yaml
scrape_configs:
  - job_name: batcher
    static_configs:
      - targets: ["batcher-host:8265"]
```

If your application already serves a `/metrics` endpoint, append the output of `prometheus_text()` to it instead of running a second server.

On a Ray cluster, every worker process that executes a query exposes its own counters. Scrape them the way you already discover Ray workers and leave the dashboard's instance picker on `All`. Every panel aggregates across instances, so one dashboard reads a single process or a whole cluster without editing a query.

## The metric vocabulary

The following table groups the metric families by the question each group answers:

| Group | Families | Answers |
|---|---|---|
| Throughput | `batcher_queries_total`, `batcher_queries_failed_total`, `batcher_queries_active`, `batcher_query_duration_seconds` | How much work, how fast, and how much of it failed |
| Data volume | `batcher_rows_scanned_total`, `batcher_rows_out_total`, `batcher_bytes_scanned_total`, `batcher_bytes_written_total` | How much data moved, in and out |
| Memory pressure | `batcher_spills_total`, `batcher_spill_bytes_total`, `batcher_out_of_core_phases_total`, `batcher_peak_rss_bytes` | Whether the cluster is under-provisioned for memory |
| Contention | `batcher_cores_busy`, `batcher_cpu_seconds_total`, `batcher_involuntary_context_switches_total`, `batcher_major_page_faults_total` | Whether the machine, rather than the plan, is the problem |
| Distribution | `batcher_partitions_done_total` | Progress through a distributed run |
| Rejects | `batcher_malformed_rows_total`, `batcher_skipped_total`, `batcher_dq_violations_total` | Rows and files a query dropped without raising |
| Accelerators | `batcher_node_throttled_devices`, `batcher_node_faulted_devices`, `batcher_node_nvlink_down_devices` | Whether a slow GPU stage is a plan problem or a hardware one |
| Resource levels | `batcher_memory_*`, `batcher_spill_*`, `batcher_admission_*`, `batcher_shuffle_*`, `batcher_result_cache_*` | What the engine is holding right now |

Every duration is exported in seconds and named for it, as the Prometheus naming conventions ask: `batcher_query_duration_seconds`, `batcher_cpu_seconds_total`, `batcher_operator_elapsed_seconds_total`. Bytes are bytes. No query needs a scale factor, and `histogram_quantile` returns a figure a panel can label directly.

`metrics_snapshot()` reports the same durations in milliseconds, under keys such as `cpu.time_ms_total`. The snapshot is a plain Python dict with its own documented shape, and the base-unit convention belongs to the Prometheus export rather than to Batcher's naming in general.

## Resource levels

The counter families say what the process did. The resource-level families are the other half of every capacity question: what the engine holds right now. They cover the buffer pool's envelope and high-water mark, the spill store's per-tier bytes and free disk, the admission limiter's queue depth, the shuffle session's credit window, and the result cache's hit rate.

```text
# HELP batcher_memory_engine_pool_used_bytes Current engine pool used bytes of the buffer-pool (level, not a total)
# TYPE batcher_memory_engine_pool_used_bytes gauge
batcher_memory_engine_pool_used_bytes 0
```

These are gauges, so each reading replaces the last. Differencing successive scrapes of `used_bytes` gives noise, not a rate. They matter because counters can't see them: a job that ran 400 queries with the pool at 30% and one that ran 400 with it at 99% and 60 GB spilled look identical by every counter.

Batcher flattens each resource's own statistics into these series generically. A resource that grows a field starts being exported with no change to the exporter, and the `HELP` text is derived from the group and field name, so it can't fall behind the series it describes. One group is capped at 128 series, which stops a per-partition or per-channel reading from becoming unbounded label cardinality. `batcher_resource_series_dropped` reports how many series the cap removed. It's exported as `0` when the cap removed none, so you have a baseline to alert against.

The section stays empty until a query has completed under a resource manager.

## Install the shipped dashboard

`tools/grafana/batcher-overview.json` is a Grafana dashboard that covers every family above. To install it, open **Dashboards > New > Import** in Grafana, upload the JSON, and pick your Prometheus data source.

The dashboard has a job picker and an instance picker, both defaulting to `All`. Three panels are worth knowing before you need them.

Cores kept busy separates a slow query from an unparallelized one, which look identical in a latency panel. A value near 1 on a many-core machine means the query ran on one core.

Bytes spilled sits beside the spill counter because a 1 GB spill and a 100 GB spill increment the counter identically.

Mean file size committed is the small-files panel. A falling line means a lakehouse table is accumulating fragments that cost every later reader, long before any query gets slow enough for someone to investigate.

`tests/integration/test_grafana_dashboard.py` holds the dashboard to the exporter in both directions. A panel can't name a metric that doesn't exist, and a metric the engine pays to produce can't go unshown.

## Requirements and limitations

Counters are cumulative from the moment collection starts, so a backend computes rates by differencing. {py:func}`~batcher.observe.reset_metrics` exists for tests, and for a long-lived service that would rather report per-interval numbers itself.

The accelerator gauges read the local node's devices. On a cluster they mean something only when summed across instances, which is what the dashboard does.

## See also

- {doc}`/integrations/observability/lineage`: run events for a governance catalog.
- {doc}`/integrations/observability/index`: OpenTelemetry tracing and the rest of this section.
- {doc}`Configuration </configuration/index>`: the full `observability` block.
