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
| Throughput | `batcher_queries_total`, `batcher_queries_failed_total`, `batcher_queries_active`, `batcher_query_duration_ms` | How much work, how fast, how much of it failed |
| Data volume | `batcher_rows_scanned_total`, `batcher_rows_out_total`, `batcher_bytes_scanned_total`, `batcher_bytes_written_total` | How much data moved, in and out |
| Memory pressure | `batcher_spills_total`, `batcher_spill_bytes_total`, `batcher_out_of_core_phases_total`, `batcher_peak_rss_bytes` | Whether the cluster is under-provisioned for memory |
| Contention | `batcher_cores_busy`, `batcher_cpu_ms_total`, `batcher_involuntary_context_switches_total`, `batcher_major_page_faults_total` | Whether the machine, rather than the plan, is the problem |
| Distribution | `batcher_partitions_done_total` | Progress through a distributed run |
| Rejects | `batcher_malformed_rows_total`, `batcher_skipped_total`, `batcher_dq_violations_total` | Rows and files a query silently dropped |
| Accelerators | `batcher_node_throttled_devices`, `batcher_node_faulted_devices`, `batcher_node_nvlink_down_devices` | Whether a slow GPU stage is a plan problem or a hardware one |

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
