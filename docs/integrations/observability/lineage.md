# OpenLineage

This page describes how to emit Batcher's column-level lineage to a governance catalog.

Batcher computes, for every output column, the source columns its values derive from. The analysis reads the plan and executes nothing, so every query has it for free. Turn on OpenLineage emission and it reaches the catalog your platform already runs, as one run event per query.

## What an event carries

Each query produces a `START` event before execution and a `COMPLETE` event after it, or a
`FAIL` event if the query raised. Both halves share a run id derived from the query id, so
a backend sees one run rather than two.

The event names its input datasets and carries the standard `columnLineage` facet, which maps each output column to the input fields it derives from. A Batcher run facet records what the run cost and whether it was distributed, with the worker-operator count and usage summed across workers. Without that facet a lineage backend can't tell a plan that ran on one node from the same plan run on a hundred, and those are different runs to audit.

## Turn it on

To turn emission on, set the flag and name a receiver:

```python
# docs: skip
import batcher as bt

bt.set_config(
    bt.active_config().replace(
        observability=bt.ObservabilityConfig(
            openlineage=True,
            openlineage_url="http://marquez:5000",
            openlineage_namespace="prod",
        )
    )
)
```

Or through the environment, which is usually what a deployment does:

```bash
export BATCHER_OBSERVABILITY_OPENLINEAGE=1
export OPENLINEAGE_URL=http://marquez:5000
export OPENLINEAGE_API_KEY=...
```

`OPENLINEAGE_URL` and `OPENLINEAGE_API_KEY` are the variables every other OpenLineage integration on a platform already sets. Batcher reads them when the config fields are empty, so the two can't drift apart. `openlineage_api_key` also accepts an `env:`, `file:`, or `cmd:` secret reference, like every other credential. The namespace defaults to `batcher` and the POST times out after `openlineage_timeout_s`, 5 seconds by default.

## What consumes the events

Any OpenLineage HTTP receiver works. Batcher POSTs the event document to `{url}/api/v1/lineage`, the endpoint Marquez, DataHub, OpenMetadata, and Atlas accept.

Batcher doesn't depend on `openlineage-python`. That client's classes have moved between modules across major versions, and binding to them would mean version-sniffing a package your other integrations also install. For a backend reachable only over Kafka, point `openlineage_url` at an HTTP proxy.

## Reading lineage without emitting it

`Dataset.lineage()` returns the same analysis locally, which is the fastest way to check
what an event will say before you turn emission on:

```python
import os
import tempfile

import batcher as bt

path = os.path.join(tempfile.mkdtemp(), "people.parquet")
bt.from_pydict({"first": ["a"], "last": ["b"], "age": [30]}).write(path, format="parquet")

ds = bt.read.parquet(path).select(name=bt.concat(bt.col("first"), bt.col("last")))
assert sorted(ds.lineage()["name"]) == sorted([f"{path}.first", f"{path}.last"])
```

## Requirements and limitations

Emission never delays a query. Events go to a bounded queue of 256 that one background thread drains in order. A receiver that is down, slow, or saturated costs a dropped event and a debug log line, never latency. When the queue is full the arriving event is dropped rather than one already queued, so a backend that recovers still receives the queued events in order.

Lineage over-approximates and never under-approximates. An opaque `map_batches` stage is
treated as though every output column derives from every input column, because a false
"this might carry PII" costs a review and a false "this cannot" costs a breach.

Lineage tracks data flow, not control flow. Filtering on a column doesn't put that column in the lineage of the surviving rows.

Turning emission on disables the small-query fast path, which otherwise skips the reporting hooks entirely. A lineage record with a hole in it is worse than no record, so deployments that opt in give up that skip.

## See also

- {doc}`/integrations/observability/metrics`: the Prometheus endpoint and dashboard.
- {doc}`Governance </user-guide/trust/governance>`: row filters, column masks, and the tags lineage carries.
- {doc}`/integrations/observability/index`: OpenTelemetry tracing and the rest of this section.
