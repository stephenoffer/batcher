# OpenLineage

This page describes how to emit Batcher's column-level lineage to a governance catalog.

Batcher computes, for every output column, the source columns its values derive from. That
analysis reads the plan and executes nothing, so it is available on any query. Turning on
OpenLineage emission ships it to the catalog your platform already runs, as one run event
per query.

## What an event carries

Each query produces a `START` event before execution and a `COMPLETE` event after it, or a
`FAIL` event if the query raised. Both halves share a run id derived from the query id, so
a backend sees one run rather than two.

The event names its input datasets, and carries the standard `columnLineage` facet mapping
each output column to the input fields it derives from. A Batcher run facet records what
the run cost and, importantly, whether it was distributed: a lineage backend cannot
otherwise tell a plan that ran on one node from the same plan that ran on a hundred, and
those are different runs to audit.

## Turning it on

Set the flag and name a receiver:

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

`OPENLINEAGE_URL` and `OPENLINEAGE_API_KEY` are the variables every other OpenLineage
integration in a platform already sets, and Batcher reads them as the fallback so the two
cannot drift apart. `openlineage_api_key` also accepts an `env:`, `file:`, or `cmd:`
secret reference, the same as every other credential.

## What consumes the events

Any OpenLineage receiver. Batcher POSTs the event document to `{url}/api/v1/lineage`,
which is the transport Marquez, DataHub, OpenMetadata, and Atlas all accept. There is no
`openlineage-python` dependency: the client library's Python classes have moved between
modules across its major versions, and binding to them would mean version-sniffing a
package your own integrations also install. A backend reachable only over Kafka is reached
by pointing `openlineage_url` at a proxy.

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

Emission never delays a query. Events go to a bounded queue drained by one background
thread, so a receiver that is down, slow, or saturated costs a dropped event and a debug
log line rather than latency. A backend that has been unreachable long enough to fill the
queue loses the oldest events first.

Lineage over-approximates and never under-approximates. An opaque `map_batches` stage is
treated as though every output column derives from every input column, because a false
"this might carry PII" costs a review and a false "this cannot" costs a breach.

Lineage tracks data flow, not control flow. Filtering on a column does not put that column
in the lineage of the surviving rows. This matches how Unity Catalog and Snowflake report
lineage.

Turning emission on disables the small-query fast path, which otherwise skips the
reporting hooks entirely. A lineage record with an invisible hole in it is worse than no
record, so the orchestration skip is given up for deployments that opted in.

## See also

- {doc}`/integrations/observability/metrics`: the Prometheus endpoint and dashboard.
- {doc}`Governance </user-guide/trust/governance>`: row filters, column masks, and the
  tags lineage carries.
