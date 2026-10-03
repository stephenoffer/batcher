# Keeping it running

This section covers a Batcher job in production: seeing what it is doing, finding out why it stopped, and keeping it alive on hardware that doesn't stay up.

Batcher is built to be watched. Every subsystem publishes to one event bus, and the terminal line, the web dashboard, the JSON event log, the Prometheus counters, and the OpenTelemetry spans all read from it under the same query id.

## See what a job is doing

In a terminal, a status line names the phase, such as `optimizing`, `admission`, or `on cluster`, and collapses to a one-line summary when the query finishes. The same measurements are queryable as a `Dataset`:

```python
import batcher as bt

orders = bt.from_pydict({"region": ["eu", "us", "eu"], "amount": [10, 20, 30]})
orders.group_by("region").agg(total=bt.col("amount").sum()).collect()

history = bt.query_history(limit=5)
print({"total_elapsed_ms", "rows_produced", "spilled"} <= set(history.columns))
# True
```

For many queries at once, {py:func}`bt.start_ui() <batcher.start_ui>` opens a dashboard that groups runs of the same plan shape and builds a baseline, and `/metrics` serves the same numbers to a Prometheus scrape loop.

## Find out why it stopped

Errors are typed and say what to do. Every failure subclasses {py:exc}`BatcherError <batcher.BatcherError>`, many also subclass the builtin you would already catch, and a near-miss column name suggests the one you meant:

```python
try:
    orders.select("amuont").collect()
except bt.ColumnNotFoundError as err:
    print(isinstance(err, bt.BatcherError), isinstance(err, KeyError))
    print(str(err).split("Did you mean")[1].split("?")[0].strip())
# True True
# 'amount'
```

Ctrl-C stops a long `collect()` between morsels, and a cancelled query never returns a partial result.

## Run on a GPU fleet

At datacenter scale a device rarely fails by disappearing. It gets slower or wronger. Batcher reads the driver's Xid log for faults no probe reports, takes a repeatedly failing node out of rotation, places collectives inside one NVLink domain, prefers a MIG partition when a model fits one, and clamps fan-out to a power budget. When a GPU stage is slow, sampling classifies each device as compute bound, transfer bound, throttled, starved, or contended.

## Pages in this section

The pages below are ordered from the surfaces every job uses to the ones only a GPU fleet needs.

| Page | What it covers |
|---|---|
| {doc}`Observability <observability>` | The event bus, verbosity, structured logs, the web dashboard, the event log, and OpenTelemetry |
| {doc}`The terminal <terminal>` | The live status line, the one-line summary a query leaves behind, and how to turn it off |
| {doc}`Metrics <metrics>` | The counters a scrape loop reads: throughput, per-operator work, writes, data quality, and machine cost |
| {doc}`Troubleshooting <troubleshooting>` | The errors you are most likely to hit, the exception types, and cancelling a query |
| {doc}`GPU fleets <gpu-fleets>` | Device sizing, power budgets, energy reports, fabric-aware placement, MIG, KV-cache sizing, and data residency |
| {doc}`Diagnose a slow GPU stage <gpu-diagnosis>` | Sampling devices across a run and reading one verdict per device |
| {doc}`Running on unstable nodes <unstable-nodes>` | Fault detection, quarantine, the collective timeout, and corrupting faults |

## See also

- {doc}`/user-guide/operate/tuning/index`: the levers to reach for once the job is stable.
- {doc}`/configuration/fault-tolerance`: the settings behind the retry and recovery behavior.
- {doc}`/architecture/fault-tolerance`: how recovery works underneath.
- {doc}`/examples/distributed`: distributed and streaming scripts, each run on every commit.

```{toctree}
:hidden:

observability
terminal
metrics
troubleshooting
gpu-fleets
gpu-diagnosis
unstable-nodes
```
