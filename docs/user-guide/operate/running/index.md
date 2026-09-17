# Keeping it running

This section covers a Batcher job in production: seeing what it is doing, finding out why it stopped, and keeping it alive on hardware that doesn't stay up.

Batcher is built to be watched. Every subsystem, from the Kyber optimizer to the distributed scheduler, publishes to one event bus, and every surface you look at reads from it. The terminal line, the web dashboard, the per-query JSON event log, the Prometheus counters and the OpenTelemetry spans all carry the same measurements under the same query id. A log line joins to the plan and the profile of the run that wrote it, so none of these surfaces can disagree with another.

## See what a job is doing

Run a query in a terminal and it tells you where it is. The status line names the phase, such as `optimizing`, `admission` or `on cluster`, so a slow small query says whether the time went to planning or to executing. When the query finishes, the line collapses to one summary that also records what else happened: inputs skipped, bytes spilled, workers lost. A job that survived a failure says so beside its result.

For more than one query at a time, {py:func}`bt.start_ui() <batcher.start_ui>` opens a dashboard that groups every run of the same plan shape into one pipeline. Re-running a query builds a baseline, so "was this run slow?" has an answer. A scrape loop gets the same numbers from `/metrics` in Prometheus format, including what each run cost the machine in CPU, memory and disk.

## Find out why it stopped

Batcher's errors are typed and they say what to do. A misspelled column raises {py:exc}`ColumnNotFoundError <batcher.ColumnNotFoundError>`, and a near miss names the column you meant. Every failure subclasses {py:exc}`BatcherError <batcher.BatcherError>`, and many also subclass the builtin you would already catch, such as `ValueError` or `ImportError`. Ctrl-C stops a long `collect()` between morsels or operators, and a cancelled query never returns a partial result.

## Run on a GPU fleet

At datacenter scale a device rarely fails by disappearing. It stays up and gets slower or wronger. Batcher reads the driver's Xid log and the kernel log for faults no GPU probe reports. It takes a node that keeps failing tasks out of rotation, and refuses to retry past a device that may have corrupted results. It also places a multi-device collective inside one NVLink domain, prefers a MIG partition over a whole device when a model fits one, and clamps fan-out to a power budget you set. When a GPU stage is correct but slow, sampling classifies each device as compute bound, transfer bound, throttled, starved or contended, and names the one thing to change.

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
