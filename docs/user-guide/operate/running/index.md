# Keeping it running

These pages cover a job in production: seeing what it is doing, working out why it stopped,
and surviving hardware that does not stay up. The GPU pages are the same concerns on a fleet
where a single device can take a job down.

| Page | What it covers |
|---|---|
| {doc}`Observability <observability>` | The one event channel every subsystem publishes to, and the sinks that read it |
| {doc}`Metrics <metrics>` | The counters a scrape loop reads: throughput, per-operator work, and what a run costs the machine |
| {doc}`Troubleshooting <troubleshooting>` | The errors you are most likely to hit, by symptom |
| {doc}`GPU fleets <gpu-fleets>` | Sizing work against what a device actually has, across a datacenter |
| {doc}`Diagnose a slow GPU stage <gpu-diagnosis>` | Finding out why a GPU stage ran slower than the device allows |
| {doc}`Running on unstable nodes <unstable-nodes>` | What keeps a job alive when nodes and devices come and go |

## See also

- {doc}`/user-guide/operate/tuning/index`: the levers to reach for once the job is stable.
- {doc}`/configuration/fault-tolerance`: the settings behind the retry and recovery behavior.

```{toctree}
:hidden:

observability
metrics
troubleshooting
gpu-fleets
gpu-diagnosis
unstable-nodes
```
