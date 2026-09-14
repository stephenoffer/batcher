# Keeping it running

These pages cover a job in production: seeing what it is doing, working out why it stopped,
and surviving hardware that does not stay up. The GPU pages are those same concerns on a
fleet, where one sick device can take a job down. The table is the whole section.

| Page | What it covers |
|---|---|
| {doc}`Observability <observability>` | The one event channel every subsystem publishes to, and the sinks that read it |
| {doc}`The terminal <terminal>` | What a query prints while it runs, and the one line it leaves behind |
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
terminal
metrics
troubleshooting
gpu-fleets
gpu-diagnosis
unstable-nodes
```
