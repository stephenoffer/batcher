# Observability

This section covers how Batcher reports what it did to the tools your platform already
runs: a Prometheus endpoint with a shipped Grafana dashboard, OpenTelemetry traces, and
OpenLineage run events carrying column-level lineage.

Batcher produces signals and owns no exporter. That division is deliberate. An enterprise
already runs a collector, a metrics backend, a tracing backend, and a lineage catalog, and
a data engine that shipped its own would be a second stack to operate rather than a
reduction in work. Every page here connects a signal the engine already measures to the
system that already consumes that kind of signal.

## In this section

| Page | Covers |
|---|---|
| {doc}`/integrations/observability/metrics` | The Prometheus endpoint, the metric vocabulary, and the shipped Grafana dashboard |
| {doc}`/integrations/observability/lineage` | OpenLineage run events, column-level lineage, and the catalogs that consume them |

## Tracing

OpenTelemetry spans are covered with the rest of the configuration surface. Set
`observability.otel_traces` and configure a provider in your host application; Batcher
emits one span per query with a child span per operator, including the worker operators of
a distributed run. Batcher depends only on the OpenTelemetry API, so with no SDK installed
the emit costs nothing.

## See also

- {doc}`Configuration </configuration/index>`: every `observability` option in one place.
- {doc}`Operating a pipeline </user-guide/operate/index>`: what to do once a signal here tells you
  something is wrong.

```{toctree}
:hidden:

metrics
lineage
```
