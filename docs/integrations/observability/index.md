# Observability

Batcher reports what it did to the monitoring stack your platform already runs: a Prometheus endpoint with a Grafana dashboard in the repository, OpenTelemetry traces, and OpenLineage run events with column-level lineage.

The engine produces signals and owns no exporter. Your organization already operates a collector, a metrics backend, a tracing backend, and a lineage catalog, and a data engine that brought its own would be one more stack to run. Each page here connects a signal Batcher already measures to the system that already consumes that kind of signal.

The following table maps each page to what it covers:

| Page | Covers |
|---|---|
| {doc}`/integrations/observability/metrics` | The Prometheus endpoint, the metric vocabulary, and the Grafana dashboard |
| {doc}`/integrations/observability/lineage` | OpenLineage run events, column-level lineage, and the catalogs that consume them |

## Tracing

Set `observability.otel_traces` to `True` and configure a tracer provider in your host application. Batcher then emits one OpenTelemetry span per query, with a child span per operator. On a distributed run the child spans include the operators that ran on the workers, so the query that matters most doesn't show up as a bare span.

Install the `otel` extra, which carries only the OpenTelemetry API. Batcher uses the same measured profile as the event log, so tracing adds the span emit and no extra measurement. The SDK and the OTLP exporter stay with your application, which already owns them.

## See also

- {doc}`Configuration </configuration/index>`: every `observability` option in one place.
- {doc}`Operating a pipeline </user-guide/operate/index>`: what to do once a signal here tells you something is wrong.
- {doc}`/integrations/index`: every other integration.

```{toctree}
:hidden:

metrics
lineage
```
