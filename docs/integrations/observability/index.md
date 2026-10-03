# Observability

Batcher reports what it did to the monitoring stack your platform already runs: a Prometheus endpoint with a Grafana dashboard in the repository, OpenTelemetry traces, and OpenLineage run events with column-level lineage.

The engine produces signals and owns no exporter, so it plugs into the collector, metrics backend, tracing backend, and lineage catalog you already run. Each signal is one option away:

```python
from batcher.config import get_option, option_context

with option_context("observability.otel_traces", True):
    print(get_option("observability.otel_traces"))
# True
```

The following table maps each page to what it covers:

| Page | Covers |
|---|---|
| {doc}`/integrations/observability/metrics` | The Prometheus endpoint, the metric vocabulary, and the Grafana dashboard |
| {doc}`/integrations/observability/lineage` | OpenLineage run events, column-level lineage, and the catalogs that consume them |

## Tracing

Set `observability.otel_traces` to `True` and configure a tracer provider in your host application. Batcher emits one OpenTelemetry span per query, with a child span per operator, including the operators that ran on workers in a distributed run.

```python
# docs: skip
import batcher as bt
from batcher.config import set_option

set_option("observability.otel_traces", True)
bt.read.parquet("events/").group_by("user").agg(n=bt.count()).collect()
```

Install the `otel` extra, which carries only the OpenTelemetry API. Tracing reuses the profile the event log already measures, and the SDK and OTLP exporter stay with your application.

## See also

- {doc}`Configuration </configuration/index>`: every `observability` option in one place.
- {doc}`Operating a pipeline </user-guide/operate/index>`: what to do once a signal here tells you something is wrong.
- {doc}`/integrations/index`: every other integration.

```{toctree}
:hidden:

metrics
lineage
```
