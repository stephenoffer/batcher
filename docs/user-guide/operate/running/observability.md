# Observability

Batcher reports what it is doing through one channel. Every subsystem publishes to a single
internal event bus, including the Kyber optimizer, the Carbonite resource manager, the Core
executor, and the distributed scheduler. Everything you can *see* is a consumer of that
bus:

| Surface | What it is for | Default |
| --- | --- | --- |
| {doc}`Terminal progress bar <terminal>` | watching a query run, interactively | on in a real terminal |
| Structured logs | what the engine decided, and why | `WARNING` and above |
| Web dashboard | plans, per-operator timings, throughput, live logs | off ({py:func}`bt.start_ui() <batcher.start_ui>`) |
| JSON event log | the durable per-query artifact, on disk | on |

Because they share one source, they can't disagree. The timeline in the dashboard and the
profile in the on-disk event log are the same measurements, under the same query id.

The event log takes one step more than the live surfaces. The per-query profile is assembled once, when the query finishes. Batcher publishes it onto the bus, which is how the dashboard gets it, and writes the same document to the event log. OpenTelemetry spans, when you turn them on, come from that same profile:

![While a query runs, Kyber, Carbonite, Core, the distributed scheduler and the engine's log records all publish to one event bus, which carries a single query id. Three surfaces subscribe to it: the terminal progress bar for the live phase, the web dashboard started by bt.start_ui() for runs, plans and logs, and the process-wide metrics counters for throughput and durations. When the query ends, one measured query profile holding the plans, decisions and per-operator measurements is published back onto the bus as stages, written as the same document to the JSON event log on disk, which is on by default, and emitted from the same profile as OpenTelemetry spans when otel_traces is on. Because the profile is measured once, the dashboard and the event log can't disagree.](/_static/diagrams/observability_bus.svg)

## Verbosity, the one dial

Nearly everything above is reachable through a single setting. `verbosity` is the
`-v`/`-vv` ladder every CLI has, spelled out:

| Level | Name | Shows | Rust tracing |
| --- | --- | --- | --- |
| 0 | `silent` | nothing but unrecoverable failures | off |
| 1 | `quiet` | errors only, and no progress bar | `error` |
| 2 | `normal` | **default**: warnings + progress bar | `warn` |
| 3 | `verbose` | + optimizer and resource decisions | `info` |
| 4 | `debug` | + per-phase timings and plan detail | `debug` |
| 5 | `trace` | + Rust per-morsel spans; progress bar forced on | `trace` |

```python
import batcher as bt
from batcher.config import ObservabilityConfig, active_config, set_config

set_config(active_config().replace(observability=ObservabilityConfig(verbosity="debug")))
```

Names and integers are interchangeable, so a CLI counting `-v` flags can pass the number
straight through:

```python
ObservabilityConfig(verbosity=4)  # same as "debug"
```

Or from the environment, without touching code:

```bash
BATCHER_OBSERVABILITY_VERBOSITY=trace python job.py
```

`trace` is the only level where the two ladders differ. Python's `logging` has no level
below `DEBUG`. The engine's Rust `tracing` spans do, and that is where per-morsel work is
visible, so `trace` means `DEBUG` in Python and `TRACE` in Rust.

### Overriding one component

`log_level` and `progress` are the two knobs `verbosity` sets. Each defaults to `None`,
meaning "derive me". Set either explicitly to override that one alone:

```python
# Chatty logs, but never draw a progress bar, such as inside a TUI you own.
ObservabilityConfig(verbosity="debug", progress="off")

# A progress bar, but keep the log stream quiet.
ObservabilityConfig(verbosity="normal", log_level="ERROR")
```

Read the effective values back with `resolved_log_level` and `resolved_progress`.
`log_level` itself is `None` whenever you are driving with the dial.

```{admonition} Why `None` and not a default string
:class: note

With a concrete default there is no way to distinguish "the user asked for `WARNING`" from
"nobody said anything", so a preset could never know whether it was allowed to act. `None`
makes the precedence unambiguous, which is why it is the default rather than `"WARNING"`.
```

## The terminal

In an interactive terminal a query renders a live status line naming the phase it is in, and
collapses to one aligned summary when it finishes. {doc}`The terminal <terminal>` covers
what each field means, how the display degrades on a terminal that cannot draw it, and when
it suppresses itself.

## Logs

All engine logs live under the `batcher.*` logger hierarchy, one logger per subsystem, such
as `batcher.kyber`, `batcher.carbonite`, and `batcher.core`. Batcher owns that hierarchy and
nothing else, so your application's own `logging` setup keeps working untouched.

For the common cases there are one-line switches. `set_log_level` takes a level name, a
`logging` constant, or a verbosity preset, and applies immediately rather than at the next
query. `enable_logging` turns the console handler on, optionally writing a rotating file as
well, and `disable_logging` silences the console without disturbing your own configuration.

```python
from batcher.config import disable_logging, enable_logging, set_log_level

set_log_level("debug")
enable_logging("info", log_file="/tmp/batcher.log")
disable_logging()
```

`set_verbosity` moves the whole ladder, log level and progress bar together, and
`set_progress` controls the bar alone. Reach for `get_logger` when you want to attach a
handler or set a level on one subsystem with plain stdlib calls:

```python
from batcher.config import get_logger, set_progress, set_verbosity

set_verbosity("verbose")
set_progress(False)
print(get_logger("kyber").name)
# batcher.kyber
```

For anything these don't cover, one config controls all of them:

```python
set_config(
    active_config().replace(
        observability=ObservabilityConfig(
            log_level="INFO",  # CRITICAL | ERROR | WARNING | INFO | DEBUG
            log_format="json",  # "human" (default) or "json" for a log shipper
            log_file="/var/log/batcher.log",
            console=False,  # file-only
        )
    )
)
```

Engine log records carry structured fields, not only a sentence. The terminal layout is
[logfmt](https://brandur.org/logfmt), the `key=value` convention from Heroku and the Go
ecosystem, behind a fixed-width prefix. One line is therefore both aligned for a human and
parseable by a log processor without a bespoke regex per message. Values are quoted only when
they contain a space, as the convention requires. Field *names* follow the OpenTelemetry
practice of carrying their unit, so the field is `duration_ms` rather than `duration` and a
number's meaning never depends on surrounding prose.

```text
14:19:25  INFO     kyber        join reorder  tables=3 cost=1.25 note="two words" query_id=20240502-141925-000003
```

```json
{"time": "2024-05-02T14:19:25.412Z", "level": "INFO", "logger": "batcher.kyber",
 "message": "join reorder", "query_id": "20240502-141925-000003", "pid": 41207,
 "thread": "MainThread", "fields": {"tables": 3, "cost": 1.25}}
```

Four fields are attached for you rather than by the call site:

`query_id` names the query in flight. It is read from the ambient scope when the record is
formatted rather than passed in, so a plain `logger.warning` deep inside a subsystem is
correlated too. It is the same id the event log document, the plan DAG, and the dashboard
row use, which is what makes a log line joinable to the plan and the
profile that describe the same run.

`time` is RFC 3339 in UTC. Log platforms reject a local-time, comma-separated stamp and
fall back to ingest time, which silently reorders a stream whose whole value is its order.

`pid` and `thread` are what make a distributed stream readable: the same subsystem logs
from the driver and from every worker into one index.

An exception is carried as `exc_type` and `exc_message` alongside the formatted traceback,
so you can alert on a class of failure instead of matching a regex against a sentence.

The `log_level` also drives the Rust data plane's tracing, so raising it to `DEBUG` reveals
the engine's per-operator work, not only the Python control plane's.

`DEBUG` additionally reveals records from the decoder libraries the engine links, such as
the audio decoder behind `.audio.decode()`. Below `DEBUG` those stay hidden, and that is
deliberate rather than an oversight. A decoder reports a payload it cannot parse as an
error, but every media expression treats an unreadable payload as a null row, because an
unstructured corpus is expected to be mixed. Forwarding those reports would put one error
line in your log per row that is behaving exactly as documented, and would bury the
engine's real diagnostics underneath them.

The suppression is also what keeps the null path fast. Each forwarded record acquires the
Python GIL, which serializes the parallel decode. Turn `DEBUG` on when you are diagnosing
why a specific file will not decode, and expect a mixed corpus to run slower while it is on.

## The web dashboard

Start it and keep working. It runs on its own port, in a daemon thread, and never blocks
the process it is observing:

```python
import batcher as bt

bt.start_ui()  # returns 'http://127.0.0.1:4040'
bt.start_ui(port=8080, open_browser=True)
```

The dashboard is a drill-down: every pipeline, then one pipeline, then one run.

- **Pipelines**: every distinct query shape, one lane each. A lane is led by a thumbnail
  of the pipeline's plan, so you recognize a pipeline by its *shape* before you read its
  name. Re-running a query builds a baseline rather than a pile of unrelated entries, so
  "was this run slow?" has an answer.
- **One pipeline**: how it behaves over time, its step history as a matrix of runs
  against steps, and what holds true across all of its runs rather than in one.
- **One run**: five renderings of the same per-step data, plus the plan as a document.

### What a pipeline is

A *pipeline* is every run of one plan shape. Its identity is the *plan signature*, the
same fingerprint Kyber keys learned statistics on, so "the dashboard's pipeline" and "the
thing the optimizer learned about" are the same thing. Two runs over different data share a
pipeline; a structurally different query starts a new one.

Each pipeline has an **id** (that signature, shown as `#` and the first characters, copyable
in full) and a **name**. Until you name it, the name is generated from the plan shape, as in
`Read → Filter → Join → Group`, which is more telling than the raw operator tag. Click the
pencil on a lane, or on the pipeline page heading, to give it a real name such as `nightly
rollup`, and a note beside it.

A name is the one thing about a pipeline that outlives the process. It is written to
`$BATCHER_HOME/pipelines.json`, which defaults to `~/.batcher/pipelines.json`, so a pipeline
you named is still named after a restart. Everything else on the dashboard is a measurement
that ages out of memory. The name does not.

Within a run, **Steps** offers the plan graph, the pipeline stages, a flame view, a
ranked list, and a sortable table. Each is annotated with rows out, elapsed time, spill
volume, and the estimate Kyber planned for. Actual against estimate is the single most
useful number when a query is slow for a reason the plan did not predict.

**Query** shows the plan as a document, in three forms:

- **Explain**: the plan as a text tree, annotated with what each step measured. Toggle
  "Show the plan as written" to see the plan before the optimizer touched it.
- **What the optimizer changed**: the two plans compared. A pushdown is reported as one
  rewrite, with the steps it dragged past it listed separately rather than as four equal
  findings.
- **Plan document**: the exact JSON IR that crossed into the Rust engine, for when the
  rendering is the thing under suspicion.

**Findings** carries what the engine concluded: the per-run insights, the optimizer and
resource-manager decisions, and any adaptive re-optimization, meaning the points where the
engine had counted the rows rather than estimated them and re-planned what was left.

**Live** is the forward-looking page, for work measured in minutes rather than
milliseconds: partition progress with a real denominator where the engine reports one,
per-device GPU utilization and VRAM against their target bands, inference throughput and
blocked time, actor-pool size, and any rows dropped under `on_read_error="skip"`.

**Logs** is the live stream, with a volume histogram you can drag a time window out of,
level and regex filtering, structured-field filters, and per-line permalinks.

```{note}
Nothing on the dashboard is inferred. A run with no measured step timings shows no
timings rather than zeroes, a partition count with no reported total shows no percentage,
and there is no Gantt chart of operator start times because the engine records how long
each operator took, not when it began.
```

The **Learn** page is for anyone arriving from another engine. It maps the panel you
already know, such as a Spark UI tab, an Airflow view, or a DuckDB `EXPLAIN`, to its
equivalent here.

{py:func}`start_ui <batcher.start_ui>` is idempotent. Calling it again returns the URL of the dashboard already
running rather than binding a second port. Ask for that URL at any time with {py:func}`bt.ui_url() <batcher.ui_url>`,
which returns `None` when no dashboard is running. That helps when a helper needs to print
or link the dashboard without caring who started it:

```python
if bt.ui_url() is None:
    bt.start_ui()
print(f"dashboard: {bt.ui_url()}")
```

Passing `port=0` asks the OS for any free port, which is the right choice in tests and in
any environment where 4040 may already be taken. Read back the actual port from the
returned URL or from {py:obj}`bt.ui_url() <batcher.ui_url>`.

Stop it when you are done, or let the process exit and it is cleaned up automatically:

```python
bt.stop_ui()
```

To have a long-running service always expose it, turn it on in config instead of calling
`start_ui` by hand:

```python
set_config(active_config().replace(observability=ObservabilityConfig(ui=True, ui_port=4040)))
```

```{admonition} The dashboard binds to loopback on purpose
:class: warning

It exposes query text, plans, and log lines, which are effectively facts about your data.
The default `ui_host="127.0.0.1"` is reachable only from the machine running the engine.
Setting it to `0.0.0.0` publishes all of that to your network. Do it deliberately, behind
whatever authentication your environment provides. Batcher ships no authentication of its
own.
```

## The JSON event log

Independently of the live surfaces, every query writes one structured JSON document to
`$BATCHER_HOME/logs`, which defaults to `~/.batcher/logs`. The document holds the logical
and optimized plan, the decisions, and the measured per-operator profile. This is the
durable artifact: the dashboard's ring buffer is a debugging window that forgets, and this
does not.

```python
set_config(
    active_config().replace(
        observability=ObservabilityConfig(
            event_log=True,  # on by default
            event_log_dir="/data/batcher-events",
            event_log_max_files=1000,  # oldest pruned on write; 0 = unbounded
        )
    )
)
```

`event_log=False` removes the per-query write. That is worth doing when you run many small
queries and nothing consumes the documents. Otherwise leave it on.

### Query the history as a table

{py:func}`bt.query_history() <batcher.query_history>` reads those documents back as a `Dataset`, one row per completed query, with the measurements rather than the plan: `total_elapsed_ms`, `rows_produced`, `spilled`, `bytes_spilled`, `peak_memory_bytes`, the CPU usage, and a `profile_path` naming the full document. The questions an operator asks of a run history are relational, such as which queries spilled or which ran longer than a second, so the answer is a filter rather than a script over a folder of JSON:

```python
import dataclasses
import tempfile

import batcher as bt
from batcher.config import active_config, config_context

log_dir = tempfile.mkdtemp()
cfg = active_config().replace(
    observability=dataclasses.replace(
        active_config().observability, event_log=True, event_log_dir=log_dir
    )
)
with config_context(cfg):
    ds = bt.from_pydict({"k": ["a", "b", "a"], "v": [1, 2, 3]})
    ds.group_by("k").agg(s=bt.col("v").sum()).collect()

history = bt.query_history(log_dir)
print(history.select("rows_produced", "spilled").to_pydict())
# {'rows_produced': [2], 'spilled': [False]}
slow = history.filter(bt.col("total_elapsed_ms") > 1000.0)
```

With no argument it reads the directory `event_log_dir` names, the same one the engine writes. Only completed queries are recorded. A query that raised writes no document, so failures are found on the event bus, in the trace, or as an OpenLineage `FAIL` event instead.

## Metrics

Everything above is built for one query at a time. A scrape loop wants a handful of numbers
forever instead, and the same bus feeds process-wide counters for that: throughput, the
duration histogram, per-operator work, data-quality contracts, and what the run cost the
machine in CPU, memory, and disk. See {doc}`Metrics <metrics>`.

## OpenTelemetry

If your organization already collects traces, emit into them rather than standing up a
second place to look. Batcher emits one span per query with a child span per operator, into
the tracer your application configured. It owns no exporter:

```python
set_config(active_config().replace(observability=ObservabilityConfig(otel_traces=True)))
```

This needs the `otel` extra, `pip install 'batcher-engine[otel]'`, plus a provider the host
app sets up. It reuses the same measured profile as the event log, so turning it on adds the
span emit and nothing else. The measurement was already happening.

The spans carry real timestamps. Emission happens after the query has finished, so the
query span is placed over the interval the query actually occupied and each operator span
is given its measured duration. A waterfall therefore ranks the operators by length, the
same ranking the `OP SHARE` column shows in
{doc}`explain(analyze=True) </user-guide/operate/tuning/explain-plans>`.

One thing is deliberately *not* reconstructed: where each operator sat inside that interval.
The profile records a duration per operator and no start offset, so every operator span
begins at the query's start. Laying them out end to end would look more like a waterfall and
would be an invention. On the streaming executor, where operators genuinely interleave, it
would be a wrong one.

A query that raises produces a span too, with the exception recorded on it and the span
status set to `ERROR`. Without it the one class of run you most want to find in a trace
backend would be the only class that was never there, and a latency histogram built from
these spans would silently exclude every timeout.

| Attribute | On | Meaning |
| --- | --- | --- |
| `batcher.query_id` | query | the id shared with the event log, the dashboard, and every log record from the run |
| `batcher.rows`, `batcher.total_ms` | query | what the query returned and how long it took |
| `batcher.bottleneck.kind`, `.op_id` | query | the dominant operator, so a backend can group by it |
| `batcher.cores_busy`, `batcher.cpu_ms`, `batcher.peak_rss_bytes` | query | what the run cost the machine; filter on `batcher.cores_busy < 2` to find queries that failed to parallelize |
| `batcher.ok` | query | `false` on a failed query; absent on a successful one |
| `batcher.op.kind`, `.rows_in`, `.rows_out`, `.elapsed_ms` | operator | the same per-operator facts `explain(analyze=True)` prints |
| `batcher.op.spill_bytes` | operator | present only when the operator spilled, because a 1 GiB spill and a 100 GiB one are the same boolean |
| `batcher.op.scope` | operator | `driver` or `worker`, distinguishing the driver tree from the distributed map sub-plan |

## OpenLineage

Column-level lineage leaves the process the same way. Set `openlineage=True` and Batcher posts
one OpenLineage run event per query, a `START` before execution and a `COMPLETE` or `FAIL`
after, carrying the column-level lineage the governance layer already computes:

```python
# docs: skip
set_config(
    active_config().replace(
        observability=ObservabilityConfig(openlineage=True, openlineage_url="http://marquez:5000")
    )
)
```

An empty `openlineage_url` reads the standard `OPENLINEAGE_URL` variable, and an empty
`openlineage_api_key` reads `OPENLINEAGE_API_KEY`. Events are posted from a bounded background
queue, so a slow receiver costs a dropped event rather than query latency.
{doc}`/integrations/observability/lineage` covers the receiver side.

## Requirements and limitations

Every surface on this page reports from the process that ran the query, which on a cluster is the driver. That has three consequences for a deployment with more than one driver:

- **The event log and `query_history()` are per driver.** The driver writes each document to its own `event_log_dir`, and `query_history()` reads that directory on the machine it runs on. Two drivers on two nodes keep two histories. Point `event_log_dir` at a mount every driver shares to read one history across them.
- **Learned statistics follow the `metadata` backend.** The `sqlite` backend's default file is `$BATCHER_HOME/metadata.db`, or `~/.batcher/metadata.db`, on the node that opened it, so each node learns alone. Only `object_storage` and `redis`, or `layered` in front of either, pointed at a location every driver reaches, share what one driver learned with the next. See {doc}`/configuration/options`.
- **Learning recorded inside a Ray worker stays in that worker.** A distributed `map_batches` stage runs in Ray workers, and the UDF sizing it learns there, such as a function's measured per-row cost and whether it runs in threads or processes, is written to the worker's own metadata store. The worker builds that store from its own process's config, because the driver's `config_context` doesn't cross the process boundary and the driver ships the engine's execution settings, not the `metadata` section. Unless the workers' own environment selects a shared backend through the `BATCHER_METADATA_*` variables, that is the default `in_process` store, so the learning never reaches the driver and is lost when the worker process exits.

## See also

- {doc}`Explain plans </user-guide/operate/tuning/explain-plans>`: reading a plan before you run it.
- {doc}`Performance </user-guide/operate/tuning/performance>`: turning what you saw here into a faster query.
- {doc}`Troubleshooting </user-guide/operate/running/troubleshooting>`: symptom-first debugging.
- {doc}`The terminal <terminal>` and {doc}`Metrics <metrics>`: the live line and the process-wide counters in depth.
- {doc}`/cookbook/operations/observability`: verbosity, logging, and execution statistics, as a script.
