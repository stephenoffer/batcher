# Observability

Batcher reports what it is doing through **one** channel. Every subsystem publishes to a
single internal event bus, including the Kyber optimizer, the Carbonite resource manager,
the Core executor, and the distributed scheduler. Everything you can *see* is a consumer
of that bus:

| Surface | What it is for | Default |
| --- | --- | --- |
| Terminal progress bar | watching a query run, interactively | on in a real terminal |
| Structured logs | what the engine decided, and why | `WARNING` and above |
| Web dashboard | plans, per-operator timings, throughput, live logs | off (`bt.start_ui()`) |
| JSON event log | the durable per-query artifact, on disk | on |

Because they share one source, they can never disagree: the timeline in the dashboard and
the profile in the on-disk event log are the same measurements, under the same query id.

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
ObservabilityConfig(verbosity=4)          # same as "debug"
```

Or from the environment, without touching code:

```bash
BATCHER_OBSERVABILITY_VERBOSITY=trace python job.py
```

**`trace` is the only level where the two ladders differ.** Python's `logging` has no level
below `DEBUG`, but the engine's Rust `tracing` spans do, and that is where per-morsel work
is visible. So `trace` means `DEBUG` in Python and `TRACE` in Rust.

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

In an interactive terminal, a query renders a live status line carrying the spinner,
operator, progress bar, rows, throughput, a throughput sparkline, elapsed time, and an ETA
when one can be known:

```text
⠹  filter            streaming     ▕████████████▋░░░░░░░░░░░▏  62%   241.6K rows   1.0M/s  ▁▃▅▆██▇  238ms  ETA 143ms
```

When it finishes, the line collapses to one aligned summary:

```text
✔  filter            383.5K rows  ·  400ms  ·  959.9K rows/s
✘  join              PlanError: unknown column 'nope'
```

Three details worth knowing, because they are deliberate:

- **The bar advances in eighth-cells**, giving it eight times the resolution of its width,
  which is what makes it read as motion rather than as stepping blocks.
- **Live row counts only exist on the streaming path.** `iter_batches` surfaces each Arrow
  batch in Python, so counting rows there is free. `collect` measures inside Rust and returns
  the profile at the end, so its bar shows an indeterminate sweep and its counts appear in the
  summary line.
- **Nothing is invented.** With no row estimate, the bar shows an honest indeterminate
  sweep instead of a fabricated percentage, and the ETA is omitted rather than guessed.
  That is the common case, because Kyber leaves an operator unbudgeted whenever the source
  size is unknown.

Rendering degrades by detected capability rather than assuming one. Color falls back from
truecolor to 256-color to 16-color to none, and block-drawing falls back to ASCII.
`NO_COLOR`, `FORCE_COLOR`/`CLICOLOR_FORCE`,
`COLORTERM`, and `TERM=dumb` are all honored, and the ASCII forms are chosen so a `LANG=C`
terminal gets readable output rather than mojibake.

This is **automatic and self-suppressing**. Batcher renders escape codes only into a real
TTY that has not asked for plain output, so a script whose output you redirect to a file,
or that runs under CI, gets no bar and no control characters. Set it explicitly when you
need to:

```python
import batcher as bt
from batcher.config import ObservabilityConfig, active_config, set_config

set_config(active_config().replace(
    observability=ObservabilityConfig(progress="off")   # "auto" | "on" | "off"; None derives it
))
```

`progress="on"` forces rendering, which helps inside a pseudo-terminal your tooling owns.
`"off"` disables it entirely. The `NO_COLOR` and `TERM=dumb` conventions are honored.

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
set_config(active_config().replace(
    observability=ObservabilityConfig(
        log_level="INFO",          # CRITICAL | ERROR | WARNING | INFO | DEBUG
        log_format="json",         # "human" (default) or "json" for a log shipper
        log_file="/var/log/batcher.log",
        console=False,             # file-only
    )
))
```

Engine log records carry **structured fields**, not only a sentence. The terminal layout is
[logfmt](https://brandur.org/logfmt), the `key=value` convention from Heroku and the Go
ecosystem, behind a fixed-width prefix. One line is therefore both aligned for a human and
parseable by a log processor without a bespoke regex per message. Values are quoted only when
they contain a space, as the convention requires. Field *names* follow the OpenTelemetry
practice of carrying their unit, so the field is `duration_ms` rather than `duration` and a
number's meaning never depends on surrounding prose.

```text
14:19:25  INFO     kyber        join reorder  tables=3 cost=1.25 note="two words"
```

```json
{"time": "2024-05-02T14:19:25Z", "level": "INFO", "logger": "batcher.kyber",
 "message": "join reorder", "fields": {"tables": 3, "cost": 1.25}}
```

That is why the same record is greppable at the terminal *and* queryable in your log
platform without anyone re-parsing prose.

The `log_level` also drives the Rust data plane's tracing, so raising it to `DEBUG` reveals
the engine's per-operator work, not only the Python control plane's.

## The web dashboard

Start it and keep working. It runs on its own port, in a daemon thread, and never blocks
the process it is observing:

```python
import batcher as bt

bt.start_ui()                    # returns 'http://127.0.0.1:4040'
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
`$BATCHER_HOME/pipelines.json` (default `~/.batcher/pipelines.json`), so a pipeline you
named is still named after a restart. Everything else on the dashboard is a measurement
that ages out of memory; the name is a fact about the pipeline as a thing you return to.

Within a run, **Steps** offers the plan graph, the pipeline stages, a flame view, a
ranked list, and a sortable table. Each is annotated with rows out, elapsed time, spill
volume, and the estimate Kyber planned for. Actual against estimate is the single most
useful number when a query is slow for a reason the plan did not predict.

**Query** shows the plan as a document, in three forms:

- **Explain**: the plan as a text tree, annotated with what each step measured. Toggle
  "Show the plan as written" to see the plan before the optimizer touched it.
- **What the optimizer changed**: the two plans compared. A pushdown is reported as one
  rewrite, with the steps it dragged past it listed separately rather than as four equal
  findings. This has no direct equivalent in other engines' UIs, which show one plan at
  a time.
- **Plan document**: the exact JSON IR that crossed into the Rust engine, for when the
  rendering is the thing under suspicion.

**Findings** carries what the engine concluded: the per-run insights, the optimizer and
resource-manager decisions, and any **adaptive re-optimization**, meaning the points where the
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

New to the dashboard, or arriving from another engine? The **Learn** page maps the panel
you already know, such as a Spark UI tab, an Airflow view, or a DuckDB `EXPLAIN`, to its
equivalent here.

`start_ui` is idempotent. Calling it again returns the URL of the dashboard already
running rather than binding a second port. Ask for that URL at any time with `bt.ui_url()`,
which returns `None` when no dashboard is running. That helps when a helper needs to print
or link the dashboard without caring who started it:

```python
if bt.ui_url() is None:
    bt.start_ui()
print(f"dashboard: {bt.ui_url()}")
```

Passing `port=0` asks the OS for any free port, which is the right choice in tests and in
any environment where 4040 may already be taken. Read back the actual port from the
returned URL or from `bt.ui_url()`.

Stop it when you are done, or let the process exit and it is cleaned up automatically:

```python
bt.stop_ui()
```

To have a long-running service always expose it, turn it on in config instead of calling
`start_ui` by hand:

```python
set_config(active_config().replace(
    observability=ObservabilityConfig(ui=True, ui_port=4040)
))
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
set_config(active_config().replace(
    observability=ObservabilityConfig(
        event_log=True,                     # on by default
        event_log_dir="/data/batcher-events",
        event_log_max_files=1000,           # oldest pruned on write; 0 = unbounded
    )
))
```

Turning it off with `event_log=False` removes the per-query write. That is worth doing only
if you run many small queries and nothing consumes the documents.

## Metrics

The event bus is the right tool when you want every detail of one query. When you want a
handful of numbers scraped every fifteen seconds forever, use the counters instead.
`metrics_snapshot` returns them as a nested dict of plain numbers, with no Batcher types in
it and nothing to close:

```python
from batcher.observe import metrics_snapshot

snap = metrics_snapshot()
print(sorted(snap))
# ['bytes', 'gpu', 'inference', 'logs', 'operators', 'partitions', 'queries', 'rows',
#  'skipped', 'spills', 'uptime_seconds']
```

The last four fill in only for work that reports them: `partitions` and `skipped` for a
distributed read, `inference` and `gpu` for a batch-inference pass. A single-node
relational query leaves them at zero rather than absent, so a scraper never has to handle
a changing key set.

If the dashboard is already running, the same counters are served from it, so a scrape
loop needs no code of yours at all:

```bash
curl -s http://127.0.0.1:4040/metrics        # Prometheus text exposition
curl -s http://127.0.0.1:4040/api/metrics    # the same numbers as JSON
```

The whole read-only API lists itself at `/api`, and every route there is a `GET` over what
the dashboard is showing.

Counters are cumulative from the moment collection starts, the convention every metrics
backend expects, so a scrape loop differences successive snapshots to get rates. Collecting
costs a few integer adds per event.

The first snapshot starts collection, which means it reports only what happened after it.
Call `start_metrics()` once during startup when the first scrape should also cover the
queries that ran before it:

```python
from batcher.observe import start_metrics

start_metrics()
```

Collection is opt-in rather than always-on for a reason. Attaching any sink to the event
bus tells the engine that per-query profiles are being consumed, so it assembles one on
every query. A process that exports no metrics shouldn't pay that on the sub-second path.

`prometheus_text` renders the same numbers in the Prometheus text exposition format. Serve
it from the `/metrics` endpoint your application already has and Batcher joins whatever you
scrape today. Batcher runs no HTTP server for this and pulls in no client library:

```python
from batcher.observe import prometheus_text

print("batcher_queries_total" in prometheus_text())
# True
```

Series are prefixed `batcher_`, counters carry the conventional `_total` suffix, and query
duration is a real histogram with `_bucket`, `_sum`, and `_count` series. OpenTelemetry and
StatsD users can map the same dict onto their own instruments. `reset_metrics` zeroes the
counters, for tests and for a service that would rather report per-interval numbers itself.

## OpenTelemetry

If your organization already collects traces, emit into them rather than adding a second
pane of glass. Batcher emits one span per query with a child span per operator, into the
tracer your application configured. Batcher owns no exporter:

```python
set_config(active_config().replace(
    observability=ObservabilityConfig(otel_traces=True)
))
```

This needs the `otel` extra, `pip install 'batcher-engine[otel]'`, plus a provider the host
app sets up. It reuses the same measured profile as the event log, so enabling it adds the
span emit and no extra measurement.

## See also

- {doc}`Explain plans </user-guide/operate/tuning/explain-plans>`: reading a plan before you run it.
- {doc}`Performance </user-guide/operate/tuning/performance>`: turning what you saw here into a faster query.
- {doc}`Troubleshooting </user-guide/operate/running/troubleshooting>`: symptom-first debugging.
- {doc}`/cookbook/operations/observability`: verbosity, logging, and execution statistics, as a script.
