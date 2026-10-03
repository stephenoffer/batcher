# The terminal

This page describes what Batcher prints while a query runs and the one line it leaves behind when it finishes. It needs no configuration: it renders into a real terminal and suppresses itself everywhere else.

## The live status line

Run any query in an interactive terminal and a status line shows the operator, phase, progress, rows, throughput, a sparkline, elapsed time, and an ETA when one is known:

```python
import batcher as bt

events = bt.from_pydict({"user": ["a", "b", "a", "c"], "ms": [120, 80, 45, 300]})
print(events.filter(bt.col("ms") > 50).group_by("user").agg(total=bt.col("ms").sum()).sort("user").to_pydict())
# {'user': ['a', 'b', 'c'], 'total': [120, 80, 300]}
```

```text
⠹  filter            streaming     ▕████████████▋░░░░░░░░░░░▏  62%   241.6K rows   1.0M/s  ▁▃▅▆██▇  238ms  ETA 143ms
```

The third column is the phase, so a slow query says *where* it is. A query stuck on `optimizing` has a different problem from one stuck on `on cluster`:

```text
⠹  aggregate         reading stats   ▕▒▓█▓▒░░░░░░░░░░░░░░░░░░░░▏   322ms
⠸  aggregate         optimizing      ▕░▒▓█▓▒░░░░░░░░░░░░░░░░░░░▏   330ms
⠼  aggregate         admission       ▕░░▒▓█▓▒░░░░░░░░░░░░░░░░░░▏   361ms
⠦  aggregate         on cluster      ▕░░░▒▓█▓▒░░░░░░░░░░░░░░░░░▏  4.59s
```

On a distributed run the bar counts *partitions*, because a stage knows exactly how many buckets it has:

```text
⠹  shuffle           hash_join     ▕██████████▊░░░░░░░░░░░░░░▏  42%   27/64 parts   3.1M rows   840K/s  12.4s  ETA 17s
```

Live row counts come from the streaming path. {py:meth}`iter_batches <batcher.Dataset.iter_batches>` surfaces each Arrow batch in Python, so the bar can count rows as they arrive:

```python
rows = 0
for batch in events.filter(bt.col("ms") > 50).iter_batches():
    rows += batch.num_rows
print(rows)
# 3
```

`collect` measures inside Rust, so its bar shows a sweep and the counts appear in the summary line.

## The summary line

When a query finishes, the line collapses to one aligned summary:

```text
✔  filter            383.5K rows  ·  400ms  ·  2.0M read  ·  5.0M rows/s
✘  join              PlanError: unknown column 'nope'
```

Throughput is rows *read* per second, not rows returned, so an aggregation reducing 400,000 rows to five in 100 ms reports 4M rows/s.

The summary also records what else happened: inputs skipped, bytes spilled, workers lost. A commit gets its own line:

```text
✔  ingest            12.4M rows  ·  1m18s  ·  48.0M read  ·  615.4K rows/s  ·  3 inputs skipped  ·  spilled 4.2 GiB  ·  1x worker lost  ·  2x recompute
✔  wrote parquet      24 files  ·  12.4M rows  ·  3.1 GiB
```

A fault-tolerance action and a failed data-quality contract don't wait for the end. Each prints an immediate `!` line naming what happened.

## Queries answered without executing

Several shapes never reach the executor. A keyless `count` or `sum` can come from source statistics, and a `limit(n)` stops reading once it has `n` rows:

```python
print(events.count())
# 4
print(events.limit(2).to_pydict())
# {'user': ['a', 'b'], 'ms': [120, 80]}
```

The summary names the shortcut in place of a throughput figure:

```text
✔  aggregate         1 row  ·  0.4ms  ·  answered from source statistics, no scan
✔  scan              1 row  ·  0.2ms  ·  row count read from metadata, no scan
✔  limit             3 rows  ·  1.1ms  ·  stopped reading once the limit was met
```

These still count in the metrics export and appear in the dashboard, grouped by pipeline signature. {py:meth}`explain(analyze=True) <batcher.Dataset.explain>` and {py:meth}`stats() <batcher.Dataset.stats>` run the query for real, so they leave a summary line too.

## Turn it on or off

The display renders only into a real TTY and is suppressed under `CI` or `GITHUB_ACTIONS`. Set `progress` to force it either way, for one block or for the session:

```python
from batcher.config import ObservabilityConfig, active_config, config_context

quiet = active_config().replace(observability=ObservabilityConfig(progress="off"))  # "auto" | "on" | "off"
with config_context(quiet):
    print(events.agg(total=bt.col("ms").sum()).to_pydict())
# {'total': [545]}
```

`progress="on"` forces rendering, which helps inside a pseudo-terminal your tooling owns. The same phases are logged with their durations at `debug` verbosity, under `run phase`.

:::{dropdown} Rendering details
The bar advances in eighth-cells, giving it eight times the resolution of its width. Throughput is measured over a trailing window rather than averaged since the start, so the sparkline shows a stall as it happens.

With no row estimate and no partition count, the bar shows an indeterminate sweep and the ETA is omitted. No row count is drawn until one has been observed.

The cursor is hidden while a bar animates and restored when the run ends or the reporter detaches. With several queries in flight, the most recent one is drawn and the rest are counted as `+4 more`.

Color falls back from truecolor to 256-color to 16-color to none, and block-drawing falls back to ASCII. `NO_COLOR`, `FORCE_COLOR`/`CLICOLOR_FORCE`, `COLORTERM`, and `TERM=dumb` are all honored.
:::

## See also

- {doc}`Observability <observability>`: the event channel this line is one sink of, and the
  other three that read it.
- {doc}`Troubleshooting <troubleshooting>`: what the failure lines mean, by symptom.
- {doc}`Metrics <metrics>`: the process-wide counters behind the summary line.
- {doc}`Configuration </configuration/index>`: the full `observability` block, including
  `progress` and `verbosity`.
