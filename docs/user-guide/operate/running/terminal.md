# The terminal

This page describes what Batcher prints while a query runs and what it leaves behind when
one finishes. It is the surface you see without configuring anything: it renders into a real
terminal, and suppresses itself everywhere else.

In an interactive terminal, a query renders a live status line carrying the spinner,
operator, progress bar, rows, throughput, a throughput sparkline, elapsed time, and an ETA
when one can be known:

```text
⠹  filter            streaming     ▕████████████▋░░░░░░░░░░░▏  62%   241.6K rows   1.0M/s  ▁▃▅▆██▇  238ms  ETA 143ms
```

The third column is the phase, and it changes as the query moves through them, so a run
that is taking longer than you expected says *where* it is:

```text
⠹  aggregate         reading stats   ▕▒▓█▓▒░░░░░░░░░░░░░░░░░░░░▏   322ms
⠸  aggregate         optimizing      ▕░▒▓█▓▒░░░░░░░░░░░░░░░░░░░▏   330ms
⠼  aggregate         admission       ▕░░▒▓█▓▒░░░░░░░░░░░░░░░░░░▏   361ms
⠦  aggregate         on cluster      ▕░░░▒▓█▓▒░░░░░░░░░░░░░░░░░▏  4.59s
⠏  aggregate         learning stats  ▕░░░░▒▓█▓▒░░░░░░░░░░░░░░░░▏  6.35s
```

A slow small query raises one question first: did the time go to planning or to executing?
The phase answers it, because a query stuck on `optimizing` has a different problem from one
stuck on `on cluster`. The same phases are recorded
with their durations at `debug` verbosity, under `run phase`, when you want the numbers
rather than the live view.

On a distributed run the bar is driven by *partitions* rather than rows, because a stage
knows exactly how many buckets it has while the row estimate is a guess the query is in the
middle of disproving:

```text
⠹  shuffle           hash_join     ▕██████████▊░░░░░░░░░░░░░░▏  42%   27/64 parts   3.1M rows   840K/s  12.4s  ETA 17s
```

When it finishes, the line collapses to one aligned summary:

```text
✔  filter            383.5K rows  ·  400ms  ·  2.0M read  ·  5.0M rows/s
✘  join              PlanError: unknown column 'nope'
```

Throughput is the rows read per second, not the rows returned. A rate built from the
output understates the engine by exactly the query's selectivity: an aggregation reducing
400,000 rows to five in 100 ms is doing 4M rows/s, not 50. The rows read are shown beside it
whenever the two differ, so the rate always has a visible denominator.

The summary carries what else happened, when anything did. These are counted from the same
bus and appended to the success line rather than printed separately, because a caveat that
scrolls away from its result is a caveat nobody connects to it:

```text
✔  ingest            12.4M rows  ·  1m18s  ·  48.0M read  ·  615.4K rows/s  ·  3 inputs skipped  ·  spilled 4.2 GiB  ·  1x worker lost  ·  2x recompute
✔  wrote parquet      24 files  ·  12.4M rows  ·  3.1 GiB
```

That line separates a run that read every file from one that skipped part of its corpus,
and a job that was merely slow from one that survived losing workers.

A commit gets its own line. It lands after the query that produced the rows has finished,
because only then is there a manifest to report.

Two events do not wait for the end, because acting on them late is acting too late: a
fault-tolerance action on the distributed path, and a data-quality contract that failed.
Both print an immediate `!` line naming what happened and where.

## Queries answered without executing

Several shapes never reach the executor. A keyless `sum` or `count` can be read from a
source's own statistics, a `limit(n)` stops reading once it has `n` rows, a contradictory
predicate is provably empty, and a repeated small query can replay a prepared plan. These
are among the largest wins Batcher has. The summary says which one fired, in place of a
throughput figure it has no basis for:

```text
✔  aggregate         1 row  ·  0.4ms  ·  answered from source statistics, no scan
✔  scan              1 row  ·  0.2ms  ·  row count read from metadata, no scan
✔  limit             3 rows  ·  1.1ms  ·  stopped reading once the limit was met
```

Read those as a *good* result rather than a suspicious one: a `sum` over a billion-row table
returning in a millisecond has read the total the writer already computed. The reason
replaces the rate because dividing one output row by the time taken to fetch it produces
something like `12 rows/s`, which is the width of the answer rather than the speed of the
engine.

These count in the metrics export and appear in the dashboard like any other query, so a job
built out of `count()` and `agg()` reports the queries it ran. Each carries a pipeline
signature, so repeated runs group together rather than filing a row each.
{py:meth}`explain(analyze=True) <batcher.Dataset.explain>` and {py:meth}`stats() <batcher.Dataset.stats>` report the same way: both
run the query for real, so both leave a summary line, a dashboard row, a span and an
event-log document.

## Why the display behaves as it does

The bar advances in eighth-cells, which gives it eight times the resolution of its width
and is what makes it read as motion rather than as stepping blocks. Throughput is measured
over a trailing window rather than averaged since the query started, and that is what lets
the sparkline show a stall: a cumulative average thirty seconds into a run moves by a few
percent per second, so it stays flat through exactly the event you are watching for.

Live row counts only exist on the streaming path. {py:meth}`iter_batches <batcher.Dataset.iter_batches>` surfaces each Arrow
batch in Python, so counting rows there is free. `collect` measures inside Rust and returns
the profile at the end, so its bar shows an indeterminate sweep, the phase carries what is
happening, and the counts appear in the summary line. No row count is drawn until one has
been observed, because a standing `0 rows` would look like a reading of zero when it is the
absence of one.

Nothing else is invented either. With no row estimate and no partition count, the bar shows
an honest indeterminate sweep instead of a fabricated percentage, and the ETA is omitted
rather than guessed. That is the common case, because Kyber leaves an operator unbudgeted
whenever the source size is unknown.

Two smaller habits. The cursor is hidden while a bar animates and restored when the run ends
or the reporter is detached, so a job interrupted mid-query leaves neither a hidden cursor
nor a frozen line behind. And with several queries in flight, the most recently started run
is drawn and the rest are counted as `+4 more`, because one moving line is an instrument and
five interleaved ones are a mess.

Rendering degrades by detected capability rather than assuming one. Color falls back from
truecolor to 256-color to 16-color to none, and block-drawing falls back to ASCII.
`NO_COLOR`, `FORCE_COLOR`/`CLICOLOR_FORCE`, `COLORTERM`, and `TERM=dumb` are all honored,
and the ASCII forms are chosen so a `LANG=C` terminal gets readable output rather than
mojibake.

The whole thing is self-suppressing. Batcher renders escape codes only into a real TTY that
has not asked for plain output, so a script whose output you redirect to a file gets no bar
and no control characters. Continuous integration is suppressed by name as well, on `CI` or
`GITHUB_ACTIONS` in the environment: a CI runner is a terminal nobody is watching, and
thousands of repainted frames bury the output someone will actually read. Set it explicitly
when you need to:

```python
import batcher as bt
from batcher.config import ObservabilityConfig, active_config, set_config

set_config(
    active_config().replace(
        observability=ObservabilityConfig(progress="off")  # "auto" | "on" | "off"; None derives it
    )
)
```

`progress="on"` forces rendering, which helps inside a pseudo-terminal your tooling owns.
`"off"` disables it entirely.

## See also

- {doc}`Observability <observability>`: the event channel this line is one sink of, and the
  other three that read it.
- {doc}`Troubleshooting <troubleshooting>`: what the failure lines mean, by symptom.
- {doc}`Metrics <metrics>`: the process-wide counters behind the summary line.
- {doc}`Configuration </configuration/index>`: the full `observability` block, including
  `progress` and `verbosity`.
