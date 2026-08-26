"""The bus sink that owns the terminal: a live progress line and permanent status lines.

While a query runs this answers the three questions a person actually has — *which
operator*, *how far*, *how fast* — and when it finishes it leaves one aligned line behind,
so a script's output reads as a clean record rather than a wall of interleaved noise.

**It reports what happened, not only that something did.** The bus carries far more than
progress: inputs skipped as unreadable, rows dropped as malformed, bytes spilled, files
written, workers lost and recovered. None of that reached the terminal before, so a run
that quietly read 98% of its corpus, or that transparently survived losing two workers,
looked exactly like a clean one. Those are accumulated per query and reported on the
summary line, rather than printed as they arrive: a line per event would bury the summary
under the thing it summarizes.

**It refuses to render when rendering would be wrong.** Not a TTY (piped, CI, notebook),
`NO_COLOR`/`TERM=dumb`, or `progress="off"` — the live line is suppressed in every case.
Redraws are rate-limited: progress arrives per batch, and repainting faster than the eye
resolves would spend real time on frames nobody sees.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from collections.abc import Callable
from typing import TextIO

from batcher._internal import events
from batcher._internal.humanize import UNKNOWN, count, duration_ms, fit, rate
from batcher._internal.mathx import clamp
from batcher.observe.console.paint import LABEL_W, compose
from batcher.observe.console.state import RunState
from batcher.observe.theme import Palette, detect

__all__ = ["ConsoleReporter", "should_render"]

# ~20 fps. Past the point the eye resolves motion, and far below the per-batch event rate.
MIN_REDRAW_S = 0.05

_LEVEL_ROLE = {
    "DEBUG": "muted",
    "INFO": "accent",
    "WARNING": "warn",
    "ERROR": "critical",
    "CRITICAL": "critical",
}

# ANSI cursor control. Hiding the cursor while a bar is animating is the difference between
# a rendered instrument and one with a block flickering at the end of it; every progress
# renderer does this, and every one of them must restore it on the way out — including on
# an exception, which is why the reporter's detach is wrapped rather than handed out raw.
_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"
_ERASE_LINE = "\x1b[2K\r"


def should_render(mode: str, stream: TextIO | None = None) -> bool:
    """Whether a live progress bar is appropriate for `mode` on `stream`.

    ``"on"`` forces rendering, ``"off"`` disables it, and ``"auto"`` (the default) renders
    only into a real terminal that has not asked for plain output. Exposed so the same
    decision can be asserted in a test without constructing a reporter.

    Args:
        mode: One of ``"auto"``, ``"on"``, ``"off"``.
        stream: The output stream to inspect; defaults to `sys.stderr`.

    Returns:
        True if the caller should draw a live, escape-code-based progress bar.
    """
    if mode == "off":
        return False
    if mode == "on":
        return True
    stream = stream if stream is not None else sys.stderr
    if os.environ.get("NO_COLOR") is not None or os.environ.get("TERM") == "dumb":
        return False
    # Continuous-integration runners are terminals that nobody is watching: an animated bar
    # there produces thousands of repainted frames in the captured log and hides the output
    # someone will actually read. Every CI provider sets one of these.
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        # A closed or non-file-like stream (pytest capture, some notebook kernels).
        return False


class ConsoleReporter:
    """Renders bus events to a terminal: a live bar while running, a summary when done.

    Attach with `attach`, which returns the detach callable. Construct with ``live=False``
    to keep the structured lines but suppress the animated bar (what `should_render`
    decides for a non-TTY).

    Examples:
        .. doctest::

            >>> import io
            >>> from batcher.observe import ConsoleReporter
            >>> reporter = ConsoleReporter(stream=io.StringIO(), live=False)
            >>> detach = reporter.attach()
            >>> detach()
    """

    def __init__(self, *, stream: TextIO | None = None, live: bool = True) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._live = live
        self._palette, self._glyphs = detect(self._stream)
        if not live:
            # A non-live reporter writes into a log or a captured buffer; escape codes
            # there are corruption, not styling.
            self._palette = Palette(0)
        # Reentrant: a record emitted while rendering (a sink failure logged at DEBUG) can
        # re-enter `handle` on this same thread. The bus guards against the cycle, but a
        # plain Lock would still deadlock rather than degrade if any path slipped through.
        self._lock = threading.RLock()
        self._runs: dict[str, RunState] = {}
        self._last_draw = 0.0
        self._frame = 0
        self._painted = False
        self._cursor_hidden = False

    def attach(self) -> Callable[[], None]:
        """Subscribe this reporter to the event bus; returns the detach callable.

        The returned callable also clears any half-drawn bar and restores the cursor, so a
        reporter turned off mid-query does not leave the terminal with a hidden cursor and
        a frozen progress line — which outlives the process and is not obviously Batcher's
        doing when it happens.

        Returns:
            A zero-argument callable that detaches and restores the terminal.
        """
        unsubscribe = events.subscribe(self.handle)

        def _detach() -> None:
            unsubscribe()
            with self._lock:
                self._clear()
                self._show_cursor()

        return _detach

    # --- ingest -------------------------------------------------------------

    def handle(self, event: events.Event) -> None:
        """Render one bus event. This is the sink handed to `subscribe`."""
        with self._lock:
            kind = event.kind
            if kind == events.QUERY_START:
                self._runs[event.query_id] = RunState(
                    str(event.fields.get("label") or event.name or "query"),
                    event.fields.get("est_rows"),
                    event.ts,
                    str(event.fields.get("stage") or "running"),
                )
            elif kind == events.QUERY_END:
                self._finish(event)
                return
            elif kind == events.LOG:
                self._write_log(event)
                return
            elif not self._fold(event):
                return
            self._draw()

    def _fold(self, event: events.Event) -> bool:
        """Fold a mid-query event into its run's state; `False` if there is nothing to draw.

        One dispatch table rather than a chain of branches, because the set of things the
        engine reports grows and each addition should cost one entry rather than one more
        `elif` in a method that is already the hottest path in this module.
        """
        run = self._runs.get(event.query_id)
        if run is None:
            return False
        kind, fields = event.kind, event.fields
        if kind == events.STAGE_START:
            run.stage = event.name
            if fields.get("est_rows") is not None:
                run.est = fields["est_rows"]
        elif kind == events.STAGE_END:
            # The measured volume, which no other event carries: a stage that spilled 40 GiB
            # and one that spilled 4 MiB were both reported as "spilled" and nothing else.
            run.spilled_bytes += int(fields.get("spill_bytes", 0) or 0)
        elif kind == events.PROGRESS:
            run.observe(int(fields.get("rows", 0)), int(fields.get("bytes", 0) or 0))
        elif kind == events.PARTITION:
            run.note_partition(fields.get("total"), int(fields.get("rows", 0)))
        elif kind == events.SKIPPED:
            run.skipped += int(fields.get("count", 1))
        elif kind == events.MALFORMED:
            run.malformed += int(fields.get("count", 1))
        elif kind == events.WRITE:
            run.written_files += int(fields.get("files", 0))
            run.written_bytes += int(fields.get("bytes", 0) or 0)
        elif kind == events.RECOVERY:
            run.note_recovery(str(fields.get("event", "recovery")))
            self._notice("warn", f"{run.label}: {self._recovery_phrase(event)}")
            return False
        elif kind == events.DQ and not fields.get("ok", True):
            self._notice(
                "warn" if fields.get("severity") == "warn" else "critical",
                f"{run.label}: data-quality check {event.name!r} failed on "
                f"{count(fields.get('violations', 0))} of {count(fields.get('rows', 0))} rows",
            )
            return False
        return True

    @staticmethod
    def _recovery_phrase(event: events.Event) -> str:
        """One recovery action as a sentence a person can act on."""
        fields = event.fields
        what = str(fields.get("event", "recovery")).replace("_", " ")
        where = fields.get("worker") or fields.get("src") or fields.get("target")
        shuffle = fields.get("shuffle")
        parts = [f"({shuffle} shuffle)" if shuffle else "", f"on {where}" if where else ""]
        detail = " ".join(part for part in parts if part)
        return f"{what} {detail}".strip()

    # --- the live line ------------------------------------------------------

    def _draw(self) -> None:
        """Repaint the status line, rate-limited. Assumes `_lock` is held."""
        if not self._live or not self._runs:
            return
        now = time.monotonic()
        if now - self._last_draw < MIN_REDRAW_S:
            return
        self._last_draw = now
        self._frame += 1
        # With several queries in flight, render the most recently started and say how many
        # others there are.
        run = self._runs[next(reversed(self._runs))]
        run.tick(now)
        self._hide_cursor()
        self._paint(
            compose(
                run,
                now,
                palette=self._palette,
                glyphs=self._glyphs,
                frame=self._frame,
                bar_width=self._bar_width(),
                others=len(self._runs) - 1,
            )
        )

    def _bar_width(self) -> int:
        """Bar width, scaled to the terminal and clamped to a legible range."""
        columns = shutil.get_terminal_size((100, 24)).columns
        return clamp((columns - 74) // 2 + 12, 12, 28)

    # --- permanent lines ----------------------------------------------------

    def _finish(self, event: events.Event) -> None:
        """Clear the live line and print the query's one-line summary."""
        p, g = self._palette, self._glyphs
        run = self._runs.pop(event.query_id, None)
        label = run.label if run else str(event.fields.get("label", "query"))
        ok = bool(event.fields.get("ok", True))
        ms = float(event.fields.get("total_ms", 0.0))
        rows = int(event.fields.get("rows", 0))
        if not ok:
            head = f"{p.critical}{g.fail}{p.reset}  {p.bold}{fit(label, LABEL_W)}{p.reset}"
            self._emit_line(f"{head}  {p.critical}{event.fields.get('error', '')}{p.reset}")
            return
        parts = [f"{count(rows)} rows", duration_ms(ms)]
        if ms > 0 and rows:
            parts.append(rate(rows / (ms / 1000)))
        if run is not None:
            written = run.written()
            if written:
                parts.append(written)
        detail = f"  {p.muted}{g.sep}{p.reset}  ".join(parts)
        head = f"{p.good}{g.ok}{p.reset}  {p.bold}{fit(label, LABEL_W)}{p.reset}"
        line = f"{head}  {p.dim}{detail}{p.reset}"
        anomalies = run.anomalies() if run is not None else []
        if anomalies:
            # Appended to the success line rather than printed separately: the query did
            # succeed, and a caveat that scrolls away from its result is a caveat nobody
            # connects to it.
            joined = f"  {g.sep}  ".join(anomalies)
            line += f"  {p.warn}{g.sep}  {joined}{p.reset}"
        self._emit_line(line)

    def _notice(self, role: str, text: str) -> None:
        """Print one immediate, unmissable line for an event that cannot wait for the end.

        Only for the rare and consequential: a recovery action, a failed data-quality
        contract. Everything countable is accumulated and reported once, on the summary.
        """
        p = self._palette
        color = getattr(p, role, "")
        self._emit_line(f"{color}!{p.reset}  {text}")

    def _write_log(self, event: events.Event) -> None:
        """Print one structured log record: timestamp, level, logger, message, key=values.

        The field layout is **logfmt** (``key=value`` pairs, the Heroku/Go convention) with
        a fixed-width prefix, so the same line is aligned for a human reading a terminal and
        parseable by a log processor without a regex per message. Field *names* follow the
        OpenTelemetry convention of a unit suffix (``duration_ms``) so a number's meaning
        does not depend on prose.
        """
        p = self._palette
        level = str(event.fields.get("level", "INFO"))
        color = getattr(p, _LEVEL_ROLE.get(level, "accent"), "")
        stamp = time.strftime("%H:%M:%S", time.localtime(event.wall))
        fields = event.fields.get("fields") or {}
        kv = "".join(
            f"  {p.dim}{k}={p.reset}{p.muted}{_fmt_value(v)}{p.reset}" for k, v in fields.items()
        )
        self._emit_line(
            f"{p.muted}{stamp}{p.reset} {color}{level:<8}{p.reset}"
            f"{p.dim}{fit(event.name or 'engine', 13)}{p.reset} "
            f"{event.fields.get('message', '')}{kv}"
        )

    def _emit_line(self, text: str) -> None:
        """Write a permanent line, erasing the transient bar first so they never collide."""
        self._clear()
        self._write(text + "\n")
        # The bar owns the last line; repaint at once rather than leaving it missing for up
        # to `MIN_REDRAW_S` after every record.
        self._last_draw = 0.0
        self._draw()
        if not self._runs:
            self._show_cursor()

    def _clear(self) -> None:
        """Erase the transient line if one is painted."""
        if self._painted:
            self._write(_ERASE_LINE)
            self._painted = False

    def _paint(self, text: str) -> None:
        self._write(_ERASE_LINE + text)
        self._painted = True

    def _hide_cursor(self) -> None:
        if self._live and not self._cursor_hidden:
            self._write(_HIDE_CURSOR)
            self._cursor_hidden = True

    def _show_cursor(self) -> None:
        if self._cursor_hidden:
            self._write(_SHOW_CURSOR)
            self._cursor_hidden = False

    def _write(self, text: str) -> None:
        """Write to the stream, swallowing a closed or broken one.

        A reporter attached in a notebook or a daemon can outlive its stream; failing to
        print progress must never surface as an exception from the query being observed.
        """
        try:
            self._stream.write(text)
            self._stream.flush()
        except (ValueError, OSError):  # pragma: no cover - closed/broken stream
            self._live = False


def _fmt_value(value: object) -> str:
    """One logfmt value — quoted only when it contains a space, as the convention requires."""
    if value is None:
        return UNKNOWN
    text = "true" if value is True else "false" if value is False else str(value)
    return f'"{text}"' if " " in text else text
