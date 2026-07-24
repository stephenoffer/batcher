"""The terminal face of the engine — a live progress bar plus structured status lines.

A bus sink that renders to stderr. While a query runs it answers the three questions a
person actually has — *which operator*, *how far*, *how fast* — and when it finishes it
leaves one aligned line behind, so a script's output reads as a clean record rather than a
wall of interleaved noise.

**Design.** Sleek and technical, with the retro register coming from typography rather than
from color: block-drawing bars, a braille spinner, a throughput sparkline, box-rule
separators. Layout is fixed-column and tabular so numbers line up as they change — the
thing that separates an instrument from a status dump. Colors are the same sequential blue
ramp the dashboard uses, because progress is a magnitude.

**Animation is real, not decorative.** The bar advances in eighth-cells (8x the resolution
of its width), so it reads as motion instead of stepping. Rates are exponentially smoothed
so the number is steady enough to read. Nothing is invented: with no row estimate the bar
shows an honest indeterminate sweep rather than a fabricated percentage, and the ETA is
omitted rather than guessed.

Written against bare ANSI and the stdlib — no `rich`, no `tqdm`. A data engine should not
make a user install a rendering library to see a progress bar. Capability detection and the
glyph/color tables live in `theme`; degradation to 16-color or ASCII is designed there.

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
from collections import deque
from collections.abc import Callable
from typing import TextIO

from batcher._internal import events
from batcher._internal.mathx import clamp
from batcher.observe.theme import Palette, detect

__all__ = ["ConsoleReporter", "should_render"]

# ~20 fps. Past the point the eye resolves motion, and far below the per-batch event rate.
_MIN_REDRAW_S = 0.05
# Smoothing factor for the displayed rate. Low enough that the number is readable, high
# enough that it still tracks a real change in throughput within a second or so.
_RATE_ALPHA = 0.3
# Sparkline history depth — one sample per repaint, so ~1.5s of throughput at 20 fps.
_SPARK_N = 30
_LABEL_W, _STAGE_W = 18, 14

_LEVEL_ROLE = {
    "DEBUG": "muted",
    "INFO": "accent",
    "WARNING": "warn",
    "ERROR": "critical",
    "CRITICAL": "critical",
}


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
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        # A closed or non-file-like stream (pytest capture, some notebook kernels).
        return False


class _Run:
    """One in-flight query's live state — what the status line is drawn from."""

    __slots__ = ("est", "label", "rate", "rows", "spark", "stage", "t0")

    def __init__(self, label: str, est: float | None, t0: float, stage: str = "running") -> None:
        self.label = label
        # "running", not "planning": the engine does not report a phase transition on the
        # `collect` path, so a hardcoded "planning" would have claimed, for the entire
        # duration of every query, a phase that ended in its first millisecond. `STAGE_START`
        # overwrites this whenever a real operator name is available.
        self.stage = stage
        self.rows = 0
        self.est = est
        self.t0 = t0
        self.rate = 0.0
        self.spark: deque[float] = deque(maxlen=_SPARK_N)

    def observe(self, rows: int) -> None:
        """Fold a progress batch into the row total."""
        self.rows += rows

    def tick(self, now: float) -> None:
        """Update the smoothed rate and the sparkline history for one repaint."""
        elapsed = max(now - self.t0, 1e-9)
        instant = self.rows / elapsed
        self.rate = instant if self.rate == 0.0 else self.rate + _RATE_ALPHA * (instant - self.rate)
        self.spark.append(self.rate)

    @property
    def fraction(self) -> float | None:
        """Completed fraction, or `None` when the operator was left unbudgeted."""
        return None if not self.est or self.est <= 0 else self.rows / self.est

    @property
    def eta_s(self) -> float | None:
        """Seconds remaining at the smoothed rate, or `None` when it cannot be known."""
        if not self.est or self.rate <= 0 or self.rows >= self.est:
            return None
        return (self.est - self.rows) / self.rate


class ConsoleReporter:
    """Renders bus events to a terminal: a live bar while running, a summary when done.

    Attach with `attach`, which returns the detach callable. Construct with ``live=False``
    to keep the structured lines but suppress the animated bar (what `should_render`
    decides for a non-TTY).
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
        self._runs: dict[str, _Run] = {}
        self._last_draw = 0.0
        self._frame = 0
        self._painted = False

    def attach(self) -> Callable[[], None]:
        """Subscribe this reporter to the event bus; returns the detach callable."""
        return events.subscribe(self.handle)

    # --- ingest -------------------------------------------------------------

    def handle(self, event: events.Event) -> None:
        """Render one bus event. This is the sink handed to `subscribe`."""
        with self._lock:
            kind = event.kind
            if kind == events.QUERY_START:
                self._runs[event.query_id] = _Run(
                    str(event.fields.get("label") or event.name or "query"),
                    event.fields.get("est_rows"),
                    event.ts,
                    str(event.fields.get("stage") or "running"),
                )
            elif kind == events.STAGE_START:
                run = self._runs.get(event.query_id)
                if run is not None:
                    run.stage = event.name
                    if event.fields.get("est_rows") is not None:
                        run.est = event.fields["est_rows"]
            elif kind == events.PROGRESS:
                run = self._runs.get(event.query_id)
                if run is not None:
                    run.observe(int(event.fields.get("rows", 0)))
            elif kind == events.QUERY_END:
                self._finish(event)
                return
            elif kind == events.LOG:
                self._write_log(event)
                return
            self._draw()

    # --- the live line ------------------------------------------------------

    def _draw(self) -> None:
        """Repaint the status line, rate-limited. Assumes `_lock` is held."""
        if not self._live or not self._runs:
            return
        now = time.monotonic()
        if now - self._last_draw < _MIN_REDRAW_S:
            return
        self._last_draw = now
        self._frame += 1
        # With several queries in flight, render the most recently started: one moving
        # line is an instrument, N interleaved ones are a mess.
        run = self._runs[next(reversed(self._runs))]
        run.tick(now)
        self._paint(self._compose(run, now))

    def _compose(self, run: _Run, now: float) -> str:
        """The status line: spinner, label, stage, bar, rows, rate, elapsed, ETA."""
        p, g = self._palette, self._glyphs
        spin = g.spinner[self._frame % len(g.spinner)]
        elapsed = now - run.t0
        fraction = run.fraction

        cells = []
        cells.append(f"{p.accent}{spin}{p.reset}")
        cells.append(f"{p.bold}{_pad(run.label, _LABEL_W)}{p.reset}")
        cells.append(f"{p.muted}{_pad(run.stage, _STAGE_W)}{p.reset}")
        cells.append(self._bar(fraction))
        if fraction is not None:
            cells.append(f"{_pct(fraction):>4}")
        cells.append(f"{p.dim}{_count(run.rows):>7}{p.reset} rows")
        if run.rate > 0:
            cells.append(f"{p.dim}{_count(run.rate):>7}/s{p.reset}")
            spark = self._sparkline(run.spark)
            if spark:
                cells.append(spark)
        cells.append(f"{p.muted}{_dur(elapsed * 1000):>7}{p.reset}")
        eta = run.eta_s
        if eta is not None:
            cells.append(f"{p.muted}ETA {_dur(eta * 1000)}{p.reset}")
        return "  ".join(cells)

    def _bar(self, fraction: float | None) -> str:
        """The progress bar: a gradient eighth-cell fill, or an honest sweep when unknown.

        An unknown total is the *common* case — Kyber leaves an operator unbudgeted whenever
        the source size is unknown — so the indeterminate form has to be a first-class
        design, not a fallback. It is a comet sweeping the track, which says "working, total
        unknown"; inventing a denominator would produce a bar that jumps backwards the
        moment the estimate is beaten.
        """
        p, g = self._palette, self._glyphs
        width = self._bar_width()
        if fraction is None:
            return f"{g.cap_l}{self._sweep(width)}{g.cap_r}"
        clamped = 0.0 if fraction < 0 else 1.0 if fraction > 1 else fraction
        exact = clamped * width
        full = int(exact)
        out = []
        # Tint each filled cell by its position, so the fill is one gradient object.
        for i in range(full):
            out.append(f"{p.ramp(i / max(width - 1, 1))}{g.full}")
        if full < width:
            remainder = exact - full
            if remainder > 0 and g.unicode:
                out.append(f"{p.ramp(full / max(width - 1, 1))}{g.eighths[int(remainder * 8)]}")
                full += 1
            out.append(f"{p.muted}{g.empty * (width - full)}")
        return f"{g.cap_l}{''.join(out)}{p.reset}{g.cap_r}"

    def _sweep(self, width: int) -> str:
        """An indeterminate comet: a bright head with a fading trail, bouncing end to end."""
        p, g = self._palette, self._glyphs
        period = max(width * 2 - 2, 1)
        pos = self._frame % period
        if pos >= width:
            pos = period - pos
        out = []
        for i in range(width):
            distance = abs(i - pos)
            if distance < len(g.shades):
                out.append(f"{p.ramp(1 - distance / len(g.shades))}{g.shades[distance]}")
            else:
                out.append(f"{p.muted}{g.empty}")
        return "".join(out) + p.reset

    def _sparkline(self, history: deque[float]) -> str:
        """Recent throughput as a sparkline, or `""` until there are enough samples.

        Shows the *shape* of throughput — a stall, a ramp, a stutter — which a single
        smoothed number cannot. Scaled to its own window's peak, so it reads as relative
        change rather than as an absolute the axis-less form could not convey anyway.
        """
        p, g = self._palette, self._glyphs
        if len(history) < 4:
            return ""
        peak = max(history)
        if peak <= 0:
            return ""
        cells = "".join(
            g.spark[min(int(v / peak * len(g.spark)), len(g.spark) - 1)] for v in history
        )
        return f"{p.muted}{cells}{p.reset}"

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
            head = f"{p.critical}{g.fail}{p.reset}  {p.bold}{_pad(label, _LABEL_W)}{p.reset}"
            self._emit_line(f"{head}  {p.critical}{event.fields.get('error', '')}{p.reset}")
            return
        parts = [f"{_count(rows)} rows", _dur(ms)]
        if ms > 0 and rows:
            parts.append(f"{_count(rows / (ms / 1000))} rows/s")
        detail = f"  {p.muted}{g.sep}{p.reset}  ".join(parts)
        head = f"{p.good}{g.ok}{p.reset}  {p.bold}{_pad(label, _LABEL_W)}{p.reset}"
        self._emit_line(f"{head}  {p.dim}{detail}{p.reset}")

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
            f"{p.dim}{_pad(event.name or 'engine', 13)}{p.reset} "
            f"{event.fields.get('message', '')}{kv}"
        )

    def _emit_line(self, text: str) -> None:
        """Write a permanent line, erasing the transient bar first so they never collide."""
        if self._painted:
            self._write("\x1b[2K\r")
            self._painted = False
        self._write(text + "\n")
        # The bar owns the last line; repaint at once rather than leaving it missing for up
        # to `_MIN_REDRAW_S` after every record.
        self._last_draw = 0.0
        self._draw()

    def _paint(self, text: str) -> None:
        self._write("\x1b[2K\r" + text)
        self._painted = True

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


# --- formatting -------------------------------------------------------------


def _pad(text: str, width: int) -> str:
    """`text` clipped to `width` and left-padded to it, so columns stay aligned."""
    if len(text) > width:
        return text[: width - 1] + "…"
    return text.ljust(width)


def _count(n: float) -> str:
    """A compact SI-style count: ``1.2K``, ``3.4M``, ``5.6B``."""
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= limit:
            return f"{n / limit:.1f}{suffix}"
    return f"{n:.0f}"


def _dur(ms: float) -> str:
    """A duration in the unit a person would say it in: ``820ms``, ``4.2s``, ``1m03s``.

    Below 10ms one decimal is kept, matching the web UI's ``UI.ms`` exactly so the same
    operator time never reads as ``4.2ms`` in the browser and ``4ms`` in the terminal. A
    differential test pins the two together.
    """
    if ms < 1000:
        return f"{ms:.1f}ms" if ms < 10 else f"{ms:.0f}ms"
    if ms < 60_000:
        return f"{ms / 1000:.2f}s"
    minutes, seconds = divmod(ms / 1000, 60)
    return f"{minutes:.0f}m{seconds:02.0f}s"


def _pct(fraction: float) -> str:
    """A clamped integer percentage, e.g. ``62%``.

    Matches the web UI's ``UI.pct`` exactly, including its two edges: a small-but-present
    share reads ``<1%`` rather than ``0%`` (which would claim nothing is there). A share
    above 100% is left as-is — operator time summed across threads legitimately exceeds
    wall-clock, so it is a real reading. A differential test pins the two together.
    """
    if fraction is None:
        return "—"
    if not fraction:
        return "0%"
    value = max(fraction, 0.0) * 100
    if 0 < value < 1:
        return "<1%"
    return f"{value:.0f}%"


def _bytes(n: float | None) -> str:
    """A binary byte size, e.g. ``1.5 KiB``, matching the web UI's ``UI.bytes`` exactly.

    ``None`` and ``0`` both read as an em dash — the terminal has no more use than the web
    for "0 bytes" as a distinct value from "nothing measured".
    """
    if not n:
        return "—"
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return str(n)


def _fmt_value(value: object) -> str:
    """One logfmt value — quoted only when it contains a space, as the convention requires."""
    text = "true" if value is True else "false" if value is False else str(value)
    return f'"{text}"' if " " in text else text
