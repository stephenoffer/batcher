"""Drawing one status line: the bar, the indeterminate sweep, and the sparkline.

Pure functions of a `RunState` and a theme. Nothing here writes to a stream or holds a
lock, which is what lets a test assert the exact glyphs of a 62%-full bar without a
terminal and without a query.

Written against bare ANSI and the stdlib — no `rich`, no `tqdm`. A data engine should not
make a user install a rendering library to see a progress bar.
"""

from __future__ import annotations

from collections.abc import Sequence

from batcher._internal.humanize import byte_size, count, duration_ms, duration_s, fit, pad, percent
from batcher.observe.theme import Glyphs, Palette

__all__ = ["bar", "compose", "sparkline", "sweep"]

#: Column widths for the fixed part of the line. Fixed so numbers stay in place as they
#: change — the thing that separates an instrument from a status dump.
LABEL_W, STAGE_W = 18, 14


def bar(fraction: float | None, width: int, palette: Palette, glyphs: Glyphs, frame: int) -> str:
    """The progress bar: a gradient eighth-cell fill, or an honest sweep when unknown.

    An unknown total is the *common* case — Kyber leaves an operator unbudgeted whenever
    the source size is unknown — so the indeterminate form is a first-class design, not a
    fallback. Inventing a denominator would produce a bar that jumps backwards the moment
    the estimate is beaten.

    Args:
        fraction: Completed share in [0, 1], or `None` when the total is unknown.
        width: Bar width in cells.
        palette: The color table.
        glyphs: The character set.
        frame: The repaint counter, which drives the sweep's position.

    Returns:
        The rendered bar, including its end caps.
    """
    if fraction is None:
        return f"{glyphs.cap_l}{sweep(width, palette, glyphs, frame)}{glyphs.cap_r}"
    clamped = 0.0 if fraction < 0 else 1.0 if fraction > 1 else fraction
    exact = clamped * width
    full = int(exact)
    out = []
    # Tint each filled cell by its position, so the fill is one gradient object.
    for i in range(full):
        out.append(f"{palette.ramp(i / max(width - 1, 1))}{glyphs.full}")
    if full < width:
        remainder = exact - full
        if remainder > 0 and glyphs.unicode:
            eighth = glyphs.eighths[int(remainder * 8)]
            out.append(f"{palette.ramp(full / max(width - 1, 1))}{eighth}")
            full += 1
        out.append(f"{palette.muted}{glyphs.empty * (width - full)}")
    return f"{glyphs.cap_l}{''.join(out)}{palette.reset}{glyphs.cap_r}"


def sweep(width: int, palette: Palette, glyphs: Glyphs, frame: int) -> str:
    """An indeterminate comet: a bright head with a fading trail, bouncing end to end.

    Args:
        width: Bar width in cells.
        palette: The color table.
        glyphs: The character set.
        frame: The repaint counter.

    Returns:
        The rendered sweep.
    """
    period = max(width * 2 - 2, 1)
    pos = frame % period
    if pos >= width:
        pos = period - pos
    out = []
    for i in range(width):
        distance = abs(i - pos)
        if distance < len(glyphs.shades):
            tint = palette.ramp(1 - distance / len(glyphs.shades))
            out.append(f"{tint}{glyphs.shades[distance]}")
        else:
            out.append(f"{palette.muted}{glyphs.empty}")
    return "".join(out) + palette.reset


def sparkline(history: Sequence[float], palette: Palette, glyphs: Glyphs) -> str:
    """Recent throughput as a sparkline, or ``""`` until there are enough samples.

    Shows the *shape* of throughput — a stall, a ramp, a stutter — which a single smoothed
    number cannot. Scaled to its own window's peak, so it reads as relative change rather
    than as an absolute the axis-less form could not convey anyway.

    Args:
        history: Recent rate samples, oldest first.
        palette: The color table.
        glyphs: The character set.

    Returns:
        The rendered sparkline, or ``""``.
    """
    if len(history) < 4:
        return ""
    peak = max(history)
    if peak <= 0:
        return ""
    cells = "".join(
        glyphs.spark[min(int(v / peak * len(glyphs.spark)), len(glyphs.spark) - 1)] for v in history
    )
    return f"{palette.muted}{cells}{palette.reset}"


def compose(
    run,
    now: float,
    *,
    palette: Palette,
    glyphs: Glyphs,
    frame: int,
    bar_width: int,
    others: int = 0,
) -> str:
    """The whole status line: spinner, label, stage, bar, counts, rate, elapsed, ETA.

    Args:
        run: The `RunState` being drawn.
        now: The monotonic time of this repaint.
        palette: The color table.
        glyphs: The character set.
        frame: The repaint counter.
        bar_width: Bar width in cells.
        others: How many other queries are in flight but not drawn.

    Returns:
        The composed line, with escape codes.
    """
    p, g = palette, glyphs
    elapsed = now - run.t0
    fraction = run.fraction
    cells = [
        f"{p.accent}{g.spinner[frame % len(g.spinner)]}{p.reset}",
        f"{p.bold}{fit(run.label, LABEL_W)}{p.reset}",
        f"{p.muted}{fit(run.stage, STAGE_W)}{p.reset}",
        bar(fraction, bar_width, p, g, frame),
    ]
    if fraction is not None:
        cells.append(pad(percent(fraction), 4, align="right"))
    if run.partitions_total:
        cells.append(f"{p.dim}{run.partitions_done}/{run.partitions_total}{p.reset} parts")
    cells.append(f"{p.dim}{pad(count(run.rows), 7, align='right')}{p.reset} rows")
    if run.rate > 0:
        cells.append(f"{p.dim}{pad(count(run.rate), 7, align='right')}/s{p.reset}")
        spark = sparkline(run.spark, p, g)
        if spark:
            cells.append(spark)
    if run.bytes:
        cells.append(f"{p.muted}{byte_size(run.bytes)}{p.reset}")
    cells.append(f"{p.muted}{pad(duration_ms(elapsed * 1000), 7, align='right')}{p.reset}")
    eta = run.eta_s
    if eta is not None:
        cells.append(f"{p.muted}ETA {duration_s(eta)}{p.reset}")
    if others:
        # One moving line is an instrument; N interleaved ones are a mess. But silently
        # drawing only the newest made a five-query script look like a one-query script.
        cells.append(f"{p.muted}+{others} more{p.reset}")
    return "  ".join(cells)
