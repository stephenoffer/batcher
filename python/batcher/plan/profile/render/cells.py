"""One operator's worth of rendered text: its estimate, its share bar, its notes.

Everything here is a pure function of a single `OpProfile` (plus the options), which is
what lets the layout module stay about columns and sections rather than about what any
given number means.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from batcher._internal.humanize import byte_size
from batcher.plan.profile.render.options import BLOCKS, RenderOptions
from batcher.plan.profile.render.tree import spine

if TYPE_CHECKING:
    from batcher.plan.profile.types import OpProfile

__all__ = ["est_cell", "hardware_flags", "notes", "op_label", "share_bar"]


def est_cell(op: OpProfile) -> str:
    """The planned row estimate, as ``est≈1,234`` — or ``est≈?`` when unbudgeted.

    Rendered with full digits and thousands separators rather than an SI abbreviation:
    this is the number a reader compares against `actual`, and ``est≈1.2M`` against
    ``actual=1,203,441`` is a comparison nobody can make. Tooling parses it too.
    """
    return "est≈?" if math.isnan(op.est_rows) else f"est≈{op.est_rows:,.0f}"


def share_bar(share: float, cells: int, unicode_ok: bool) -> str:
    """A `cells`-wide bar for a share in [0, 1], eighth-resolved when Unicode is available.

    The bar is the reason the table can be *scanned*: a column of percentages must be read
    value by value, while a column of bars shows the distribution at a glance and makes the
    one operator that owns the run impossible to miss.
    """
    clamped = 0.0 if share < 0 else 1.0 if share > 1 else share
    if not unicode_ok:
        full = round(clamped * cells)
        return "#" * full + "." * (cells - full)
    exact = clamped * cells
    full = int(exact)
    out = "█" * min(full, cells)
    if full < cells:
        remainder = exact - full
        eighths = int(remainder * 8)
        out += BLOCKS[eighths] if eighths > 0 else "░"
        out += "░" * (cells - full - 1)
    return out


def hardware_flags(o: OpProfile) -> str:
    """The hardware conditions worth flagging on an operator's plan line, or `""`.

    Only conditions that change what a reader should *do* appear here, and only when they are
    actually present. A plan line is already dense, and a row of always-on counters would push
    the fields people read every time off the right edge to make room for numbers that are
    usually zero. Disk reads are the exception to "only when abnormal": knowing a scan reached
    the device rather than the page cache is the difference between a timing worth trusting
    and one that measured a warm cache.
    """
    parts = []
    if o.paging:
        # First, because it invalidates the reading of everything else on the line: an operator
        # taking disk-backed faults is waiting on storage for its own memory, and its time and
        # utilization describe that rather than its work.
        parts.append(f"PAGING({o.major_faults:,} major faults)")
    if o.contended:
        parts.append(f"contended({o.preemption_rate:,.0f} preempt/core-s)")
    if o.io_read_bytes:
        parts.append(f"disk-read={byte_size(o.io_read_bytes)}")
    if o.io_write_bytes:
        parts.append(f"disk-write={byte_size(o.io_write_bytes)}")
    return " ".join(parts)


def notes(op: OpProfile, opts: RenderOptions) -> str:
    """The trailing free-text column: strategy, backend, spill, pushdown, pressure.

    Last on the line and never truncated, because every clause in it is conditional — it
    is present exactly when it has something to say, so cutting it would cut the signal.
    """
    style = opts.style
    parts: list[str] = []
    if op.algorithm:
        parts.append(style("accent", op.algorithm))
    if opts.analyze and op.backend:
        parts.append(op.backend)
    if not opts.analyze and op.provenance:
        parts.append(style("muted", f"({op.provenance})"))
    if op.spilled:
        volume = byte_size(op.spill_bytes)
        parts.append(style("warn", f"spill {volume}" if op.spill_bytes else "spill"))
    if opts.analyze and op.peak_rss_bytes:
        parts.append(style("muted", f"rss+{byte_size(op.peak_rss_bytes)}"))
    if opts.analyze:
        hardware = hardware_flags(op)
        if hardware:
            parts.append(style("critical" if op.paging else "warn", hardware))
    if op.pushed:
        parts.append(style("good", f"pushed[{op.pushed}]"))
    return "  ".join(parts)


def op_label(ops: Sequence[OpProfile], i: int, flags: Sequence[bool], opts: RenderOptions) -> str:
    """The first column: the tree spine, the operator's kind, and what it does.

    The detail is bracketed and muted so the kinds still read as a column down the left
    edge while the thing that distinguishes one `hash_join` from another is right there
    beside it, rather than in a JSON document nobody opens.
    """
    head = spine(ops, i, flags, opts.glyphs) + ops[i].kind
    detail = ops[i].detail
    return f"{head}  {opts.style('muted', f'[{detail}]')}" if detail else head
