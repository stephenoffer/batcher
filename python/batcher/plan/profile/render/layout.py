"""Columns, sections, and the order they appear in — the profile as a page.

The layout half of the renderer: it turns operators into aligned rows, decides how wide
each column may be, and assembles the header, the table, and the summary sections a reader
works through in order. The module docstring for the package as a whole, and the reasoning
behind this shape, is in `batcher/plan/profile/render/__init__.py`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from batcher._internal.humanize import (
    byte_size,
    count,
    display_width,
    duration_ms,
    fit,
    pad,
    percent,
    plural,
    signed_ratio,
)
from batcher.plan.profile.render.cells import est_cell, notes, op_label, share_bar
from batcher.plan.profile.render.options import (
    ESTIMATE_MISS_FACTOR,
    FOLD_ABOVE_OPS,
    FOLD_BELOW_SHARE,
    HOTSPOTS,
    RenderOptions,
    Styler,
    plain_styler,
    terminal_width,
    unicode_ok,
)
from batcher.plan.profile.render.tree import (
    critical_path,
    has_branch,
    last_child_flags,
    spine,
    subtree_ms,
)
from batcher.plan.profile.render.tree import (
    folded as folded_indices,
)

if TYPE_CHECKING:
    from batcher.plan.profile.types import OpProfile, QueryProfile

__all__ = ["render_profile"]


def _rows(ops: Sequence[OpProfile], opts: RenderOptions) -> tuple[list[list[str]], int]:
    """One list of cells per visible operator, plus the fold markers, in plan order.

    Returns:
        The cell rows and how many operators were folded away, so the caller can print the
        one footnote that explains the elision rather than repeating it on every marker.
    """
    flags = last_child_flags(ops)
    subtree = subtree_ms(ops)
    hidden = folded_indices(ops, opts, subtree)
    # A mark on every row of a straight-line plan is decoration, not information: the hot
    # path *is* the plan. It earns its column only where the tree branches and a reader has
    # to choose which side to look at.
    branching = has_branch(ops)
    critical = critical_path(ops, subtree) if opts.analyze and branching else set()
    # Share is of *measured operator time*, not of `total_ms`. Against `total_ms` every
    # bar on a short query is empty — the operators are 4% of a wall clock dominated by
    # planning and result assembly — so the column that exists to rank operators against
    # each other ranked them all at zero. The operators-vs-everything-else split is the
    # accounting section's job, and it does it in one line instead of six blank bars.
    total = sum(o.elapsed_ms for o in ops if o.measured)
    out: list[list[str]] = []
    folded = 0
    i = 0
    while i < len(ops):
        if i in hidden:
            # Consume the whole *run* of consecutive hidden operators, not one subtree at a
            # time. Four cold single-operator siblings folded individually produce four
            # "… 1 operator folded" lines in place of four operator lines, which is strictly
            # worse than not folding: same height, less information. One line for the run is
            # the only form that actually shortens the tree.
            end = i
            while end < len(ops) and end in hidden:
                end += 1
            if end - i == 1:
                hidden.discard(i)  # nothing to gain; render it normally
                continue
            folded += end - i
            marker = opts.style("muted", f"… {end - i} more")
            label = f"{'  ' if critical else ''}{spine(ops, i, flags, opts.glyphs)}{marker}"
            out.append([label, "", "", "", "", "", ""])
            i = end
            continue
        op = ops[i]
        label = op_label(ops, i, flags, opts)
        if critical:
            # The mark column exists only where there are timings *and* a branch to choose.
            label = f"{opts.glyphs['mark'] if i in critical else ' '} {label}"
        if not opts.analyze or not op.measured:
            out.append([label, est_cell(op), "", "", "", "", notes(op, opts)])
            i += 1
            continue
        share = (op.elapsed_ms / total) if total else 0.0
        out.append(
            [
                label,
                est_cell(op),
                f"actual={op.rows_out:,}",
                signed_ratio(op.rows_out, op.est_rows),
                duration_ms(op.elapsed_ms),
                f"{share_bar(share, 6, opts.unicode)} {pad(percent(share), 4, align='right')}",
                notes(op, opts),
            ]
        )
        i += 1
    return out, folded


#: Column headers, by whether the profile is analyzed. Deliberately never the word
#: ``actual`` in the planned form: there is no such column there, and printing the header
#: for a column that does not exist is how a reader comes to believe a plan was run.
_HEAD_ANALYZE = ("OPERATOR", "ESTIMATE", "ACTUAL", "MISS", "TIME", "OP SHARE", "NOTES")
_HEAD_PLANNED = ("OPERATOR", "ESTIMATE", "", "", "", "", "NOTES")


def _table(rows: list[list[str]], opts: RenderOptions) -> list[str]:
    """Lay `rows` out as aligned columns, giving the operator column whatever is left.

    Numeric widths come from the data so a row count is never truncated — the operator
    column absorbs the slack instead, because a clipped operator name still identifies its
    row while a clipped number is a wrong number.
    """
    head = list(_HEAD_ANALYZE if opts.analyze else _HEAD_PLANNED)
    body = [*rows, head]
    widths = [max((display_width(r[c]) for r in body), default=0) for c in range(7)]
    fixed = sum(widths[1:6]) + 2 * len([w for w in widths[1:6] if w])
    label_w = max(24, opts.width - fixed - widths[6] - 4)
    label_w = min(label_w, max((display_width(r[0]) for r in body), default=24))
    label_w = max(label_w, min(48, max((display_width(r[0]) for r in body), default=24)))

    def line(cells: Sequence[str], *, header: bool = False) -> str:
        parts = [fit(cells[0], label_w)]
        for c in range(1, 6):
            if widths[c]:
                parts.append(pad(cells[c], widths[c], align="right"))
        parts.append(cells[6])
        text = "  ".join(p for p in parts if p is not None).rstrip()
        return opts.style("head", text) if header else text

    return [line(head, header=True), *(line(r) for r in rows)]


# --- summary sections -------------------------------------------------------


def _accounting(profile: QueryProfile, opts: RenderOptions) -> list[str]:
    """Where the wall clock went, operators against everything else.

    The line this section exists for is the second one. `total_ms` is the whole terminal
    operation and the per-operator times cover only the engine call inside it, so a query
    whose operators account for 1% of its wall clock was reporting a bottleneck at "1% of
    wall time" and leaving the remaining 99% unnamed. On a small query that remainder is
    usually the dominant cost and it is the number worth acting on: it is planning,
    optimization, admission, the FFI crossing, and assembling the Arrow result.

    **`OpProfile.elapsed_ms` is not wall time and must not be divided by `total_ms`.** It is
    the operator's own transform time *summed over every morsel and every worker that ran
    it* (`bc_interp::stream::meter`), which is deliberate: operators interleave in a
    pipelined model, so no wall-clock interval belongs to one alone. The consequence is that
    a parallel operator's figure exceeds the whole query's wall clock, and this block used
    to divide the two anyway. A three-operator `filter` over 20M rows on 64 workers printed:

        filter [a > 500]  ...  112ms  #####.  96%
        where the time went
          operators       116ms   315%  of 37ms

    -- a filter reporting 112 ms inside a 37 ms query, and a share of 315%. Worse silently:
    `rest` was `max(0.0, total - ops_ms)`, so the "elsewhere" line this section exists for
    was clamped to zero and vanished *exactly* when the operators were parallel, which is
    when the planning/FFI remainder is most worth naming.

    So each operator is converted to the wall time its work must have occupied -- `T` cpu-ms
    spread over `N` workers occupies `T / N` -- and that is what is reported against the
    clock. It is an estimate in one direction only: pipelined operators overlap, so summing
    per-operator occupancy can over-count the wall time actually spent inside operators. It
    cannot under-count. When the sum exceeds the clock the split is not attributable at all,
    and the block says so rather than printing a number: an unattributable split is a fact
    about the measurement, and clamping it to zero reported the opposite.

    Sequential operators are unaffected -- `threads <= 1` makes the conversion the identity,
    so a single-threaded profile renders exactly as before.
    """
    measured = [o for o in profile.ops if o.measured]
    ops_cpu = sum(o.elapsed_ms for o in measured)
    ops_wall = sum(o.elapsed_ms / max(o.threads, 1) for o in measured)
    total = profile.total_ms
    if total <= 0:
        return []
    style = opts.style
    lines = [style("head", "where the time went")]
    cpu_note = f"  ({duration_ms(ops_cpu)} cpu across workers)" if ops_cpu > ops_wall * 1.05 else ""
    if ops_wall > total:
        # Overlapping operators, or a `threads` the engine did not record. Either way the
        # wall clock cannot be split between operators and everything else.
        lines.append(
            f"  operators   {pad(duration_ms(ops_wall), 9, align='right')}"
            f"  {pad('n/a', 5, align='right')}  occupancy exceeds the {duration_ms(total)}"
            f" clock — overlapping operators, not attributable{cpu_note}"
        )
        return lines
    lines.append(
        f"  operators   {pad(duration_ms(ops_wall), 9, align='right')}"
        f"  {pad(percent(ops_wall / total), 5, align='right')}  of {duration_ms(total)}{cpu_note}"
    )
    rest = total - ops_wall
    if rest > 0:
        note = "planning, optimization, admission, FFI crossing, result assembly"
        role = "warn" if rest / total > 0.5 else "muted"
        lines.append(
            f"  elsewhere   {pad(duration_ms(rest), 9, align='right')}"
            f"  {pad(percent(rest / total), 5, align='right')}  {style(role, note)}"
        )
    return lines


def _hotspots(profile: QueryProfile, opts: RenderOptions) -> list[str]:
    """The operators that own the run, named before the tree a reader has to walk.

    On a plan small enough to take in at once this adds nothing, so it is emitted only
    past `FOLD_ABOVE_OPS` operators — the same point at which the tree starts folding.
    """
    ops = [o for o in profile.ops if o.measured and o.elapsed_ms > 0]
    if len(profile.ops) <= FOLD_ABOVE_OPS or not ops:
        return []
    total = sum(o.elapsed_ms for o in ops)
    top = sorted(ops, key=lambda o: o.elapsed_ms, reverse=True)[:HOTSPOTS]
    lines = [opts.style("head", f"hot operators (top {len(top)} of {len(ops)} measured)")]
    for op in top:
        share = op.elapsed_ms / total if total else 0.0
        lines.append(
            f"  {pad(f'{op.kind} (op {op.op_id})', 28)}"
            f"{pad(duration_ms(op.elapsed_ms), 9, align='right')}  "
            f"{share_bar(share, 6, opts.unicode)} {pad(percent(share), 4, align='right')}"
            f"  {op.rows_out:,} rows"
        )
    return lines


def _insights(profile: QueryProfile, opts: RenderOptions) -> list[str]:
    """The short list of things about this run that a person should act on.

    Everything here is derived from measurements already on the profile — nothing new is
    computed and nothing is guessed. It exists because the per-operator table shows *that*
    an operator spilled or missed its estimate, and a reader still has to know that either
    is worth doing something about.
    """
    style = opts.style
    found: list[tuple[str, str]] = []
    for op in profile.ops:
        if not op.measured:
            continue
        error = op.est_error
        missed = error >= ESTIMATE_MISS_FACTOR or error <= 1 / ESTIMATE_MISS_FACTOR
        if not math.isnan(error) and missed:
            found.append(
                (
                    "warn",
                    f"{op.kind} (op {op.op_id}) row estimate was "
                    f"{signed_ratio(op.rows_out, op.est_rows)} — the plan above it was chosen "
                    f"for {count(op.est_rows)} rows and ran on {count(op.rows_out)}",
                )
            )
        if op.spilled:
            found.append(
                (
                    "warn",
                    f"{op.kind} (op {op.op_id}) spilled {byte_size(op.spill_bytes)} to disk — "
                    "raise the memory envelope or add partitions to keep it in memory",
                )
            )
        if op.paging:
            found.append(
                (
                    "critical",
                    f"{op.kind} (op {op.op_id}) took {op.major_faults:,} disk-backed page "
                    "faults — the machine was paging against this query",
                )
            )
    if not found:
        return []
    seen: set[str] = set()
    lines = [opts.style("head", "what to look at")]
    for role, text in found:
        if text in seen:
            continue
        seen.add(text)
        lines.append(f"  {style(role, '!')} {text}")
    return lines


def _header(profile: QueryProfile, opts: RenderOptions, width: int) -> list[str]:
    """The title line and its rule: what this is, how big, and what it cost.

    `width` is the width of the table that follows, not the terminal's. A rule drawn to the
    terminal instead makes the same plan render differently in every window it is read in —
    which is a problem for a string that gets pasted into issues and asserted on in tests,
    and it looks wrong besides: a rule should underline its table, not the screen.
    """
    style = opts.style
    kind = "measured" if opts.analyze and profile.measured else "planned"
    facts = [plural(len(profile.ops), "operator")]
    if opts.analyze and profile.measured:
        facts.append(plural(profile.rows, "row"))
        facts.append(duration_ms(profile.total_ms))
    if profile.distributed:
        facts.append("distributed")
    title = style("head", f"query plan ({kind})")
    detail = style("muted", "  ·  ".join(facts))
    gap = max(4, width - display_width(title) - display_width(detail))
    rule = display_width(title) + gap + display_width(detail)
    return [f"{title}{' ' * gap}{detail}", style("muted", opts.glyphs["rule"] * rule)]


def _footer(profile: QueryProfile, opts: RenderOptions) -> list[str]:
    """Everything after the operator table, in the order a reader needs it.

    Accounting first (is the profile even describing the time you waited?), then the hot
    operators, then the diagnosis, then the actionable list, then the subsystem decisions.
    """
    style = opts.style
    out: list[str] = []
    if opts.analyze and profile.measured:
        for section in (_accounting, _hotspots):
            block = section(profile, opts)
            if block:
                out.extend(["", *block])
        out.append("")
        out.append(f"total: {profile.total_ms:.2f} ms, {plural(profile.rows, 'row')} out")
        out.append(profile.bottleneck_summary())
        util = profile.utilization_summary()
        if util:
            out.append(util)
        # Only on a single-node run. On a distributed one the profile is assembled on the
        # driver while the work happened on the workers, so naming the driver's machine
        # here would attribute every timing above it to hardware that ran none of it.
        if not profile.distributed:
            out.append(f"machine: {profile.machine}")
        insights = _insights(profile, opts)
        if insights:
            out.extend(["", *insights])
    if profile.decisions:
        out.extend(["", style("head", "decisions:")])
        out.extend(
            f"  - {style('muted', f'[{d.subsystem}/{d.category}]')} {d.summary}"
            for d in profile.decisions
        )
    if opts.analyze and profile.measured and profile.worker_ops:
        out.extend(["", style("head", "distributed map sub-plan (summed across workers):")])
        worker_rows, _ = _rows(profile.worker_ops, opts)
        out.extend("  " + line for line in _table(worker_rows, opts))
    if opts.analyze and profile.measured and profile.adaptive_stages:
        out.extend(["", style("head", "adaptive re-optimization:")])
        for stage in profile.adaptive_stages:
            out.append(
                f"  - {stage.get('kind', '?')} (op {stage.get('op_id', '?')}): "
                f"est≈{stage.get('est_rows', 0):,.0f} actual={stage.get('actual_rows', 0):,} "
                f"→ {stage.get('action', '')}"
            )
    return out


class QueryProfileView:
    """A `QueryProfile`-shaped stand-in for a sub-plan that shares the parent's totals.

    The distributed worker sub-plan is a separate operator-id space with its own tree, but
    its shares are only meaningful against the query's wall clock. Rather than fabricate a
    second `QueryProfile` (which would then claim its own bottleneck, machine, and row
    count), this exposes only what `_rows` reads.
    """

    __slots__ = ("ops", "total_ms")

    def __init__(self, ops: tuple[OpProfile, ...], parent: QueryProfile) -> None:
        self.ops = ops
        self.total_ms = parent.total_ms


def render_profile(
    profile: QueryProfile,
    *,
    analyze: bool | None = None,
    style: Styler | None = None,
    width: int | None = None,
    unicode: bool | None = None,
    fold: bool = True,
) -> str:
    """Render `profile` as the operator tree plus its summary sections.

    Args:
        profile: The profile to render.
        analyze: Show measured columns; defaults to whether the profile carries them.
        style: A `Styler` for color; defaults to `plain_styler`, which adds nothing.
        width: Line width; defaults to the terminal's, clamped to a legible range.
        unicode: Draw with box-drawing glyphs; defaults to whether stdout can encode them.
        fold: Fold cold subtrees on a large analyzed plan.

    Returns:
        The rendered profile.

    Examples:
        .. doctest::

            >>> from batcher.plan.profile import OpProfile, QueryProfile
            >>> op = OpProfile(op_id=0, kind="scan", depth=0, est_rows=10.0)
            >>> "scan" in QueryProfile(ops=(op,)).render()
            True
    """
    opts = RenderOptions(
        analyze=profile.measured if analyze is None else analyze,
        width=width if width is not None else terminal_width(),
        unicode=unicode_ok() if unicode is None else unicode,
        style=style or plain_styler,
        fold=fold,
    )
    if not profile.ops:
        head = _header(profile, opts, 40)
        return "\n".join([*head, "  (no operators)", *_footer(profile, opts)])
    rows, folded = _rows(profile.ops, opts)
    body = _table(rows, opts)
    if folded:
        # One footnote rather than an explanation on every marker: the markers are inside a
        # width-constrained column and the reader needs the rule once, with the way to see
        # what was hidden.
        body.append(
            opts.style(
                "muted",
                f"  {plural(folded, 'operator')} folded: subtrees under "
                f"{percent(FOLD_BELOW_SHARE)} of operator time. "
                'explain(format="json") lists every one.',
            )
        )
    table_width = max(display_width(line) for line in body)
    return "\n".join([*_header(profile, opts, table_width), *body, *_footer(profile, opts)])
