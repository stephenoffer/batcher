#!/usr/bin/env python3
"""Draw `output_modes.svg` -- what append, complete and update each emit for one input.

Source of truth: `python/batcher/core/streaming_query/processors.py`. `AggregateProcessor`
folds every micro-batch into a running aggregate and emits `finalize()` whole under
`complete`; under `update` it anti-joins the result against the one it last emitted over
*every column*, so a group whose value did not move is not re-sent and a trigger that
moved nothing emits no rows at all. `StatelessProcessor` runs the per-batch plan and is
the only `append` shape a breaker-free pipeline has. `make_processor` is where a
combination is refused, at `start()` rather than mid-stream: `append` over an aggregate
raises unless that aggregate carries a watermark and a windowed group key, and
`complete`/`update` raise over a stateless pipeline. `api/io_namespace/writer.py` adds the
sink-side restriction -- a path or Delta sink takes `append` only.

The figure exists because the three modes are only distinguishable by *what comes out of
the same input*, and that is a grid of nine cells. Prose can define them one at a time; it
cannot put the third trigger's three answers side by side, which is where they differ most.

The aggregate is a `max` rather than a `count` on purpose: it lets the third batch arrive,
be read, and still move no group's value, which is the case that separates `complete` from
`update` without depending on whether an idle trigger reaches the processor at all.

Deliberately not drawn: `append` over a windowed aggregate closing on a watermark, which
is a fourth shape and belongs beside the watermark figure.
"""

from __future__ import annotations

from _authoring import BLUE_MID, FONT, GREY, band, label, note, svg, write

W, H = 980, 604

COLS = ((206, 236), (452, 236), (698, 236))
ROWS = (160, 254, 348)
RH = 78

HEADERS = (
    ("trigger 1", "arrives: (a,5) (a,3) (b,2)"),
    ("trigger 2", "arrives: (b,7) (c,1)"),
    ("trigger 3", "arrives: (a,1) (b,4)"),
)


def cell(col: int, row: int, lines: tuple[str, ...], live: bool = True) -> str:
    """One emitted result, drawn as the rows the sink receives on that trigger."""
    x, w = COLS[col]
    y = ROWS[row]
    color = BLUE_MID if live else GREY
    out = (
        f'<rect x="{x}" y="{y}" width="{w}" height="{RH}" rx="8" fill="{color}" '
        f'fill-opacity="0.14" stroke="{color}" stroke-width="1.6"/>'
    )
    top = y + RH / 2 - 8 * (len(lines) - 1) + 5
    for i, text in enumerate(lines):
        out += (
            f'<text x="{x + w / 2}" y="{top + 17 * i}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="12.5" font-weight="600" class="t-title">{text}</text>'
        )
    return out


body: list[str] = [
    note(
        490,
        50,
        "Three micro-batches of (key, value) pairs, into group_by(k).agg(m = col('v').max()). Each cell is what the sink receives on that trigger.",
        anchor="middle",
    ),
    band(20, 96, 940, 366, "THE SAME INPUT, THREE OUTPUT MODES", "blue"),
]

for i, (title, sub) in enumerate(HEADERS):
    x, w = COLS[i]
    body += [
        label(x + w / 2, 128, title, anchor="middle"),
        note(x + w / 2, 145, sub, anchor="middle"),
    ]

# APPEND. Not legal over this aggregate, so the row is drawn for the pipeline it does fit.
body += [
    label(38, ROWS[0] + 30, "APPEND"),
    note(38, ROWS[0] + 48, "a stateless pipeline"),
    note(38, ROWS[0] + 64, "over the same input;"),
    note(38, ROWS[0] + 80, "refused over this agg"),
    cell(0, 0, ("(a,5) (a,3) (b,2)",)),
    cell(1, 0, ("(b,7) (c,1)",)),
    cell(2, 0, ("(a,1) (b,4)",)),
]

# COMPLETE. The whole running result, every trigger, unconditionally.
body += [
    label(38, ROWS[1] + 30, "COMPLETE"),
    note(38, ROWS[1] + 48, "the whole running"),
    note(38, ROWS[1] + 64, "result, every trigger"),
    cell(0, 1, ("a = 5", "b = 2")),
    cell(1, 1, ("a = 5", "b = 7", "c = 1")),
    cell(2, 1, ("a = 5", "b = 7", "c = 1")),
]

# UPDATE. The rows that differ from the last emission, found by an anti-join.
body += [
    label(38, ROWS[2] + 30, "UPDATE"),
    note(38, ROWS[2] + 48, "only the rows that"),
    note(38, ROWS[2] + 64, "changed since the"),
    note(38, ROWS[2] + 80, "last trigger"),
    cell(0, 2, ("a = 5", "b = 2")),
    cell(1, 2, ("b = 7", "c = 1")),
    cell(2, 2, ("nothing",), live=False),
]

body += [
    band(20, 482, 940, 108, "READING THE THIRD COLUMN", "grey"),
    note(
        490,
        524,
        "Nothing in the third batch raises any group's maximum, so the running result is the one already emitted. Complete re-sends all three rows anyway;",
        anchor="middle",
    ),
    note(
        490,
        542,
        "update anti-joins against what it last emitted and sends none. On a wide key space that difference is the whole reason to pick update.",
        anchor="middle",
    ),
    note(
        490,
        566,
        "The mode is checked at start(). An aggregate refuses append without a watermark and a windowed group key; a stateless pipeline refuses complete and update.",
        anchor="middle",
    ),
]

write("output_modes", svg(W, H, "".join(body)))
print("wrote output_modes.svg")
