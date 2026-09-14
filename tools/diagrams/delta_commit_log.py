#!/usr/bin/env python3
"""Draw `delta_commit_log.svg` -- a Delta table as a log of file sets, and what vacuum ends.

Source of truth: `python/batcher/io/formats/lakehouse/delta/_commit.py`
(``commit_add_actions`` registers the files the workers already wrote and moves none of
them, so a commit is O(files) and the driver writes only the log) and
`lakehouse/delta/maintenance.py`, whose module docstring is explicit about the three
operations: compact and z_order commit ``remove`` actions that retire files *from the log*
while leaving them on storage, "so every existing version still reads and time travel
survives", and vacuum "is the only operation that deletes", removing files no live version
references once they are older than the retention window -- "the files it removes are
exactly the ones time travel and any in-flight reader depend on". Time travel itself is
`delta/source.py`, which takes ``version=`` or ``timestamp=``.

The figure is a grid rather than a flow because the claim is about *set membership over
time*: version 3 does not contain the files version 1 did, and both file sets are on
storage at once. Drawn this way, time travel is a row of the grid and vacuum is the
deletion of a column, which is the whole argument in one picture. A prose sentence has to
assert the relationship; the grid lets the reader check it.

Deliberately not drawn: log checkpointing, deletion vectors, and concurrent-writer
conflict resolution. They are real and they are not what makes time travel work.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, BLUE_MID, FONT, GREY, band, label, note, svg, write

W, H = 980, 620

FILES = ("f1", "f2", "f3", "f4", "f5")
COL_X = (240, 356, 472, 588, 704)
COL_W = 100
ROW_Y = (166, 222, 278, 334)
ROW_H = 48

#: (version, the operation that produced it, the actions it recorded, its live file set).
VERSIONS = (
    ("v0", "create", "add f1, f2", {"f1", "f2"}),
    ("v1", "append", "add f3", {"f1", "f2", "f3"}),
    ("v2", "merge", "add f4 · remove f2", {"f1", "f3", "f4"}),
    ("v3", "compact", "add f5 · remove f1, f3", {"f4", "f5"}),
)

body: list[str] = [
    note(490, 54, "Every commit records which files the table now holds. It adds files and retires references; it never rewrites an earlier version.", anchor="middle"),
    band(20, 90, 940, 292, "ONE TABLE, FOUR COMMITS, FIVE DATA FILES", "blue"),
]

for name, x in zip(FILES, COL_X):
    body.append(
        f'<text x="{x + COL_W / 2}" y="146" text-anchor="middle" font-family="{FONT}" '
        f'font-size="12.5" font-weight="700" class="t-arrow">{name}</text>'
    )
body.append(note(824, 146, "data files on storage", anchor="start"))

for (version, op, actions, live), y in zip(VERSIONS, ROW_Y):
    body += [
        label(40, y + 22, version),
        note(76, y + 22, op),
        note(40, y + 39, actions),
    ]
    for name, x in zip(FILES, COL_X):
        if name in live:
            body.append(
                f'<rect x="{x}" y="{y}" width="{COL_W}" height="{ROW_H}" rx="7" '
                f'fill="{BLUE_MID}" fill-opacity="0.16" stroke="{BLUE_MID}" stroke-width="1.6"/>'
                f'<text x="{x + COL_W / 2}" y="{y + 30}" text-anchor="middle" '
                f'font-family="{FONT}" font-size="12" font-weight="600" class="t-title">live</text>'
            )
        else:
            body.append(
                f'<rect x="{x}" y="{y}" width="{COL_W}" height="{ROW_H}" rx="7" fill="none" '
                f'stroke="{GREY}" stroke-width="1.2" stroke-dasharray="4 4"/>'
                f'<text x="{x + COL_W / 2}" y="{y + 30}" text-anchor="middle" '
                f'font-family="{FONT}" font-size="11.5" class="t-sub">not in {version}</text>'
            )

body += [
    note(824, ROW_Y[1] + 22, "read version=1", anchor="start"),
    note(824, ROW_Y[1] + 38, "reads this row", anchor="start"),
    note(824, ROW_Y[3] + 22, "read latest", anchor="start"),
    note(824, ROW_Y[3] + 38, "reads this row", anchor="start"),
]

# The columns vacuum would reclaim: every file the bottom row does not mark.
for name, x in zip(FILES, COL_X):
    if name in VERSIONS[-1][3]:
        continue
    body += [
        f'<path d="M {x + COL_W / 2} 392 V 424" stroke="{AMBER_DEEP}" stroke-width="2.2" '
        f'stroke-dasharray="5 4" marker-end="url(#arA)"/>',
        label(x + COL_W / 2, 444, f"delete {name}", anchor="middle"),
    ]

body += [
    band(20, 462, 940, 140, "VACUUM IS THE ONLY OPERATION THAT DELETES", "amber"),
    note(40, 504, "Those three files are referenced by no live version, so vacuum may reclaim them. Until it does, every one of them is still on storage, which is the only", anchor="start"),
    note(40, 522, "reason reading version 1 works at all. Time travel is not a backup taken on the side: it is what a log that only ever adds leaves behind.", anchor="start"),
    note(40, 550, "Compact produced v3 by bin-packing f1 and f3 into f5. Its remove actions retire the two from the log and leave them on storage, so v1 and v2 still read.", anchor="start"),
    note(40, 572, "Vacuum is the step that makes that irreversible, so it defaults to a dry run and keeps an unreferenced file for a retention window first.", anchor="start"),
]

write("delta_commit_log", svg(W, H, "".join(body)))
print("wrote delta_commit_log.svg")
