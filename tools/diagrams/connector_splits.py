#!/usr/bin/env python3
"""Draw `connector_splits.svg` -- how a registered custom format fans out into parallel read tasks.

Source of truth: `docs/user-guide/moving-data/custom-connectors.md` ("Splits are the unit of
read parallelism", "The write side", "The registries") and `python/batcher/io/`: a source's
`splits()` returns `FileSplit`s that carry `(format_name, path, kwargs)` rather than data, a
worker rebuilds the reader as `SOURCES.get(format_name)(path, **kwargs)`, and a sink's
per-shard `WrittenFile`s are concatenated into one `WriteManifest` that `commit` finalizes once.
Registration is the `@SOURCES.register` / `@SINKS.register` decorator, run on import.

Layout: the read path as the main lane, left to right, ending in one task per split, and
the write path as a thinner lane under it, because the same name resolves both.
"""

from __future__ import annotations

from _authoring import arrow, band, card, code, label, note, svg, tint, write

W, H = 1000, 512
YC = 220  # centre line of the read lane
SPLIT_Y = (148, 198, 248)

body: list[str] = [
    band(20, 20, 960, 330, "READ: ONE TASK PER SPLIT", "blue"),
    code(270, 52, ['@SOURCES.register("psv")', "class PSVSource(FileSource)"], 250),
    arrow(395, 115, 395, YC - 34),
    label(407, 146, "registers", size=11.5),
    label(407, 161, "on import", size=11.5),
    card(44, YC - 28, 190, 56, "bt.read(path)", 'format="psv"'),
    arrow(238, YC, 296, YC),
    label(267, YC - 12, "look up", anchor="middle", size=11.5),
    card(300, YC - 28, 190, 56, "SOURCES", '"psv" is PSVSource'),
    arrow(494, YC, 566, 218),
    label(530, YC - 12, "splits()", anchor="middle", size=11.5),
]
for y in SPLIT_Y:
    mid = y + 20
    body += [
        tint(572, y, 170, 40, "FileSplit"),
        arrow(746, mid, 796, mid),
        label(771, mid - 7, "task", anchor="middle", size=11),
        card(802, y, 156, 40, "worker"),
    ]
body += [
    note(657, 312, "format name, path, kwargs:", anchor="middle"),
    note(657, 330, "locators, never data", anchor="middle"),
    note(880, 312, "rebuilds PSVSource and", anchor="middle"),
    note(880, 330, "reads storage directly", anchor="middle"),
    note(139, 312, "a directory of files gives", anchor="middle"),
    note(139, 330, "one FileSplit per file", anchor="middle"),
    # ---- The write lane ----------------------------------------------------------------
    band(20, 370, 960, 122, "WRITE: THE SAME NAME RESOLVES A SINK", "amber"),
    card(44, 414, 190, 48, "ds.write(path)", 'format="psv"'),
    arrow(238, 438, 296, 438, "amber"),
    label(267, 426, "look up", anchor="middle", size=11.5),
    card(300, 414, 190, 48, "SINKS", '"psv" is PSVSink'),
    arrow(494, 438, 552, 438, "amber"),
    label(523, 426, "per shard", anchor="middle", size=11.5),
    card(556, 414, 190, 48, "WrittenFile lists", "path, rows, bytes"),
    arrow(750, 438, 798, 438, "amber"),
    label(774, 426, "merge", anchor="middle", size=11.5),
    card(802, 414, 156, 48, "commit once", "one WriteManifest"),
]

write("connector_splits", svg(W, H, "".join(body)))
print("wrote connector_splits.svg")
