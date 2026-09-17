#!/usr/bin/env python3
"""Draw `kafka_offsets.svg` -- partitions to readers, and when an offset is recorded.

Source of truth: `python/batcher/io/formats/streaming/kafka.py` (module docstring,
`_poll`, `_commit_delivered`, `_on_assign`, `_apply_seek`) and the page
`docs/integrations/streams/kafka.md`.

`splits()` returns one split per topic-partition, each rebuilding a consumer with an explicit
assignment for that one partition. Per micro-batch the order is: poll without committing, write
ahead the consumed position to Batcher's checkpoint, publish, then commit the consumer group
synchronously. A crash between publish and commit re-delivers the batch, so the failure mode is
a duplicate and never a gap. On restart each partition resumes at the checkpointed offset + 1;
a partition with no checkpointed position keeps the group's committed offset, or the
`starting_offsets` position for a new group.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, BLUE, arrow, band, card, label, note, step, svg, tint, write

W, H = 980, 560

ROWS = (88, 188, 288)

body: list[str] = [
    band(20, 20, 330, 360, "ONE SPLIT PER PARTITION", "blue"),
    band(370, 20, 590, 360, "EACH MICRO-BATCH, IN ORDER", "amber"),
]

for i, y in enumerate(ROWS):
    body += [
        card(40, y, 110, 64, f"P{i}", "partition"),
        tint(212, y, 118, 64, "Reader", "own consumer"),
        arrow(154, y + 32, 206, y + 32),
        label(180, y + 22, "assign", "middle", 11.5),
    ]
body.append(note(185, 368, "Read parallelism = partition count.", anchor="middle"))

# The four steps, top to bottom, each with the system it touches on the right.
SX, SW, SH = 446, 244, 58
STEPS = (
    (58, "Poll", "no commit at poll time"),
    (138, "Write ahead", "the consumed position"),
    (218, "Publish", "the micro-batch"),
    (298, "Commit group offset", "synchronous, after publish"),
)
for n, (y, title, sub) in enumerate(STEPS, start=1):
    body += [tint(SX, y, SW, SH, title, sub, kind="amber"), step(SX + 2, y + 2, n, "amber")]

body += [
    # The three readers join into one micro-batch stream.
    f'<path d="M 334 {ROWS[0] + 32} L 360 {ROWS[0] + 32} M 334 {ROWS[1] + 32} L 360 {ROWS[1] + 32} '
    f'M 334 {ROWS[2] + 32} L 360 {ROWS[2] + 32} M 360 {ROWS[2] + 32} L 360 87 L 438 87" '
    f'fill="none" stroke="{BLUE}" stroke-width="2.4" marker-end="url(#arB)"/>',
    label(404, 78, "messages", "middle", 11.5),
    card(780, 138, 150, SH, "Checkpoint", "source of truth"),
    arrow(SX + SW + 6, 167, 774, 167),
    label(733, 157, "offsets", "middle", 11.5),
    card(780, 218, 150, SH, "Sink", "idempotent"),
    arrow(SX + SW + 6, 247, 774, 247),
    label(733, 237, "rows", "middle", 11.5),
    card(780, 298, 150, SH, "Consumer group", "the fallback"),
    arrow(SX + SW + 6, 327, 774, 327),
    label(733, 317, "commit", "middle", 11.5),
    # Restart.
    band(20, 440, 940, 100, "ON RESTART", "grey"),
    f'<path d="M 934 167 L 946 167 L 946 432" fill="none" stroke="{AMBER_DEEP}" '
    'stroke-width="2.4" marker-end="url(#arA)"/>',
    label(934, 412, "restart", anchor="end", size=12),
    note(
        44,
        490,
        "Each partition resumes at its checkpointed offset + 1. Without a checkpointed position it",
    ),
    note(
        44,
        510,
        "keeps the group's committed offset, or starting_offsets for a new group. A crash between",
    ),
    note(44, 530, "publish and commit replays a batch: a duplicate, never a gap."),
]

write("kafka_offsets", svg(W, H, "".join(body)))
print("wrote kafka_offsets.svg")
