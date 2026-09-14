#!/usr/bin/env python3
"""Draw `merge_into_branches.svg` -- MERGE INTO's three populations, landing in one commit.

Source of truth: `python/batcher/api/merge/builder.py` (``when_matched`` /
``when_not_matched`` / ``when_not_matched_by_source``, each returning only the actions
legal for its population, and applied in the order they are added with first match
winning), `api/merge/native.py` (``NATIVE_MERGE_SINKS`` is Delta and Iceberg), and
`api/merge/delta_native.py`, which states that on a Delta target delta-rs "rewrites only
the data files the join touches and commits them as a single version".

The branching is the half a table can state. The half it cannot is the *timeline*
underneath: a merge is one commit, and the delete-then-append people reach for instead is
two, with an interval in between during which a reader sees the deleted rows gone and the
replacement not yet there. Two commits on a time axis show that interval; a sentence about
atomicity does not.

Two things here are deliberately precise rather than generous. The one-commit property
belongs to a transactional target: on a plain directory the merge writes its new files and
then deletes the ones it replaced, so a crash between the two leaves *both* copies of a
key, which is why that path is documented single-writer. And a
``when_not_matched_by_source`` clause forces a full rewrite -- every target row is a
candidate, so no file can be skipped -- which is inherent to the clause rather than a
limitation here.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, BLUE, FONT, GREY, arrow, band, card, label, note, svg, write

W, H = 980, 684

BR_Y, BR_H = 148, 84
BRANCHES = (
    (50, 280, "NOT MATCHED BY SOURCE", "target key 1, absent from the change set", "update · update_all · delete", "1 target row"),
    (350, 250, "MATCHED", "keys 2 and 3, present in both", "update · update_all · delete", "2 matched rows"),
    (630, 280, "NOT MATCHED", "source key 4, new to the target", "insert · insert_all", "1 new row"),
)

COMMIT_X, COMMIT_W, COMMIT_Y, COMMIT_H = 320, 340, 366, 76

body: list[str] = [
    note(490, 50, "One statement over a target holding keys 1, 2, 3 and a change set holding keys 2, 3, 4. Each key falls into exactly one population.", anchor="middle"),
    band(20, 78, 940, 222, "THREE POPULATIONS, THREE CLAUSES", "blue"),
]

for x, w, title, sub, actions, contributes in BRANCHES:
    body += [
        card(x, BR_Y, w, BR_H, title, sub),
        note(x + w / 2, BR_Y + BR_H + 20, actions, anchor="middle"),
        label(x + w / 2, BR_Y + BR_H + 42, contributes, anchor="middle"),
    ]

body += [
    band(20, 320, 940, 136, "ALL OF IT IN ONE VERSION", "amber"),
    arrow(190, BR_Y + BR_H + 54, COMMIT_X + 60, COMMIT_Y - 6, "amber"),
    arrow(475, BR_Y + BR_H + 54, COMMIT_X + COMMIT_W / 2, COMMIT_Y - 6, "amber"),
    arrow(770, BR_Y + BR_H + 54, COMMIT_X + COMMIT_W - 60, COMMIT_Y - 6, "amber"),
    card(COMMIT_X, COMMIT_Y, COMMIT_W, COMMIT_H, "one Delta commit", "only the files the join touches are rewritten"),
    note(50, 400, "clauses are tried in", anchor="start"),
    note(50, 416, "the order they were", anchor="start"),
    note(50, 432, "added; first match wins", anchor="start"),
    note(700, 408, "no clause of the three", anchor="start"),
    note(700, 424, "is visible on its own", anchor="start"),
    band(20, 476, 940, 190, "THE ALTERNATIVE: DELETE, THEN APPEND", "grey"),
]

# Two mini-timelines. The merge is one tick; the two-statement version is two, with the
# interval between them shaded and named.
for lane_y, name in ((538, "MERGE INTO"), (596, "DELETE, THEN APPEND")):
    body += [
        note(40, lane_y + 4, name, anchor="start"),
        f'<path d="M 230 {lane_y} H 900" stroke="{GREY}" stroke-width="1.6" marker-end="url(#arG)"/>',
    ]

body += [
    # One commit: a single tick, and nothing in between to observe.
    f'<path d="M 330 524 V 552" stroke="{BLUE}" stroke-width="3"/>',
    note(330, 518, "one commit", anchor="middle"),
    note(560, 518, "readers see the table before it, or after it", anchor="middle"),
    # Two commits: the interval between them is the whole point.
    f'<rect x="420" y="582" width="240" height="28" rx="5" fill="{AMBER_DEEP}" '
    f'fill-opacity="0.16" stroke="{AMBER_DEEP}" stroke-width="1.4"/>',
    f'<path d="M 420 582 V 610" stroke="{AMBER_DEEP}" stroke-width="3"/>',
    f'<path d="M 660 582 V 610" stroke="{AMBER_DEEP}" stroke-width="3"/>',
    note(420, 576, "delete commits", anchor="middle"),
    note(660, 576, "append commits", anchor="middle"),
    f'<text x="540" y="600" text-anchor="middle" font-family="{FONT}" font-size="11.5" '
    f'font-weight="700" class="t-arrow">rows gone, not yet replaced</text>',
    note(40, 634, "One commit is a property of a transactional target. On a plain directory there is no log: the merge writes its new files and then deletes the ones it", anchor="start"),
    note(40, 652, "replaced, so a crash between the two leaves both copies of a key. A when_not_matched_by_source clause forces a full rewrite: no file can be skipped.", anchor="start"),
]

write("merge_into_branches", svg(W, H, "".join(body)))
print("wrote merge_into_branches.svg")
