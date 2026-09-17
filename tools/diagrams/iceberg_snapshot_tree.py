#!/usr/bin/env python3
"""Draw `iceberg_snapshot_tree.svg` -- catalog, snapshots, manifests, data files.

Source of truth: the page `docs/integrations/lakehouse/iceberg.md` and the Iceberg table
specification's metadata tree. The page states that a catalog maps an identifier such as
`db.orders` to a metadata file, that every commit is a snapshot and `snapshot_id=` reads an
older one, that `count()` is answered from the snapshot summary's `total-records`, that Kyber
pushes the predicate into `plan_files`, which pyiceberg answers against the manifests'
partition values and column bounds, and that Batcher makes one split per surviving data file,
carrying the manifest's record count. The "manifest list" edge between a snapshot and its
manifests is the specification's name for that link.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, mark, note, svg, tint, write

W, H = 980, 560

BH = 106  # band height
Y = (20, 158, 296, 434)  # band tops
NX = 610  # the note column, inside each band


def notes(y: float, lines: tuple[str, ...]) -> list[str]:
    """Right-hand notes: what Batcher does at this level of the tree."""
    return [note(NX, y + 50 + 19 * i, text) for i, text in enumerate(lines)]


body: list[str] = [
    band(20, Y[0], 940, BH, "CATALOG", "grey"),
    card(144, Y[0] + 36, 300, 56, "db.orders", "identifier to metadata file"),
    *notes(Y[0], ("catalog= takes a configured name", "or a property mapping.")),
    arrow(294, Y[0] + BH + 4, 294, Y[1] - 4),
    label(308, Y[0] + BH + 22, "table metadata", size=12),
    band(20, Y[1], 940, BH, "SNAPSHOTS  ·  ONE PER COMMIT", "blue"),
    card(44, Y[1] + 36, 190, 56, "Earlier snapshot", "snapshot_id=before"),
    tint(354, Y[1] + 36, 190, 56, "Current snapshot", "the default read"),
    arrow(240, Y[1] + 64, 348, Y[1] + 64),
    label(294, Y[1] + 54, "a write", "middle", 12),
    *notes(Y[1], ("snapshot_id= reads an older one.", "count() comes from total-records.")),
    arrow(449, Y[1] + BH + 4, 449, Y[2] - 4),
    label(463, Y[1] + BH + 22, "manifest list", size=12),
    band(20, Y[2], 940, BH, "MANIFESTS", "blue"),
    card(44, Y[2] + 36, 250, 56, "Manifest", "partition values, bounds"),
    card(314, Y[2] + 36, 250, 56, "Manifest", "partition values, bounds"),
    *notes(Y[2], ("Kyber pushes the predicate into", "plan_files, answered right here.")),
    arrow(304, Y[2] + BH + 4, 304, Y[3] - 4, "amber"),
    label(318, Y[2] + BH + 22, "files that can match", size=12),
    band(20, Y[3], 940, BH, "DATA FILES", "amber"),
]

# Five data files: three survive the manifest check, two never get scheduled.
for i, kept in enumerate((True, False, True, True, False)):
    x = 44 + i * 108
    body += [
        card(x, Y[3] + 36, 96, 56, "Parquet", "kept" if kept else "pruned"),
        mark(x + 88, Y[3] + 38, kept),
    ]
body += notes(Y[3], ("One split per kept file, sized by", "its record count. No footer read."))

write("iceberg_snapshot_tree", svg(W, H, "".join(body)))
print("wrote iceberg_snapshot_tree.svg")
