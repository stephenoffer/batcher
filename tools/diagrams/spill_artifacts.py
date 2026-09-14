#!/usr/bin/env python3
"""Draw `spill_artifacts.svg` — the files a spill writes, and their whole lifecycle
including the end that `Drop` does not reach.

Source of truth: `crates/bc-runtime/src/agg/spill/store.rs` — `DiskSpillStore`
(`claim_scratch_dir`, the `bc-spill-{pid}-{seq}` name, `part-{i}.arrow`, the lazily
opened writers, `rows_per_partition` / `verify_rows`, `sweep_orphaned_scratch`,
`orphan_pid`, `process_is_alive`, `restrict_to_owner`, `SpillCodec::classify`) and
`python/batcher/carbonite/spill/scratch.py::scratch_dir` for where the root comes
from.

Two details the picture keeps because the code is careful about them: the directory
mode is best-effort, so it is drawn as an intent rather than a guarantee; and the
sweep only removes directories whose embedded pid is not a live process, which is
what keeps a concurrently spilling sibling safe.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 960, 610

body = [
    band(24, 20, 912, 216, "WHAT A SPILL PUTS ON DISK", "grey"),
    '<rect x="56" y="58" width="420" height="162" rx="10" class="surface" stroke-width="1.2"/>',
    label(72, 84, "&lt;spill root&gt;/"),
    note(72, 104, "memory.spill_dir, else the node's local scratch"),
    '<rect x="80" y="118" width="372" height="90" rx="8" class="band-blue" stroke-width="1.4"/>',
    label(96, 142, "bc-spill-{pid}-{seq}/"),
    note(96, 160, "one per store, owner-only where the filesystem allows"),
    note(96, 184, "part-0.arrow     part-1.arrow     ...     part-N.arrow"),
    note(96, 202, "one Arrow IPC stream each"),
    note(500, 86, "A partition is a hash bucket for the grace operators"),
    note(500, 104, "and a sorted run for the external sort. Same file."),
    note(500, 132, "A writer opens on the first append, so a bucket that"),
    note(500, 150, "received no rows has no file at all."),
    note(500, 178, "The codec comes from the first batch's schema: ZSTD"),
    note(500, 196, "for blob-bearing columns, none for anything else."),
    band(24, 252, 912, 150, "THE ORDINARY LIFECYCLE", "blue"),
    card(48, 294, 236, 66, "created", "writers open lazily, per file"),
    arrow(286, 327, 370, 327),
    label(328, 315, "append", anchor="middle"),
    card(374, 294, 236, 66, "appended", "rows and bytes counted per file"),
    arrow(612, 327, 696, 327),
    label(654, 315, "merge phase", anchor="middle"),
    card(700, 294, 236, 66, "read back once", "then the partition is released"),
    note(
        480,
        384,
        "The store removes its own directory on drop - on success, on an error and on a panic alike.",
        anchor="middle",
    ),
    band(24, 418, 912, 168, "THE TWO ENDS DROP DOES NOT REACH", "grey"),
    card(48, 458, 400, 72, "SIGKILL leaves it behind", "the OOM killer picks the spilling process"),
    arrow(448, 494, 540, 494, "amber"),
    label(494, 482, "swept by", anchor="middle"),
    card(540, 458, 372, 72, "orphan sweep", "removes only directories whose pid is dead"),
    note(
        48,
        560,
        "A truncated IPC stream reads back as a shorter valid one, so every read is checked against the row count taken on the way in.",
    ),
]

write("spill_artifacts", svg(W, H, "".join(body)))
print("wrote spill_artifacts.svg")
