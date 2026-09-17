#!/usr/bin/env python3
"""Draw `partitioned_write_modes.svg` -- a Hive-partitioned output under the two overwrite modes.

Source of truth: `docs/user-guide/moving-data/writing-data.md` ("Partitioned output",
"Reloading one partition", "Confirm a write finished") and
`python/batcher/api/io_namespace/writer.py`: `_prune_stale_after_overwrite` deletes every
file a plain `overwrite` did not rewrite, and with `only_written_partitions` (the
`overwrite_partitions` mode) narrows that to the partition directories the incoming data
wrote into. Mode names are `SAVE_MODES` in `api/io_namespace/_write_opts.py`.

The directory listings were checked by running the page's own example (`dt` = a, b, c, then
a one-row reload of `b`): each partition directory holds `part-00000.parquet`, the root holds
`_SUCCESS`, a plain overwrite leaves only `dt=b/`, and `overwrite_partitions` leaves all
three with `dt=b` replaced.
"""

from __future__ import annotations

from _authoring import MONO, arrow, band, code, label, mark, note, svg, write

W, H = 1000, 490


def line(x: float, y: float, text: str, muted: bool = False) -> str:
    """One monospace line of a directory listing."""
    cls = "t-sub" if muted else "t-title"
    return (
        f'<text x="{x}" y="{y}" font-family="{MONO}" font-size="12.5" class="{cls}" '
        f'xml:space="preserve">{text}</text>'
    )


body: list[str] = [
    # ---- The state before, and the rows arriving ------------------------------------
    band(20, 24, 290, 206, "BEFORE: out/", "grey"),
    line(44, 80, "_SUCCESS", muted=True),
    line(44, 112, "dt=a/part-00000   v=1"),
    line(44, 142, "dt=b/part-00000   v=2"),
    line(44, 172, "dt=c/part-00000   v=3"),
    note(44, 208, "one directory per value of dt"),
    band(20, 250, 290, 112, "NEW ROWS", "blue"),
    line(44, 306, "dt=b   v=99"),
    note(44, 340, "they mention only dt=b"),
    # ---- The write ---------------------------------------------------------------------
    arrow(314, 130, 368, 180),
    label(318, 112, "target", size=11.5),
    arrow(314, 306, 368, 252),
    label(318, 340, "input", size=11.5),
    code(374, 170, ["write.parquet(", '  "out/",', '  partition_by=["dt"],', "  mode=...)"], 196),
    # ---- The two results ---------------------------------------------------------------
    arrow(576, 190, 640, 126, "grey"),
    label(560, 142, "overwrite", anchor="middle", size=11.5),
    arrow(576, 246, 640, 316, "blue"),
    label(566, 306, "overwrite_", anchor="middle", size=11.5),
    label(566, 321, "partitions", anchor="middle", size=11.5),
    band(646, 24, 334, 180, 'MODE="OVERWRITE"', "grey"),
    line(704, 80, "_SUCCESS", muted=True),
    mark(680, 108, False),
    line(704, 112, "dt=a/   deleted", muted=True),
    mark(680, 138, True),
    line(704, 142, "dt=b/   v=99"),
    mark(680, 168, False),
    line(704, 172, "dt=c/   deleted", muted=True),
    band(646, 226, 334, 180, 'MODE="OVERWRITE_PARTITIONS"', "blue"),
    line(704, 282, "_SUCCESS", muted=True),
    mark(680, 310, True),
    line(704, 314, "dt=a/   v=1    kept"),
    mark(680, 340, True),
    line(704, 344, "dt=b/   v=99   replaced"),
    mark(680, 370, True),
    line(704, 374, "dt=c/   v=3    kept"),
    # ---- What to take away -----------------------------------------------------------
    note(
        500,
        442,
        "A plain overwrite replaces the whole output, so partitions the new rows never mention "
        "are deleted.",
        anchor="middle",
    ),
    note(
        500,
        464,
        "overwrite_partitions needs partition_by. _SUCCESS is written last, and readers skip it.",
        anchor="middle",
    ),
]

write("partitioned_write_modes", svg(W, H, "".join(body)))
print("wrote partitioned_write_modes.svg")
