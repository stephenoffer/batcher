#!/usr/bin/env python3
"""Draw `blob_offload.svg`: a payload carried inline, against a handle and a late fetch.

Source of truth: `docs/ml/preparing/multimodal/pipelines.md` ("Keep large payloads out of
shuffles and spills"), `python/batcher/api/dataset/frame.py` (`offload_blobs`,
`materialize_blobs`) and `python/batcher/io/formats/multimodal/blob.py`. Offload writes
each payload to `{root}/{sha256}`, deduped by content, and leaves a short URI handle in
a `uri` column; the root defaults to `spill_remote_uri` when set, else the local spill
directory (`default_blob_root`). Materialize reads each handle back into a
`large_binary` column. `execution.auto_offload_blobs` places the pair around a sort for
`large_binary` columns the sort does not key on, and is off by default.

Layout: the inline path on top, the offloaded path beneath it on the same columns, and
the store below the offloaded path with the write and the read-back as dashed curves.
"""

from __future__ import annotations

from _authoring import arrow, band, card, curve, label, note, svg, tint, write

W, H = 980, 600

XS = (40, 280, 520, 760)
CW = 180
TOP_Y, TOP_H = 72, 76
BOT_Y, BOT_H = 250, 76

body = [
    # ---- Before: the payload rides every operator ------------------------------
    band(16, 20, 948, 160, "INLINE  ·  THE PAYLOAD RIDES EVERY OPERATOR", "grey"),
    card(XS[0], TOP_Y, CW, TOP_H, "id, payload", "GB of bytes per row"),
    card(XS[1], TOP_Y, CW, TOP_H, "sort by id", "copies the payload"),
    card(XS[2], TOP_Y, CW, TOP_H, "join", "copies it again"),
    card(XS[3], TOP_Y, CW, TOP_H, "next step", "finally reads the bytes"),
    note(
        490,
        168,
        "Sort and join only touch id, yet every copy and every spill buffer "
        "carries the full payload.",
        anchor="middle",
    ),
    # ---- After: only a handle rides ---------------------------------------------
    band(16, 198, 948, 382, "OFFLOADED  ·  ONLY A HANDLE RIDES", "blue"),
    tint(XS[0], BOT_Y, CW, BOT_H, "offload_blobs", "payload -> uri handle", kind="amber"),
    card(XS[1], BOT_Y, CW, BOT_H, "sort by id", "moves short strings"),
    tint(XS[2], BOT_Y, CW, BOT_H, "materialize_blobs", "handle -> bytes", kind="amber"),
    card(XS[3], BOT_Y, CW, BOT_H, "next step", "the bytes, fetched late"),
    card(
        280,
        436,
        420,
        74,
        "Content-addressed store",
        "{root}/{sha256}: spill_remote_uri, else local spill dir",
    ),
    curve(130, BOT_Y + BOT_H + 6, 150, 470, 272, 470, "amber"),
    label(150, 382, "write once"),
    label(150, 400, "per hash"),
    curve(630, 430, 648, 382, 624, BOT_Y + BOT_H + 8, "amber"),
    label(656, 382, "read back"),
    label(656, 400, "just in time"),
    note(
        490,
        550,
        "auto_offload_blobs=True places this pair around a sort for large_binary "
        "columns. It is off by default.",
        anchor="middle",
    ),
]

mid_top = TOP_Y + TOP_H / 2
mid_bot = BOT_Y + BOT_H / 2
for i, (top_text, bot_text) in enumerate(
    (("bytes", "handle"), ("bytes", "handle"), ("bytes", "bytes"))
):
    x1 = XS[i] + CW + 6
    x2 = XS[i + 1] - 6
    kind = "amber" if i == 2 else "blue"
    body += [
        arrow(x1, mid_top, x2, mid_top, "grey"),
        label((x1 + x2) / 2, mid_top - 10, top_text, anchor="middle", size=11.5),
        arrow(x1, mid_bot, x2, mid_bot, kind),
        label((x1 + x2) / 2, mid_bot - 10, bot_text, anchor="middle", size=11.5),
    ]

write("blob_offload", svg(W, H, "".join(body)))
print("wrote blob_offload.svg")
