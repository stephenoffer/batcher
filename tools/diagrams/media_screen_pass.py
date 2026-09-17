#!/usr/bin/env python3
"""Draw `media_screen_pass.svg`: screen, decode and fingerprint images in one plan.

Source of truth: the runnable example in `docs/ml/preparing/multimodal/index.md`. Two
PNGs enter; `col("bytes").image.brightness() > 0.05` drops the black one; the survivor
gets `image.to_tensor(8, 8)`, which the example prints as a `fixed_shape_tensor` of
`uint8` with shape `[8, 8, 3]`, and `image.phash()`, which
`python/batcher/plan/expr_ir/image.py` documents as a 64-bit DCT perceptual hash
returned as an Int64. The page states that the decode is a Rust expression over whole
batches, so a million images never become a million Python calls.

Layout: input on the left, the screen in the middle with the dropped row falling below
it, then a fork into the two expressions that run on the survivor, joined at the output.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, pill, svg, tint, write

W, H = 980, 430

body = [
    band(16, 20, 948, 314, "ONE PLAN  ·  ONE collect()", "blue"),
    # Input: the two rows the example builds.
    card(40, 128, 176, 96, "bytes column", "two PNGs"),
    pill(128, 250, "id 1  noisy", "grey", anchor="middle"),
    pill(128, 280, "id 2  black", "grey", anchor="middle"),
    # The screen.
    tint(290, 128, 196, 96, "Screen", "brightness() > 0.05"),
    arrow(222, 176, 284, 176, "blue"),
    label(253, 164, "2 rows", anchor="middle", size=11.5),
    # The dropped row falls out of the plan.
    arrow(388, 230, 388, 266, "amber"),
    label(398, 254, "id 2 fails", size=11.5),
    note(388, 292, "never reaches the", anchor="middle"),
    note(388, 310, "tensor or the hash", anchor="middle"),
    # The survivor forks into the two expressions, evaluated in the same pass.
    tint(572, 62, 180, 88, "Decode", "to_tensor(8, 8)", kind="amber"),
    tint(572, 202, 180, 88, "Fingerprint", "phash()", kind="amber"),
    arrow(492, 164, 564, 112, "blue"),
    arrow(492, 188, 564, 240, "blue"),
    label(516, 180, "id 1", size=11.5),
    # The output row, with the type each expression produced.
    card(822, 128, 124, 96, "one row", "id, image, phash"),
    arrow(758, 106, 816, 150, "blue"),
    label(772, 98, "image"),
    arrow(758, 246, 816, 202, "blue"),
    label(772, 266, "phash"),
    note(884, 252, "image: 8x8x3 uint8", anchor="middle"),
    note(884, 272, "phash: Int64", anchor="middle"),
    # Why this is cheap.
    band(16, 352, 948, 58, "", "grey"),
    note(
        490,
        386,
        "Each step is an expression in Rust over whole batches, so no image becomes a Python call.",
        anchor="middle",
    ),
]

write("media_screen_pass", svg(W, H, "".join(body)))
print("wrote media_screen_pass.svg")
