#!/usr/bin/env python3
"""Draw `inference_stages.svg` — an overlapped, credit-bounded inference pipeline.

Source of truth: `python/batcher/ml/pipeline.py` (`Stage`, `run_pipeline`). Each
stage runs on its own thread with its worker built once, and `credits` bounds how
many finished batches may sit between one stage and the next. That bound is the
whole point of the picture: it is what stops a fast decoder from filling memory
ahead of a slow GPU, and it is why the pipeline streams rather than materializes.

`num_gpus` on a stage is a placement hint for the distributed scheduler, with 0
meaning CPU. Keep this diagram in step with that module's docstrings.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 430

ROW_Y = 104
MID = ROW_Y + 45  # vertical centre of the card row, for the connecting arrows

body = [
    band(20, 20, 940, 250, "ONE PIPELINE  ·  EACH STAGE ON ITS OWN THREAD", "blue"),
    card(44, ROW_Y, 176, 90, "Source", "Arrow batches"),
    card(300, ROW_Y, 200, 90, "CPU stage", "decode, tokenize"),
    card(580, ROW_Y, 200, 90, "GPU stage", "the model"),
    card(830, ROW_Y, 106, 90, "Out", "batches"),
    # The two credit windows are the load-bearing detail, so they are labeled.
    arrow(220, MID, 300, MID, "blue"),
    label(260, MID - 18, "credits", anchor="middle", size=12),
    arrow(500, MID, 580, MID, "amber"),
    label(540, MID - 18, "credits", anchor="middle", size=12),
    arrow(780, MID, 830, MID, "blue"),
    note(400, 232, "A stage blocks once its credit window to the next stage is full.", anchor="middle"),
    # What the bound buys you.
    band(20, 292, 940, 118, "WHY THE WINDOW IS BOUNDED", "grey"),
    note(180, 338, "Fast stage cannot", anchor="middle"),
    note(180, 356, "run ahead into memory.", anchor="middle"),
    note(490, 338, "Stages overlap, so the GPU", anchor="middle"),
    note(490, 356, "is fed while the CPU decodes.", anchor="middle"),
    note(800, 338, "Output order is preserved,", anchor="middle"),
    note(800, 356, "and the run streams.", anchor="middle"),
    note(490, 388, "Set credits per stage; num_gpus=0 marks a stage as CPU.", anchor="middle"),
]

write("inference_stages", svg(W, H, "".join(body)))
print("wrote inference_stages.svg")
