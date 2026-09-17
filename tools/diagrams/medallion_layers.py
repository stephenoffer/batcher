#!/usr/bin/env python3
"""Draw `medallion_layers.svg`: bronze to silver to gold, as the lakehouse tutorial builds it.

Source of truth: `docs/getting-started/tutorials/pipelines/building-a-lakehouse.md`.
Bronze is raw Parquet with no cleaning and no dedup. Silver is a Delta table written by
`mode="overwrite"` from `filter(status == "paid")` with `partition_by=["day"]`. Gold is a
plain `group_by("day")` over silver. After the first load, `merge_on=` runs a native Delta
`MERGE INTO` as one commit, `replace_where=` atomically replaces exactly the matching rows
so re-running it is a no-op, and `version=0` reads the table as `overwrite` left it. Step 8
of the page says the transaction log records each file's partition values and min/max,
which the read prunes against.
"""

from __future__ import annotations

from _authoring import arrow, band, card, hero, label, note, svg, write

W, H = 980, 470

TOP_Y, TOP_H = 72, 104
TOP_MID = TOP_Y + TOP_H / 2
SILVER_X, SILVER_W = 380, 220
SILVER_BOTTOM = TOP_Y + TOP_H + 4
LOW_Y, LOW_W, LOW_H = 322, 256, 84
LOW_XS = (44, 362, 680)

body = [
    band(20, 20, 940, 186, "MEDALLION LAYERS", "blue"),
    card(44, TOP_Y, 236, TOP_H, "Bronze", "raw Parquet, no cleaning"),
    hero(SILVER_X, TOP_Y - 4, SILVER_W, TOP_H + 8, "Silver", "Delta table, by day"),
    card(700, TOP_Y, 236, TOP_H, "Gold", "a plain query over silver"),
    arrow(280, TOP_MID, SILVER_X - 2, TOP_MID),
    label((280 + SILVER_X) / 2, TOP_MID - 12, "overwrite", anchor="middle", size=11.5),
    note((280 + SILVER_X) / 2, TOP_MID + 22, "paid rows", anchor="middle"),
    arrow(SILVER_X + SILVER_W, TOP_MID, 698, TOP_MID),
    label((SILVER_X + SILVER_W + 700) / 2, TOP_MID - 12, "group_by", anchor="middle", size=11.5),
    note((SILVER_X + SILVER_W + 700) / 2, TOP_MID + 22, "sum by day", anchor="middle"),
    band(20, 230, 940, 220, "AFTER THE FIRST LOAD", "amber"),
    # Two later commits into silver, and one read of an earlier version out of it.
    card(LOW_XS[0], LOW_Y, LOW_W, LOW_H, 'merge_on="order_id"', "upsert, one commit"),
    arrow(LOW_XS[0] + LOW_W - 40, LOW_Y, SILVER_X + 30, SILVER_BOTTOM + 4, "amber"),
    label(284, 296, "commit", anchor="end", size=11.5),
    card(LOW_XS[1], LOW_Y, LOW_W, LOW_H, "version=0", "the table before the merge"),
    arrow(490, SILVER_BOTTOM, 490, LOW_Y - 4, "blue"),
    label(502, 296, "read", size=11.5),
    card(LOW_XS[2], LOW_Y, LOW_W, LOW_H, "replace_where=pred", "backfill, safe to re-run"),
    arrow(LOW_XS[2] + 40, LOW_Y, SILVER_X + SILVER_W - 30, SILVER_BOTTOM + 4, "amber"),
    label(696, 296, "commit", size=11.5),
    note(
        490,
        434,
        "Every commit is a version. Pruning reads each file's bounds from the log.",
        anchor="middle",
    ),
]

write("medallion_layers", svg(W, H, "".join(body)))
print("wrote medallion_layers.svg")
