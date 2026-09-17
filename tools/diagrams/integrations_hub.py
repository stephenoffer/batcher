#!/usr/bin/env python3
"""Draw `integrations_hub.svg` -- the six integration groups around the engine.

Source of truth: `docs/integrations/index.md` (its toctree and group table) and the group
index pages under `docs/integrations/`. The four groups on the left hold data: connectors
divide them into splits that read in parallel and push filters down, and results are written
back where the connector has a sink (BigQuery and Databricks have none, which the figure
states in its footer rather than implying every connector writes). On the right, compute
and ML integrations schedule the work and consume its output, and observability receives the
metrics, traces, and lineage events the engine emits. Every connector is built on the public
`Source`, `Sink`, and `Split` contracts.
"""

from __future__ import annotations

from _authoring import arrow, band, card, heading, hero, label, note, pill, ribbon, svg, write

W, H = 980, 524

LEFT = (
    ("Streams", "Kafka, Kinesis, Pulsar, Pub/Sub"),
    ("Warehouses", "Snowflake, BigQuery, Databricks"),
    ("Lakehouse", "Delta Lake, Iceberg, Hudi"),
    ("Databases", "SQL, key-value, MongoDB"),
)
LX, LW, LH = 30, 250, 64
LY0, LSTEP = 70, 84
HX, HY, HW, HH = 410, 176, 180, 110
HCY = HY + HH / 2
RX, RW, RH = 716, 244, 72

body: list[str] = [
    heading(LX, 40, "WHERE DATA LIVES", kind="grey"),
    heading(RX, 40, "AROUND THE ENGINE", kind="grey"),
]

for i in range(len(LEFT)):
    cy = LY0 + i * LSTEP + LH / 2
    body.append(ribbon(LX + LW, cy, HX, HCY - 24 + i * 16, "blue"))
for i, (title, sub) in enumerate(LEFT):
    body.append(card(LX, LY0 + i * LSTEP, LW, LH, title, sub))

body += [
    hero(HX, HY, HW, HH, "Batcher", "Arrow in, Arrow out"),
    pill(HX + HW / 2, HY - 16, "parallel reads in", "blue", anchor="middle"),
    pill(HX + HW / 2, HY + HH + 26, "writes back out", "amber", anchor="middle"),
    # Right side.
    card(RX, 132, RW, RH, "Compute and ML", "Ray, schedulers, PyTorch"),
    card(RX, 248, RW, RH, "Observability", "metrics, traces, lineage"),
    arrow(HX + HW + 6, HCY - 18, RX - 8, 168),
    label(624, 182, "scheduling", "middle", 11.5),
    arrow(HX + HW + 6, HCY + 18, RX - 8, 284, "amber"),
    label(624, 280, "signals", "middle", 11.5),
    band(20, 420, 940, 84, "EVERY CONNECTOR", "grey"),
    note(
        44, 466, "Built on the same public contracts, Source, Sink and Split, so an unlisted system"
    ),
    note(
        44,
        486,
        "plugs in the same way. Not every connector writes: BigQuery and Databricks have no sink.",
    ),
]

write("integrations_hub", svg(W, H, "".join(body)))
print("wrote integrations_hub.svg")
