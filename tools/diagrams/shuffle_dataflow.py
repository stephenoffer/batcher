#!/usr/bin/env python3
"""Draw `shuffle_dataflow.svg` -- the two planes a shuffle runs on, side by side.

Source of truth, read before drawing:
  * `crates/bc-transport/src/{exchange,handler,ticket,store}.rs` -- one Flight server per
    worker process over a shared `Arc<PartitionStore>`; batches live in memory, not on disk;
    `ShuffleTicket` is `{plan_id, stage_id, src_partition, dst_partition, epoch}` and
    serializes to `"{plan}/{stage}/{src}/{dst}/{epoch}"` in `flight_descriptor.path[0]`.
    Only `do_exchange` is on the production path.
  * `crates/bc-py/src/shuffle.rs::drive` -- co-located buckets read from the local store with
    no socket and no credit permit; the rest spawned into a `JoinSet` bounded by
    `flow_control.shuffle_fetch_fan_in` (32); arrivals folded in Rust by `gather_combine`.
  * `python/batcher/dist/flight_worker.py` -- the Ray actor returns an **address**, not
    batches; `plan_id` is minted per query as a 63-bit value from a uuid4.
  * `python/batcher/dist/executors/aggregate.py` -- the disk map task returns a `list[str]`
    of file paths.

The load-bearing fact, and the only reason this is a figure rather than a sentence: the two
planes are drawn as two physically separate channels between the same pair of workers. The
sentence "bulk data bypasses the Ray object store" is easy to read past; a picture in which
the Ray channel is visibly a thin string of identifiers while the batches take a different
wire is not. Every item on the Ray side is one this repository's return types actually carry
-- addresses, tickets, paths, row counts, a metrics JSON string -- and nothing else.

Layout: mappers left, reducers right, the data plane on the lower rail and the control plane
on the upper one, so the split is spatial rather than asserted in a caption.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, BLUE, BLUE_MID, GREY, arrow, band, card, label, note, svg, write

W, H = 980, 560


def chip(x: float, y: float, w: float, h: float, kind: str = "blue") -> str:
    """A bucket: one mapper's rows for one reducer."""
    fill = {"blue": BLUE_MID, "grey": GREY}[kind]
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" '
        f'fill-opacity="0.28" stroke="{fill}" stroke-width="1.4"/>'
    )


body: list[str] = []

# ---- The control plane: what Ray carries --------------------------------------------
body.append(band(20, 24, 940, 118, "CONTROL PLANE -- THROUGH RAY", "grey"))
body.append(card(300, 52, 380, 62, "Ray tasks and actors", "schedules the workers"))
body.append(note(490, 128, "an address, a ticket, a file path, a row count, a metrics JSON "
                           "string -- and nothing else", anchor="middle"))

# The two thin control edges. Grey and dashed-thin on purpose: the volume is the point.
body.append(
    f'<path d="M 150 246 L 150 118 L 296 118" fill="none" stroke="{GREY}" stroke-width="1.6" '
    f'stroke-dasharray="4 4" marker-end="url(#arG)"/>'
)
body.append(label(160, 174, "flight address", size=11.5))
body.append(
    f'<path d="M 684 118 L 830 118 L 830 246" fill="none" stroke="{GREY}" stroke-width="1.6" '
    f'stroke-dasharray="4 4" marker-end="url(#arG)"/>'
)
body.append(label(824, 174, "ticket", anchor="end", size=11.5))

# ---- The data plane: what never touches Ray -----------------------------------------
body.append(band(20, 216, 940, 234, "DATA PLANE -- ARROW FLIGHT, NEVER THE OBJECT STORE", "blue"))

body.append(card(44, 252, 212, 92, "mappers", "partition_batches"))
x = 60
for _ in range(4):
    body.append(chip(x, 362, 42, 30))
    x += 48
body.append(note(150, 412, "every bucket published,", anchor="middle"))
body.append(note(150, 428, "including the empty ones", anchor="middle"))

# The fat data rail. Drawn thick so it reads as bulk against the thin control edges above.
body.append(
    f'<path d="M 262 298 L 718 298" fill="none" stroke="{BLUE}" stroke-width="7" '
    f'marker-end="url(#arB)"/>'
)
body.append(label(490, 276, "Arrow record batches, credit-bounded", anchor="middle"))
body.append(note(490, 320, "one credit is one batch slot; the producer blocks at zero",
                 anchor="middle"))
body.append(note(490, 342, "do_exchange over gRPC, LZ4 by default", anchor="middle"))

body.append(card(724, 252, 212, 92, "reducers", "gather_combine, in Rust"))
body.append(note(830, 382, "folded into a running partial:", anchor="middle"))
body.append(note(830, 398, "the intermediate never crosses", anchor="middle"))
body.append(note(830, 414, "back into Python", anchor="middle"))

# ---- The bypass, stated once, where the reader is looking ----------------------------
body.append(band(20, 464, 940, 78, "THE ONE THING TO TAKE AWAY", "amber"))
body.append(
    f'<path d="M 300 504 L 680 504" fill="none" stroke="{AMBER_DEEP}" stroke-width="2.4" '
    f'stroke-dasharray="7 5"/>'
    f'<path d="M 476 488 L 504 520" fill="none" stroke="{AMBER_DEEP}" stroke-width="3"/>'
    f'<path d="M 504 488 L 476 520" fill="none" stroke="{AMBER_DEEP}" stroke-width="3"/>'
)
body.append(label(296, 509, "bulk batches", anchor="end"))
body.append(label(684, 509, "the Ray object store"))
body.append(note(490, 534, "routing them through it reintroduces the serialization the "
                           "columnar design removes", anchor="middle"))

write("shuffle_dataflow", svg(W, H, "".join(body)))
print("wrote shuffle_dataflow.svg")
