#!/usr/bin/env python3
"""Draw `gpu_fabric_topology.svg` -- the three fabric facts Batcher reads, and the schedule.

Source of truth, read before drawing:
  * `python/batcher/_internal/hardware/fabric/p2p.py` -- `peer_matrix` overlays the live
    NVLink pairs onto the bus matrix so `"nvlink"` is a class alongside `pix`/`pxb`/`phb`/
    `node`/`sys`, and `peer_islands` reports the connected groups a collective can stay
    inside. An unreadable pair reports `sys`, never the fabric.
  * `python/batcher/_internal/hardware/fabric/rails.py` -- `assign_rails` is the node-wide
    decision: a device takes its closest NIC unless that NIC already holds its share and an
    equally close one is free, and balance never overrides distance. `rail_imbalance` is the
    figure worth alerting on, because a node with every device on one rail carries a fraction
    of its port rate while every counter reads healthy.
  * `python/batcher/carbonite/transfer/device_exchange.py` -- `pairwise_rounds` (the circle
    method: `n-1` rounds of `n/2` disjoint pairs), `ring_order` (greedy nearest-neighbour,
    keeping whichever tour has the best *worst* hop), and `worth_device_exchange`, which
    requires a `_WORTH_MARGIN` of 1.25 over the host path before the device path is used.
  * `python/batcher/dist/executors/ray_runtime/fabric/placement.py` -- a collective is
    strict-packed inside one domain, and a plan covering fewer devices than the stage asked
    for is refused rather than reserved.

The exchange rounds drawn in the middle band are the literal output of
`pairwise_rounds([0, 1, 2, 3])`, not an illustration of one.

This is a topology, which is the case a diagram exists for: the same eight devices are one
island or two depending on wires nothing in a job's timings reports, and both arrangements
return correct results. Prose can say that. It cannot show a reader the difference between the
two pictures, which is the whole content.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, BLUE, BLUE_MID, GREY, band, card, label, note, svg, write

W, H = 980, 640


def dev(x: float, y: float, n: int) -> str:
    """One accelerator, labelled by ordinal."""
    return (
        f'<rect x="{x}" y="{y}" width="44" height="34" rx="5" fill="{BLUE_MID}" '
        f'fill-opacity="0.28" stroke="{BLUE_MID}" stroke-width="1.4"/>'
        f'<text x="{x + 22}" y="{y + 22}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" '
        f'font-size="12" font-weight="700" class="t-title">{n}</text>'
    )


def wire(x1: float, y1: float, x2: float, y2: float, kind: str) -> str:
    """A link between two devices. `nvlink` is solid and heavy; `bus` is thin and dashed."""
    if kind == "nvlink":
        return f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{BLUE}" stroke-width="4"/>'
    return (
        f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{GREY}" stroke-width="1.6" '
        f'stroke-dasharray="4 4"/>'
    )


body: list[str] = []

# ---- Panel 1: peer islands --------------------------------------------------------------
body.append(band(20, 24, 458, 246, "PEER ISLANDS", "blue"))
body.append(note(44, 60, "the connected groups of the NVLink-over-bus overlay"))

for i in range(4):
    body.append(dev(52 + i * 56, 84, i))
for i in range(4):
    body.append(dev(280 + i * 56, 84, i + 4))
for group in (52, 280):
    for i in range(3):
        body.append(wire(group + 96 + i * 56, 101, group + 112 + i * 56, 101, "nvlink"))
    body.append(wire(group + 22, 76, group + 190, 76, "nvlink"))
body.append(wire(226, 101, 278, 101, "bus"))

body.append(label(120, 140, "island of 4", anchor="middle", size=11.5))
body.append(label(348, 140, "island of 4", anchor="middle", size=11.5))
body.append(label(252, 140, "sys", anchor="middle", size=11.5))

body.append(note(44, 176, "Heavy line: a coherent fabric link. Dashed: the PCI bus."))
body.append(note(44, 196, "Two devices under different root complexes are the"))
body.append(note(44, 216, "furthest apart the bus can express, and exchange at"))
body.append(note(44, 236, "full fabric rate anyway if NVLink joins them. A group"))
body.append(note(44, 256, "picked on bus distance alone picks the wrong four."))

# ---- Panel 2: rails ----------------------------------------------------------------------
body.append(band(502, 24, 458, 246, "RAILS", "blue"))
body.append(note(526, 60, "which NIC each device leaves the node through"))

for i in range(4):
    body.append(dev(534 + i * 56, 84, i))
for i in range(4):
    body.append(dev(762 + i * 56, 84, i + 4))

body.append(card(534, 156, 196, 44, "mlx5_0", ""))
body.append(card(762, 156, 196, 44, "mlx5_1", ""))
for i in range(4):
    body.append(wire(556 + i * 56, 118, 632, 154, "bus"))
    body.append(wire(784 + i * 56, 118, 860, 154, "bus"))
body.append(label(746, 140, "assigned node-wide", anchor="middle", size=11.5))

body.append(note(526, 224, "Asked one at a time, all eight devices can name the"))
body.append(note(526, 244, "same NIC: one rail carries the shuffle and seven sit"))
body.append(note(526, 264, "idle, with every counter reporting a healthy fabric."))

# ---- The exchange schedule ---------------------------------------------------------------
body.append(
    band(20, 292, 940, 206, "THE EXCHANGE SCHEDULE: n-1 ROUNDS OF n/2 DISJOINT PAIRS", "amber")
)
body.append(
    note(
        44,
        328,
        "pairwise_rounds([0, 1, 2, 3]) -- no device is the source of one "
        "copy and the destination of another in the same round",
    )
)

rounds = ((0, ((0, 3), (1, 2))), (1, ((0, 2), (1, 3))), (2, ((0, 1), (2, 3))))
for r, pairs in rounds:
    ox = 60 + r * 300
    body.append(label(ox + 110, 364, f"round {r}", anchor="middle"))
    for i in range(4):
        body.append(dev(ox + i * 56, 378, i))
    for idx, (a, b) in enumerate(pairs):
        ax, bx = ox + a * 56 + 22, ox + b * 56 + 22
        lift = 424 + idx * 11
        body.append(
            f'<path d="M {ax} 414 L {ax} {lift} L {bx} {lift} L {bx} 414" fill="none" '
            f'stroke="{AMBER_DEEP}" stroke-width="2.4"/>'
        )
    body.append(note(ox + 110, 462, " and ".join(f"{a}-{b}" for a, b in pairs), anchor="middle"))

body.append(
    note(
        490,
        486,
        "The ring for a reduction is ordered by the fabric, not by device "
        "index: its rate is its worst hop.",
        anchor="middle",
    )
)

# ---- What changes because of it ----------------------------------------------------------
body.append(band(20, 520, 940, 100, "WHAT CHANGES BECAUSE OF IT", "grey"))
body.append(label(44, 556, "a collective is strict-packed inside one island"))
body.append(note(44, 576, "a plan covering fewer devices than the stage asked"))
body.append(note(44, 594, "for is refused, because a partial gang hangs"))

body.append(label(400, 556, "shards are dealt by measured throughput"))
body.append(note(400, 576, "largest-remainder apportionment; an unmeasured"))
body.append(note(400, 594, "device is treated as average, never as idle"))

body.append(label(700, 556, "the device path must win by 1.25x"))
body.append(note(700, 576, "an unpriced link makes a plan refuse;"))
body.append(note(700, 594, "a plan that merely ties loses"))

write("gpu_fabric_topology", svg(W, H, "".join(body)))
print("wrote gpu_fabric_topology.svg")
