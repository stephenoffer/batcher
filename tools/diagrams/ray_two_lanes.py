#!/usr/bin/env python3
"""Draw `ray_two_lanes.svg` -- Ray schedules, Arrow Flight carries the shuffle.

Source of truth: the page `docs/integrations/compute/ray.md` and
`.claude/rules/python-control-plane.md` ("Distribution: Ray is scheduling only"). Ray carries
tasks, actors, placement groups, and small control-plane messages such as file paths, worker
addresses, and metrics. Bulk Arrow batches move worker to worker over Arrow Flight
(`bc-transport`) with credit-based flow control: one credit is one in-flight batch slot, and a
producer blocks when its credits reach zero. The Ray object store is not on that path.

Mapper and reducer are the page's own words for the shuffle's two sides.
"""

from __future__ import annotations

from _authoring import arrow, band, card, hero, label, mark, note, pill, svg, write

W, H = 980, 560

body: list[str] = [
    hero(330, 20, 320, 76, "Driver", "collect(distributed=True)"),
    arrow(490, 100, 490, 140),
    label(504, 124, "tasks, actors, placement groups", size=12),
    # The control lane.
    band(20, 146, 680, 118, "RAY  ·  SCHEDULING AND SMALL MESSAGES", "blue"),
    pill(44, 196, "tasks", "blue"),
    pill(126, 196, "actors", "blue"),
    pill(214, 196, "placement groups", "blue"),
    pill(44, 232, "file paths", "grey"),
    pill(152, 232, "worker addresses", "grey"),
    pill(304, 232, "metrics", "grey"),
    note(420, 204, "Everything here is small:"),
    note(420, 222, "descriptions of work,"),
    note(420, 240, "never the rows themselves."),
    # What is deliberately absent.
    card(720, 146, 240, 118, "Ray object store", "not on the data path"),
    # Down into the workers.
    arrow(360, 268, 360, 318),
    label(374, 298, "schedules each stage on", size=12),
    band(20, 322, 940, 218, "WORKERS  ·  SHUFFLE OVER ARROW FLIGHT (bc-transport)", "amber"),
    '<path d="M 840 270 L 840 284 M 840 308 L 840 322" stroke="#94a3b8" stroke-width="2"/>',
    mark(840, 296, False),
    label(858, 300, "bypassed", size=12),
    card(52, 372, 220, 84, "Mapper", "Rust engine, Arrow"),
    card(380, 372, 220, 84, "Reducer", "Rust engine, Arrow"),
    card(708, 372, 220, 84, "Mapper", "Rust engine, Arrow"),
    arrow(276, 414, 374, 414, "amber"),
    label(325, 400, "batches", "middle", 12),
    arrow(704, 414, 606, 414, "amber"),
    label(655, 400, "batches", "middle", 12),
    note(
        490,
        490,
        "Worker to worker, credit-bounded: one credit is one in-flight batch slot,",
        anchor="middle",
    ),
    note(490, 510, "and a producer blocks when its credits reach zero.", anchor="middle"),
]

write("ray_two_lanes", svg(W, H, "".join(body)))
print("wrote ray_two_lanes.svg")
