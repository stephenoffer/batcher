#!/usr/bin/env python3
"""Draw `observability_bus.svg`: one event bus fanning out to what you can see.

Source of truth:

* `docs/user-guide/operate/running/observability.md`: every subsystem (Kyber, Carbonite, Core,
  the distributed scheduler) publishes to one internal event bus; the terminal progress bar,
  the web dashboard and the process-wide metrics counters consume it; the JSON event log holds
  the plans, decisions and measured per-operator profile; OpenTelemetry emits one span per
  query from the same measured profile as the event log.
* `python/batcher/_internal/events.py`: `publish` / `subscribe`, free when nobody listens.
* `python/batcher/observe/console/reporter.py`, `observe/store.py` (behind the dashboard) and
  `observe/metrics.py`: the three `events.subscribe` call sites.
* `python/batcher/api/terminal/event_log.py::write_event_log`: the per-query profile is
  assembled once when the query finishes, published onto the bus (`_publish_stages`,
  `_publish_end`), written as the on-disk document when `event_log` is on, and handed to
  `emit_query_spans`. That is why the event log and OpenTelemetry are drawn off the profile
  rather than as bus subscribers, and why the dashboard still shows the same measurements.
* `python/batcher/_internal/logging.py`: log records are also published as `LOG` events,
  which is what the dashboard's Logs page reads.

OpenLineage is left out: it carries governance's column lineage, not the profile, and would
be an eighth box for a different fact.
"""

from __future__ import annotations

from _authoring import (
    arrow,
    band,
    hero,
    label,
    note,
    ribbon,
    svg,
    tint,
    write,
)

W, H = 980, 540

PUB_X, PUB_W, PUB_H = 44, 196, 38
CON_X, CON_W, CON_H = 724, 212, 54
BUS_X, BUS_Y, BUS_W, BUS_H = 336, 122, 220, 96
BUS_CY = BUS_Y + BUS_H / 2

body: list[str] = [
    band(20, 20, 940, 290, "WHILE THE QUERY RUNS", "blue"),
    band(20, 330, 940, 190, "AT QUERY END", "amber"),
]

# ---- Publishers ---------------------------------------------------------------------------------
publishers = ("Kyber", "Carbonite", "Core", "distributed scheduler", "log records")
for i, name in enumerate(publishers):
    y = 64 + i * 46
    body.append(tint(PUB_X, y, PUB_W, PUB_H, name))
    body.append(ribbon(PUB_X + PUB_W + 2, y + PUB_H / 2, BUS_X - 4, BUS_CY))
body.append(label(288, 304, "publish", anchor="middle", size=11.5))

body.append(hero(BUS_X, BUS_Y, BUS_W, BUS_H, "event bus", "one channel, one query_id"))

# ---- Subscribers --------------------------------------------------------------------------------
subscribers = (
    ("terminal progress bar", "on in a real terminal", "live phase"),
    ("web dashboard", "bt.start_ui()", "runs, plans, logs"),
    ("metrics counters", "process-wide", "throughput, durations"),
)
for i, (name, sub, what) in enumerate(subscribers):
    y = 70 + i * 76
    cy = y + CON_H / 2
    body.append(tint(CON_X, y, CON_W, CON_H, name, sub))
    body.append(arrow(BUS_X + BUS_W + 6, BUS_CY + (i - 1) * 22, CON_X - 8, cy))
    body.append(label(640, (98, 160, 266)[i], what, anchor="middle", size=11))

# ---- The profile ---------------------------------------------------------------------------------
PROF_Y = 392
body += [
    tint(BUS_X, PROF_Y, BUS_W, 66, "query profile", "plans, decisions, per-operator", "amber"),
    arrow(BUS_X + BUS_W / 2, PROF_Y - 6, BUS_X + BUS_W / 2, BUS_Y + BUS_H + 8, "amber"),
    label(BUS_X + BUS_W / 2 - 12, 372, "published as stages", anchor="end", size=11.5),
    tint(CON_X, 362, CON_W, CON_H, "JSON event log", "on by default, on disk"),
    tint(CON_X, 440, CON_W, CON_H, "OpenTelemetry spans", "when otel_traces is on"),
    arrow(BUS_X + BUS_W + 6, PROF_Y + 22, CON_X - 8, 389, "amber"),
    label(640, 376, "same document", anchor="middle", size=11),
    arrow(BUS_X + BUS_W + 6, PROF_Y + 44, CON_X - 8, 467, "amber"),
    label(640, 484, "same profile", anchor="middle", size=11),
    note(162, 430, "Measured once, so the", anchor="middle"),
    note(162, 448, "dashboard and the event", anchor="middle"),
    note(162, 466, "log can't disagree.", anchor="middle"),
]

write("observability_bus", svg(W, H, "".join(body)))
print("wrote observability_bus.svg")
