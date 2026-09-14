#!/usr/bin/env python3
"""Draw `hardware_awareness.svg` -- how the machine's real shape reaches a plan decision.

The topology is the point, and it is a fold: every fact here is the **tightest** of
several bounds rather than the one the obvious API reports. Prose can state that; it
cannot hold the shape of three bounds converging and then fanning out to three unrelated
decisions.

Source, and what to keep in step:

* `crates/bc-arrow/src/hardware.rs::usable_cores` -- `available_parallelism` (which
  honours the affinity mask but **not** the CFS bandwidth quota) capped by
  `cfs_quota_cores()` and `scheduler_granted_cores()`, never below 1. The Kubernetes
  `cpu` limit is a bandwidth quota, which is why a pod limited to 15 cores on a 16-core
  node reports 16 and sizes every pool one thread too wide. `CORE_COUNT_TTL_NANOS` is
  100 ms, so a Ray worker whose affinity lands after process start is picked up.
  `python/batcher/_internal/hardware/cpu.py::available_cpu_count` is the control plane's
  same-shaped answer.
* `crates/bc-arrow/src/hardware.rs::operator_cores` -- a relational pipeline runs on
  every physical core plus a third of the SMT siblings among them, because a plan
  interleaves stalling work with bandwidth-bound work and neither end of the SMT trade is
  right for it.
* `python/batcher/_internal/hardware/memory.py::machine_memory_bytes` -- the same fold on
  the memory side: host RAM less any reserved hugepage pool, `memory.max`, `memory.high`,
  and the batch scheduler's grant, tightest first.
* `python/batcher/_internal/hardware/profile.py::HardwareProfile.fingerprint` -- the
  12-character digest, and `python/batcher/metadata/hardware_scope.py` for the rule it
  enforces: scope anything measured in machine units, never scope a statement about data.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 630

body = [
    band(20, 20, 940, 270, "WHAT THIS PROCESS REALLY GETS  ·  NOT WHAT THE HOST REPORTS", "grey"),
    card(44, 64, 250, 64, "Affinity mask", "the cpuset pin"),
    card(44, 138, 250, 64, "cgroup CFS quota", "the Kubernetes cpu limit"),
    card(44, 212, 250, 64, "Scheduler grant", "Slurm, PBS, LSF, SGE"),
    card(430, 120, 300, 120, "The effective machine", "cores, memory, devices"),
    note(580, 210, "the tightest bound wins, never below 1", anchor="middle"),
    arrow(294, 96, 430, 150, "grey"),
    label(362, 110, "min", anchor="middle"),
    arrow(294, 170, 430, 180, "grey"),
    label(362, 166, "min", anchor="middle"),
    arrow(294, 244, 430, 210, "grey"),
    label(362, 242, "min", anchor="middle"),
    note(936, 104, "available_parallelism honours", anchor="end"),
    note(936, 124, "the affinity mask, not the", anchor="end"),
    note(936, 144, "bandwidth quota: a pod capped", anchor="end"),
    note(936, 164, "at 15 cores on a 16-core node", anchor="end"),
    note(936, 184, "reports 16, and sizes every", anchor="end"),
    note(936, 204, "pool one thread too wide.", anchor="end"),
    note(936, 238, "Re-read every 100 ms, so a", anchor="end"),
    note(936, 258, "worker pinned after start is", anchor="end"),
    note(936, 278, "picked up.", anchor="end"),
    band(20, 330, 940, 190, "WHERE A HARDWARE FACT REACHES A PLAN DECISION", "blue"),
    card(44, 386, 280, 96, "Shard and pool width", "every physical core, plus a"),
    note(184, 452, "third of the SMT siblings", anchor="middle"),
    card(350, 386, 280, 96, "Memory budget, spill", "the cgroup ceiling, not the"),
    note(490, 452, "node's advertised RAM", anchor="middle"),
    card(656, 386, 280, 96, "Device placement", "inventory, MIG profiles,"),
    note(796, 452, "NVLink and PCIe islands", anchor="middle"),
    arrow(500, 240, 184, 382, "blue"),
    label(300, 300, "core count", anchor="end"),
    arrow(580, 240, 490, 378, "blue"),
    label(500, 300, "memory ceiling", anchor="end"),
    arrow(660, 240, 796, 378, "blue"),
    label(740, 300, "device inventory", anchor="start"),
    note(
        490,
        556,
        "The same record hashes to a 12-character fingerprint, and that fingerprint scopes every learned value measured in machine units --",
        anchor="middle",
    ),
    note(
        490,
        576,
        "nanoseconds, bytes, batch sizes -- so unlike machines never blend their coefficients. A statement about the data is never scoped:",
        anchor="middle",
    ),
    note(
        490,
        596,
        "a column has the same distinct count whatever machine reads it, and scoping those would fragment the statistics hardest to collect.",
        anchor="middle",
    ),
]

write("hardware_awareness", svg(W, H, "".join(body)))
print("wrote hardware_awareness.svg")
