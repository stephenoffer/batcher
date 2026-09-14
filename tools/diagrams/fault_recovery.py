#!/usr/bin/env python3
"""Draw `fault_recovery.svg` -- classify first, then recompute, and what the recompute costs.

Source of truth, read before drawing:
  * `python/batcher/carbonite/resilience/classify.py` -- the one taxonomy the single-node
    executor and the distributed scheduler share: `is_retryable`, `must_move` (a failure local
    to a host must move or it walks the whole queue onto that host), and `results_untrusted`.
  * `python/batcher/dist/executors/ray_runtime/policies/_faults.py` -- the Ray-specific half.
    A `RayError` that is **not** a `RayTaskError` is the death of an actor, worker or node; a
    `RayTaskError` carries whatever the task raised and has to be classified.
    `_FATAL_RAY_ERROR_NAMES` (`RuntimeEnvSetupError`, `TaskCancelledError`, `GetTimeoutError`)
    are never absorbed as deaths. `check_results_trusted` refuses to retry past a fault that
    corrupts data already computed. `retry_budget` is job-wide.
  * `python/batcher/dist/executors/ray_runtime/policies/_barrier.py::map_barrier` -- the lost
    worker is recorded in `dead` and its source republished on a survivor under the **same**
    `src` id, so the reducers' `(stage, src, bucket)` tickets still resolve; the partition is a
    deterministic function of its durable descriptor, so the regenerated buckets are
    byte-identical.
  * `crates/bc-transport/src/ticket.rs` -- `epoch`, the re-execution fence.
  * `python/batcher/dist/shuffle_replication.py` and `config.distributed.shuffle_replication`
    (default 1, raised only by the `"spot"` profile).

Why this is a figure. The instinct is that recovery means "retry", and the code's first move is
not a retry at all -- it is a three-way classification in which two of the three answers are
*not* to retry. Absorbing a deterministic bug as a lost worker is how five TPC-H queries once
failed as `shuffle did not recover after 3 attempts` on a cluster where all four workers were
healthy, with the real traceback three frames down. A branching decision whose wrong branch
looks exactly like the right one is the case a diagram is for.

The bottom band is the part prose keeps burying: a recompute has three prices depending on
what was arranged beforehand, and the cheapest one runs before the worker is gone.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 664

body: list[str] = []

# ---- The event ---------------------------------------------------------------------------
body.append(band(20, 24, 940, 100, "A TASK DID NOT RETURN", "grey"))
body.append(card(310, 48, 360, 56, "a map or reduce task raised", "or its worker stopped answering"))

# ---- Classify, and note that two of three answers are not a retry -------------------------
body.append(arrow(490, 130, 490, 166))
body.append(label(504, 158, "classify before retrying", size=11.5))

body.append(band(20, 172, 940, 244, "THREE VERDICTS, AND ONLY ONE OF THEM RETRIES", "blue"))

verdicts = (
    (
        44, "lost data", "recompute it",
        (
            "A RayError that is not a RayTaskError:",
            "an actor, a worker or a node died.",
            "Also RetryableShuffleError (an",
            "unreachable peer) and ResourceError",
            "(a spill file on an ephemeral disk).",
        ),
    ),
    (
        356, "a deterministic bug", "re-raise it",
        (
            "A UDF exception, a bad cast, a schema",
            "mismatch, a broken runtime env.",
            "Every retry re-runs it, burns the",
            "job-wide budget, and reports a",
            "resource error for a Python bug.",
        ),
    ),
    (
        668, "results untrusted", "refuse to continue",
        (
            "An uncontained ECC fault: the device",
            "kept running and answered wrongly.",
            "Work already finished on it is as",
            "suspect as the task that failed, so",
            "retrying writes out corruption.",
        ),
    ),
)
for x, title, verdict, lines in verdicts:
    cx = x + 137
    body.append(card(x, 206, 274, 62, title, verdict))
    for i, line in enumerate(lines):
        body.append(note(cx, 292 + i * 17, line, anchor="middle"))

body.append(arrow(181, 392, 181, 440))
body.append(label(193, 424, "only this one", size=11.5))

# ---- What a recompute costs ---------------------------------------------------------------
body.append(band(20, 446, 940, 138, "WHAT A RECOMPUTE COSTS, AND WHAT WAS ARRANGED BEFOREHAND",
                 "amber"))

costs = (
    (44, "re-read and re-map", "the default", "Re-read the source partition and re-run",
     "the map. Usually the longest phase."),
    (356, "fetch a replica", "shuffle_replication > 1", "An off-node copy was acknowledged",
     "before the bucket was advertised."),
    (668, "migrate while alive", "on advance notice", "Spot metadata, SIGTERM or a Slurm",
     "deadline: one copy, not a re-read."),
)
for x, title, when, a, b in costs:
    cx = x + 137
    body.append(card(x, 472, 274, 58, title, when))
    body.append(note(cx, 550, a, anchor="middle"))
    body.append(note(cx, 567, b, anchor="middle"))

# ---- The hazard recovery itself introduces ------------------------------------------------
body.append(band(20, 596, 940, 56, "AND THE HAZARD RECOVERY INTRODUCES", "grey"))
body.append(label(44, 632, "a worker presumed dead may not be"))
body.append(note(340, 632, "so each round carries a higher epoch, and a reducer discards any "
                           "batch arriving under a stale one"))

write("fault_recovery", svg(W, H, "".join(body)))
print("wrote fault_recovery.svg")
