#!/usr/bin/env python3
"""Draw `gpu_shadow_verify.svg` -- the device tier's two oracles, and what each one can see.

Source of truth, read line by line before drawing:
  * `python/batcher/api/terminal/gpu_backend/verify.py` -- `enforce_schema_contract` (holds a
    device result against `LogicalPlan.available_schema`, the same static analysis
    `Dataset.schema` is answered from: names and types only, no rows, no second execution,
    no device) and `shadow_verify` (re-runs the plan on the CPU engine, `compare_results`
    checking `_schema_mismatch` **before** `_values_mismatch`, floats within `_FLOAT_RTOL`
    = 1e-9, and returning the **CPU** result on disagreement).
  * `python/batcher/api/terminal/gpu_backend/failure.py` -- `note_gpu_failure`, and why
    `DeviceDivergence` is matched there as a defect rather than passed to `note_suppressed`.
  * `.claude/rules/device-tier.md` -- the three shipped defects named in the bottom band, and
    the instruction not to gate the schema contract behind a flag.

Two things the picture is built to carry, both of which the prose states and a reader skims:

  * **The cheap oracle and the expensive one see different things.** The schema contract is
    free and unconditional and sees only types; the shadow re-run is expensive and off by
    default and is the only oracle for *values*. Drawing them as one pipeline with their costs
    attached is what stops "verification" reading as a single switch.
  * **Every defect this tier has shipped was a column type with correct values.** That is why
    the free check is the schema one and why schema is compared first inside the expensive one
    too. The bottom band names all three rather than asserting the pattern.

`gpu_tier_decision.svg` is the zoom level out: what reaches the device at all. This one starts
after a device result exists.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 660

body: list[str] = []

# ---- A result exists. Nothing has trusted it yet. --------------------------------------
body.append(band(20, 24, 940, 104, "THE DEVICE PRODUCED A RESULT", "grey"))
body.append(card(330, 50, 320, 58, "a pyarrow.Table", "from cuDF, not from the engine"))

# ---- The two oracles, in cost order ----------------------------------------------------
body.append(arrow(490, 134, 490, 168))
body.append(label(504, 158, "checked before it is returned", size=11.5))

body.append(band(20, 174, 940, 268, "TWO ORACLES, AND THEY COST VERY DIFFERENT THINGS", "blue"))

GATE_Y, GATE_H, GATE_W = 210, 72, 274
gates = (
    (
        44,
        "schema contract",
        "enforce_schema_contract",
        (
            "On for every device run.",
            "Reads a field list: no rows, no",
            "second execution, no device.",
            "Held against the engine's own",
            "available_schema. Sees TYPES.",
        ),
    ),
    (
        356,
        "shadow re-run",
        "shadow_verify",
        (
            "Off by default, behind",
            "distributed.gpu_shadow_verify.",
            "Re-runs the plan on the CPU",
            "engine: schema first, then",
            "values. The only oracle for VALUES.",
        ),
    ),
)
for x, title, where, lines in gates:
    cx = x + GATE_W / 2
    body.append(card(x, GATE_Y, GATE_W, GATE_H, title, where))
    for i, line in enumerate(lines):
        body.append(note(cx, 302 + i * 17, line, anchor="middle"))
    body.append(arrow(cx, 396, cx, 476, "amber"))
    body.append(label(cx + 10, 442, "differs", size=11.5))

body.append(arrow(320, GATE_Y + GATE_H / 2, 350, GATE_Y + GATE_H / 2))
body.append(label(335, GATE_Y - 8, "agrees", anchor="middle", size=11.5))
body.append(arrow(632, GATE_Y + GATE_H / 2, 662, GATE_Y + GATE_H / 2))
body.append(label(647, GATE_Y - 8, "agrees", anchor="middle", size=11.5))

body.append(card(668, GATE_Y, GATE_W, GATE_H, "the device result stands", "returned unchanged"))
body.append(note(805, 302, "A verified run differs from an", anchor="middle"))
body.append(note(805, 319, "unverified one only in cost.", anchor="middle"))
body.append(note(805, 353, "If the CPU oracle itself raises,", anchor="middle"))
body.append(note(805, 370, "that is reported as verifying", anchor="middle"))
body.append(note(805, 387, "nothing -- never as a pass.", anchor="middle"))

# ---- The verdict -----------------------------------------------------------------------
body.append(band(20, 462, 940, 106, "A DIFFERENCE IS A DEFECT, NEVER A DECLINE", "amber"))
body.append(card(44, 486, 420, 60, "the CPU engine answers", "the right columns, the right rows"))
body.append(note(516, 506, "Reported through note_gpu_failure, which logs at WARNING."))
body.append(note(516, 524, "note_suppressed is for declines, and this is not one: the tier's"))
body.append(note(516, 542, "contract is that a device changes where a plan runs, never what"))
body.append(note(516, 560, "it computes."))

# ---- Why the free check is the schema one ----------------------------------------------
body.append(
    band(20, 582, 940, 62, "EVERY DEFECT ON RECORD: A COLUMN TYPE, WITH CORRECT VALUES", "grey")
)
body.append(note(44, 626, "a DATE returning timestamp[ms] on a device, date32 under pandas"))
body.append(note(430, 626, "an integer abs widening to double"))
body.append(note(672, 626, "an empty cuDF string column arriving as null"))

write("gpu_shadow_verify", svg(W, H, "".join(body)))
print("wrote gpu_shadow_verify.svg")
