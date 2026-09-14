#!/usr/bin/env python3
"""Draw `cost_model_inputs.svg` -- what the cost model eats, and what it emits.

The non-obvious half is the output, not the input: `Cost` has four axes and the scalar
that ranks plans is built from **three** of them. `mem` is a peak rather than a sum, so
it is a max along the tree and never enters `total()`; it gates feasibility instead. A
reader who assumes four axes in, one weighted sum out, has the model wrong in the one
place it matters.

Source, and what to keep in step:

* `python/batcher/kyber/cost/model.py` -- the `Cost` dataclass, `total()`, and the
  per-operator closed forms. `total()` is `w.cpu*cpu + w.io*io + w.net*net`.
* `python/batcher/config/config.py::CostWeights` -- the default weights drawn here:
  cpu 1.0, io 1.0, net 2.0 (a shuffled byte costs twice a local one). They are
  fabric-adjusted per node before use, and swapped per query to honour an SLA target.
* `python/batcher/kyber/cost/terms.py` -- the machine-shaped multipliers: the L3 cache
  size, the memory budget, the spill device factor, the external-merge fan-in.
* `python/batcher/kyber/calibration.py` -- the coefficients ship as constants and are
  fitted from Core's measured `op_stats`, shrunk toward the default in proportion to how
  little evidence there is.
* `python/batcher/kyber/expr_cost/` -- an expression has a cost of its own, folded into
  the per-row terms.

The consumer list at the bottom is `model.py`'s own: join order, join strategy, whether
to spill.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 590

body = [
    band(20, 40, 940, 150, "WHAT THE COST MODEL CONSUMES", "blue"),
    card(40, 86, 190, 86, "Estimated rows", "per node, from the estimator"),
    card(278, 86, 190, 86, "Row width", "type-exact, not a flat 64"),
    card(516, 86, 190, 86, "Machine terms", "L3, memory budget, device"),
    card(754, 86, 190, 86, "Coefficients", "constants, then calibrated"),
    card(340, 250, 300, 100, "CostModel.cost(node)", "one fold over the plan tree"),
    arrow(135, 172, 355, 246, "blue"),
    label(200, 212, "rows", anchor="end"),
    arrow(373, 172, 430, 246, "blue"),
    label(350, 212, "bytes per row", anchor="end"),
    arrow(611, 172, 540, 246, "blue"),
    label(620, 212, "the machine", anchor="start"),
    arrow(849, 172, 620, 246, "blue"),
    label(770, 212, "work units", anchor="start"),
    band(20, 390, 940, 140, "WHAT IT EMITS  ·  FOUR AXES, THREE IN THE SCALAR", "amber"),
    card(100, 420, 340, 88, "One comparable number", "1.0 x cpu  +  1.0 x io  +  2.0 x net"),
    card(540, 420, 340, 88, "Peak working set", "a max along the tree, never summed"),
    arrow(420, 350, 300, 416, "amber"),
    label(300, 382, "cpu, io, net", anchor="end"),
    arrow(560, 350, 700, 416, "amber"),
    label(700, 382, "mem", anchor="start"),
    note(270, 552, "ranks the alternatives:", anchor="middle"),
    note(270, 572, "join order, join strategy, whether to spill", anchor="middle"),
    note(710, 552, "gates feasibility, not throughput --", anchor="middle"),
    note(710, 572, "a peak is not a quantity you can add up", anchor="middle"),
]

write("cost_model_inputs", svg(W, H, "".join(body)))
print("wrote cost_model_inputs.svg")
