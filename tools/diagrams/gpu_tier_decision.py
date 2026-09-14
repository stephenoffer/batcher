#!/usr/bin/env python3
"""Draw `gpu_tier_decision.svg` -- the device tier as a translator that must classify every tag.

Source of truth, read before drawing:
  * `.claude/rules/device-tier.md` -- the tier's contract, and the reason it is the one tier
    that cannot share the Rust `Expr`.
  * `python/batcher/core/gpu_plan/ops.py` -- `SUPPORTED_OPS` (filter, project, aggregate, sort,
    distinct, limit, window, unnest, unpivot, row_id) and `DECLINED_OPS`, a dict of tag to
    *reason*. `scan` / `hash_join` / `union` are declined as chain steps because `eligibility`
    matches them as plan shapes instead.
  * `python/batcher/core/gpu_plan/exprs.py` -- `_HANDLERS` against `DECLINED_EXPRS`, likewise
    keyed by reason.
  * `python/batcher/core/gpu_plan/eligibility.py` -- a plan reaches the device only as a chain
    over one scan, an equi-join of two chains, or a union of chains, with every node
    translatable.
  * `tests/unit/test_gpu_vocabulary_contract.py` -- the classification is exhaustive against
    `plan.ir_tags.{Op, ExprTag}` by test, in both directions: an unclassified tag fails, and so
    does a handler keyed on a tag the engine no longer has.
  * `python/batcher/api/terminal/gpu_backend/route.py` -- every decline returns `None` and the
    caller uses the CPU engine, which is what makes `backend="gpu"` safe to ask for.

Why this is a figure rather than a paragraph. The tier's safety rests on a *partition* -- every
tag is in exactly one of two sets, and a third state (absent) is what the contract test exists
to make impossible. A partition with a forbidden middle is a shape, and prose states it as a
list of features, which is the reading the page has to fight. The left band carries the reason
the machinery is needed at all: without it, invariant #6 has nothing mechanical behind it here.

Deliberately stops at the tier's own gate. What happens to a result the device *does* produce
is `gpu_shadow_verify.svg`, one zoom level in.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 616

body: list[str] = []

# ---- Why this tier needs machinery no other tier needs --------------------------------
body.append(band(20, 24, 940, 148, "WHY THIS TIER IS DIFFERENT", "grey"))

body.append(card(44, 58, 396, 66, "every other tier", "consumes the same Rust bc_expr::Expr"))
body.append(note(242, 144, "one definition of what a shape means, so it cannot drift",
                 anchor="middle"))

body.append(card(540, 58, 396, 66, "the device tier", "cuDF has no Rust binding"))
body.append(note(738, 144, "a second statement of the semantics, in another language",
                 anchor="middle"))

body.append(label(490, 96, "vs", anchor="middle"))

# ---- The partition: translated, or declined with a reason -----------------------------
body.append(arrow(738, 158, 738, 196))
body.append(label(752, 186, "so every tag is classified", size=11.5))

body.append(band(20, 202, 940, 202, "EVERY IR TAG IS IN EXACTLY ONE OF TWO SETS", "blue"))

body.append(card(44, 238, 420, 72, "translated", "SUPPORTED_OPS  /  exprs._HANDLERS"))
body.append(note(254, 332, "filter, project, aggregate, sort, distinct, limit,", anchor="middle"))
body.append(note(254, 349, "window, unnest, unpivot, row_id -- and the", anchor="middle"))
body.append(note(254, 366, "expression vocabularies keyed beside them", anchor="middle"))

body.append(card(516, 238, 420, 72, "declined, with a reason", "DECLINED_OPS  /  DECLINED_EXPRS"))
body.append(note(726, 332, "asof_join, range_join, sample: not translated.", anchor="middle"))
body.append(note(726, 349, "image / audio / geo: Rust kernels with no", anchor="middle"))
body.append(note(726, 366, "dataframe equivalent to translate onto", anchor="middle"))

body.append(label(490, 282, "no third state", anchor="middle", size=11.5))
body.append(note(490, 390, "a tag in neither set fails test_gpu_vocabulary_contract -- so a new "
                           "operator is a decision, not an oversight", anchor="middle"))

# ---- What that buys at run time --------------------------------------------------------
body.append(arrow(254, 404, 254, 442))
body.append(label(266, 432, "all nodes translate", size=11.5))
body.append(arrow(726, 404, 726, 442))
body.append(label(738, 432, "any node declines", size=11.5))

body.append(band(20, 448, 940, 148, "THE PLAN GOES ONE WAY OR THE OTHER, WHOLE", "amber"))
body.append(card(44, 474, 420, 66, "run on the device", "cuDF, one shard per device"))
body.append(card(516, 474, 420, 66, "run on the CPU engine", "the same rows, more slowly"))

body.append(note(254, 562, "eligible only as a chain over a scan, a join of two", anchor="middle"))
body.append(note(254, 579, "chains, or a union of chains", anchor="middle"))
body.append(note(726, 562, "a decline costs time; an approximation would cost a", anchor="middle"))
body.append(note(726, 579, "wrong answer, so backend=\"gpu\" is always safe to ask for",
                 anchor="middle"))

write("gpu_tier_decision", svg(W, H, "".join(body)))
print("wrote gpu_tier_decision.svg")
