#!/usr/bin/env python3
"""Draw `join_order_search.svg` — how a join order gets chosen, and how hard it is looked for.

Source of truth, all under `python/batcher/kyber/`:

* `rules/joins/order.py::reorder_joins` — the one rule, registered as `join_reorder` in
  `Phase.JOIN_REORDER`, which runs **once** rather than to a fixpoint (`rule.py::Phase`:
  the cost-based phases "make a decision, they don't converge"). Inner joins only, and
  `if len(leaves) < 3: return None  # two-way: leave it to build-side selection`.
* `rules/joins/order_budget.py` — the budget, in evaluated join pairs:
  `pairs = units * _UNIT_SECONDS * _PLANNING_SHARE / _PAIR_SECONDS`, clamped to
  `[_MIN_PAIRS, _MAX_PAIRS]` = `[512, 200_000]`, with `_PLANNING_SHARE = 0.10` and
  `_PAIR_SECONDS = 1.5e-4`. `units` is the cost of the region *as written*.
* `rules/joins/order_search.py` — `_rebuild_dphyp`, the connected-subset DP that splits
  each subset into two connected halves, so the shapes it reaches are bushy;
  `_MAX_DP_LEAVES = 20` is a memory guard. `_rebuild_greedy` is the fallback: start from
  the smallest leaf, then repeatedly add the connected leaf with the lowest incremental
  cost, which is left-deep by construction. `_rebuild_dp`, the exhaustive subset DP, is
  the test oracle and `order.py` does not import it.
* `stats/estimator.py::_inner_join_rows` — Selinger containment,
  `|L| * |R| / max(d_L, d_R)` on non-null row counts, refined by Misra-Gries MCV skew,
  KLL quantile range overlap, PK-FK and unique-key caps, and a per-signature learned row
  count from the MetadataHub when one exists.
* `cost/model.py::_hash_join_cost` and `join_op_cost` — cpu from rows (`hash_build_row`
  2.0, `hash_probe_row` 1.0 times a cache-residency factor, `output_row` 0.5), mem from
  build bytes. An inner join is priced at the **cheaper** of its two build orientations,
  because SELECTION picks that one later.

The thing worth a diagram rather than a paragraph: **how hard to search is itself a cost
decision**, taken before the search starts and re-checked inside it, and the fallback is
not a failure mode but a second real search. Drawing the budget as a step in the flow is
the only way to make that read as deliberate rather than as a timeout.

Note for anyone keeping this in step: `config.py`'s `join_dp_max_tables` and
`greedy_max_tables` are *not* read by this search. The live limits are the two module
constants above.

Form: a serpentine flow (region, budget, DP, greedy) over a band holding the two things
every candidate is ranked by, because both searches rank with the same two.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, BLUE, FONT, MUTED, band, card, label, note, svg, write

W, H = 980, 630

MONO = "ui-monospace,SFMono-Regular,Menlo,monospace"


def mono(x: float, y: float, text: str) -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="middle" font-family="{MONO}" font-size="10.5" '
        f'fill="{BLUE}">{text}</text>'
    )


body = [
    # ---- Band 1: what is being searched, and the budget for searching it ---
    band(20, 20, 940, 150, "THE REGION, AND HOW HARD TO LOOK AT IT", "grey"),
    card(50, 62, 340, 84, "Join region", "connected inner joins, three leaves or more"),
    mono(220, 136, "fewer: build-side selection handles it"),
    card(570, 62, 340, 84, "Search budget", "512 to 200,000 evaluated pairs"),
    mono(740, 136, "and it is re-checked inside the loop"),
    f'<path d="M 398 104 L 560 104" fill="none" stroke="{BLUE}" stroke-width="2.4" '
    f'marker-end="url(#arB)"/>',
    label(479, 92, "cost the region as written", anchor="middle", size=11.5),
    note(479, 126, "10% of its estimated run time", anchor="middle"),
]

# ---- Band 2: the two searches ---------------------------------------------
body += [
    band(20, 186, 940, 230, "THE SEARCH, AND THE SEARCH IT FALLS BACK TO", "blue"),
    card(510, 236, 400, 96, "Connected-subset DP", "each subset split into two connected halves"),
    mono(710, 322, "bushy shapes, up to 20 leaves"),
    card(50, 236, 400, 96, "Greedy", "smallest leaf, then the cheapest next join"),
    mono(250, 322, "left-deep by construction"),

    # budget -> DP
    f'<path d="M 740 150 L 716 228" fill="none" stroke="{BLUE}" stroke-width="2.4" '
    f'marker-end="url(#arB)"/>',
    label(756, 182, "search inside", size=11.5),
    note(756, 200, "the budget"),

    # DP -> greedy, routed under both cards so it crosses no text
    f'<path d="M 540 340 V 368 H 420 V 338" fill="none" stroke="{AMBER_DEEP}" '
    f'stroke-width="2.4" stroke-dasharray="6 4" marker-end="url(#arA)"/>',
    f'<text x="480" y="392" text-anchor="middle" font-family="{FONT}" font-size="11.5" '
    f'font-weight="700" fill="{AMBER_DEEP}">budget spent, over 20 leaves, or a '
    f'disconnected graph</text>',
    note(710, 352, "a bushy tree", anchor="middle"),
    note(250, 352, "a left-deep tree", anchor="middle"),
]

# ---- Band 3: what ranks a candidate ---------------------------------------
body += [
    band(20, 432, 940, 150, "WHAT EVERY CANDIDATE IS RANKED BY, IN BOTH SEARCHES", "amber"),
    card(50, 478, 400, 80, "Cardinality", "|L| x |R| / max(ndv), then skew and range"),
    mono(250, 554, "HLL, Misra-Gries, KLL"),
    card(510, 478, 400, 80, "Cost", "build rows, probe rows, cache residency"),
    mono(710, 554, "priced at the cheaper build side"),
    f'<path d="M 180 476 L 180 338" fill="none" stroke="{BLUE}" stroke-width="2.2" '
    f'marker-end="url(#arB)"/>',
    f'<path d="M 780 476 L 780 338" fill="none" stroke="{BLUE}" stroke-width="2.2" '
    f'marker-end="url(#arB)"/>',
]

body += [
    f'<text x="490" y="606" text-anchor="middle" font-family="{FONT}" font-size="11.5" '
    f'fill="{MUTED}">Both searches build the same relation. They differ only in how much '
    f'of the space they read.</text>',
    f'<text x="490" y="624" text-anchor="middle" font-family="{FONT}" font-size="11.5" '
    f'fill="{MUTED}">An exhaustive subset DP exists beside them, as the test oracle, and '
    f'is never on the live path.</text>',
]

write("join_order_search", svg(W, H, "".join(body)))
print("wrote join_order_search.svg")
