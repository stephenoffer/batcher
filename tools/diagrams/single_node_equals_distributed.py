#!/usr/bin/env python3
"""Draw `single_node_equals_distributed.svg` -- what the equality promises, and its three
stated exceptions.

Source of truth: `.claude/rules/python-control-plane.md`, whose wording this figure is drawn
to clause by clause, plus the code each clause names:
  * `crates/bc-runtime/src/agg/{accum,var,stats}.rs` -- Neumaier compensation and Chan's
    parallel Welford formula, which **bound** the reassociation error rather than removing it.
  * `crates/bc-interp/src/ops/reshape.rs::sample_n_batches` -- breaks its hash ties by row
    *content*, which is why the window tie-break is a decision available to anyone who wants
    it rather than a limit of the architecture.

Two things this diagram is deliberately careful about, because the flattering version of each
is the one a reader expects:

  * **It does not say bit-identical.** `combine` is associative in exact arithmetic and IEEE
    addition is not, so a float reduction is identical *up to reassociation* and nothing can
    make it more than that while the partition count is free. The rules file records that
    stating it as bit-identity was worse than useless -- `assert_same` tolerates float
    rounding, so nothing ever checked the stronger claim.
  * **It does not present the exceptions as defects or as a licence.** All three are places
    where the *query* does not determine an answer. What still binds is drawn inside the same
    band, because a divergence outside these three is a defect however plausible its rows look.

`mergeable_algebra.svg` already draws *why* one operator serves several execution modes; the
top band here is the one-line restatement that makes this figure stand alone, not a second
copy of it. The subject here is the guarantee, not the algebra.

Layout: the promise on top, then what is exact, then the carve-outs with their own bounds
underneath, so the two halves are never read apart.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 690

body: list[str] = []

# ---- Why one implementation serves three deployments ---------------------------------
body.append(band(20, 24, 940, 196, "ONE OPERATOR, THREE DEPLOYMENTS", "blue"))

deployments = (
    (76, "one core", "bc-interp::execute"),
    (390, "many cores", "bc-interp::par"),
    (704, "many machines", "bc-interp::dist"),
)
for x, title, sub in deployments:
    body.append(card(x, 56, 200, 60, title, sub))
    body.append(arrow(x + 100, 118, 490, 154))

body.append(label(300, 140, "the same three functions", anchor="middle", size=11.5))
body.append(card(340, 158, 300, 56, "partial / combine / finalize", "written once"))
body.append(
    note(
        490,
        206,
        "combine is associative and commutative, so arrival order cannot "
        "change which rows come back",
        anchor="middle",
    )
)

# ---- What is exact --------------------------------------------------------------------
body.append(arrow(490, 226, 490, 262))
body.append(label(504, 252, "therefore", size=11.5))

body.append(band(20, 268, 940, 96, "EXACT, HOWEVER MANY PARTITIONS", "grey"))
for x, text in (
    (166, "the multiset of rows"),
    (490, "every column name"),
    (814, "every column type"),
):
    body.append(label(x, 314, text, anchor="middle"))
body.append(
    note(
        490,
        342,
        "no tolerance and no qualification -- these three have no exceptions",
        anchor="middle",
    )
)

# ---- The three stated exceptions ------------------------------------------------------
body.append(arrow(490, 370, 490, 406))
body.append(label(504, 396, "except where the query fixes no answer", size=11.5))

body.append(
    band(20, 412, 940, 238, "THREE PLACES THE QUERY ITSELF DOES NOT DETERMINE AN ANSWER", "amber")
)

cases = (
    (
        44,
        "float reassociation",
        "a SUM over partitions",
        (
            "IEEE addition is not associative.",
            "Neumaier and Chan bound the error to",
            "near the last bits; nothing removes it.",
        ),
        "the row count and every type",
    ),
    (
        340,
        "an open window tie",
        "row_number over tied rows",
        (
            "SQL leaves the tie undefined, so the",
            "two paths break it by different physical",
            "orders. rank / dense_rank are unaffected.",
        ),
        "the partitions, and the ranks 1..n",
    ),
    (
        636,
        "LIMIT with no order",
        "limit over a group_by",
        (
            "A hash walk is not a query property, so",
            "LIMIT n keeps some n groups.",
            "sort(...).limit(n) does not diverge.",
        ),
        "n rows, each one of the full result",
    ),
)
for x, title, sub, lines, binds in cases:
    cx = x + 150
    body.append(card(x, 446, 300, 62, title, sub))
    for i, line in enumerate(lines):
        body.append(note(cx, 532 + i * 17, line, anchor="middle"))
    body.append(label(cx, 596, "still exact here:", anchor="middle", size=11.5))
    body.append(note(cx, 614, binds, anchor="middle"))

body.append(
    note(
        490,
        672,
        "A divergence that is not one of these three is a defect, however plausible its rows look.",
        anchor="middle",
    )
)

write("single_node_equals_distributed", svg(W, H, "".join(body)))
print("wrote single_node_equals_distributed.svg")
