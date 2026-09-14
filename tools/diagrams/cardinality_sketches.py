#!/usr/bin/env python3
"""Draw `cardinality_sketches.svg` -- the sketches behind an estimate, split by how
they merge.

The whole value of this figure is the split, because the prose around it says the
sketches are "all `Mergeable` with a fixed seed, so a sketch built on partition 3 of
worker 7 merges with one built anywhere else, in any order" -- true of the `Mergeable`
contract, and it hides a distinction a reader needs. Three of them reach a
**bit-identical** state whatever the merge order; the quantile sketches do not, and
cannot, because their merge compacts or re-clusters and that is order-sensitive by
construction rather than by defect.

Sources, all of which must be kept in step with this drawing:

* `crates/bc-sketches/tests/merge_order.rs` -- pins both halves. The exact half asserts
  equality for HyperLogLog (register-wise max), CountMin (cell-wise sum), Bloom (bitwise
  OR) and `ColumnStats`' min/max/count/ndv scalars. The approximate half asserts only
  that two merge orders agree within the sketch's own rank error, which is the property
  a caller is actually entitled to.
* The worst-case rank gaps quoted here are that test's own recorded measurements: 0.0097
  for KLL at k=200 (against the ~2/k the algorithm promises) and 0.0050 for TDigest at
  compression 100.
* `crates/bc-sketches/src/hll.rs` -- precision 14 is about 0.81% relative error in 16 KB.
* `crates/bc-sketches/src/kll.rs` -- rank error about 1/k; k=200 is roughly 1%.

`FrequentItems` is deliberately absent. `.claude/rules/rust-engine.md` groups it with the
order-sensitive sketches, `frequent.rs` argues its algorithm is order-independent, and
`merge_order.rs` pins neither. Drawing either answer would assert something no test checks.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 520

body = [
    # ---- merges to one state -----------------------------------------------
    band(20, 20, 556, 300, "SAME STATE IN ANY MERGE ORDER", "blue"),
    card(44, 72, 508, 66, "HyperLogLog", "distinct count -- folds by register-wise max"),
    card(44, 152, 508, 66, "Count-Min", "how often is THIS key -- folds by cell-wise sum"),
    card(44, 232, 508, 66, "Bloom", "membership, data skipping -- folds by bitwise OR"),
    note(298, 312, "ColumnStats' min, max, count and ndv fold the same way.", anchor="middle"),

    # ---- merges to a close enough state ------------------------------------
    band(604, 20, 356, 300, "WITHIN ITS OWN RANK ERROR", "amber"),
    card(628, 72, 308, 66, "KLL", "quantiles -- its merge compacts"),
    card(628, 152, 308, 66, "TDigest", "quantiles -- re-clusters centroids"),
    note(782, 246, "Two merge orders give two answers.", anchor="middle"),
    note(782, 266, "Worst measured gap in rank: 0.0097 at", anchor="middle"),
    note(782, 286, "k=200, and 0.0050 at compression 100.", anchor="middle"),
    note(782, 312, "That is the error the sketch promises.", anchor="middle"),

    # ---- what they are for --------------------------------------------------
    card(300, 386, 380, 88, "The cardinality estimate", "row counts and per-column stats"),
    arrow(298, 320, 400, 382, "blue"),
    label(292, 364, "exact counts", anchor="end"),
    arrow(782, 320, 600, 382, "amber"),
    label(700, 356, "quantiles, selectivity", anchor="start"),

    note(490, 502, "Never assert that a quantile sketch merges to an identical state, and never set out to fix the fact that it does not.", anchor="middle"),
]

write("cardinality_sketches", svg(W, H, "".join(body)))
print("wrote cardinality_sketches.svg")
