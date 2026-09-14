#!/usr/bin/env python3
"""Draw `vector_index_search.svg` - the two ways to answer a nearest-neighbour query.

Source of truth: `python/batcher/api/dataset/ml.py::nearest_neighbors` (exact),
`python/batcher/ml/embed.py` (`build_vector_index`, `vector_search`) and
`python/batcher/io/formats/structured/lance.py` (the Lance delegation). Distances are
Rust kernels in `crates/bc-expr/src/eval/list.rs`; `sort` with a `limit` is mergeable per
`python/batcher/plan/distribution/mergeable.py`.

The picture exists for one inversion that reads backwards until you see it: the *exact*
path is the one that shards, because a global top-k is the top-k of the shards' top-ks,
while the index path is a single driver call into Lance. It also states what the repo
does and does not implement: `nprobes` and `refine_factor` are passed through verbatim
and default to `None`, so Lance chooses, and nothing here measures the recall given up.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 750

body = [
    band(20, 20, 940, 96, "EMBED THE COLUMN ONCE", "grey"),
    card(
        150,
        44,
        680,
        56,
        "ds.ml.embed(model, column='text', output_type='fixed_size_list')",
        "one fixed_size_list&lt;float32&gt; vector per row, computed in the engine",
    ),
    arrow(250, 104, 250, 196, "blue"),
    label(264, 140, "no build step", size=12),
    arrow(730, 104, 730, 196, "amber"),
    label(744, 140, "write.lance first", size=12),
    # Exact.
    band(20, 152, 460, 420, "EXACT  -  AN ENGINE OPERATOR", "blue"),
    card(60, 196, 380, 74, "ds.ml.nearest_neighbors(q, k)", "cosine, l2, l1, hamming, dot"),
    arrow(250, 270, 250, 308, "blue"),
    label(264, 294, "lowers to", size=12),
    card(60, 308, 380, 74, "distance, sort, limit k", "Rust kernels, one scan"),
    arrow(250, 382, 250, 420, "blue"),
    label(264, 406, "shards", size=12),
    card(60, 420, 380, 74, "exact top-k", "the top-k of the shards' top-ks"),
    note(250, 518, "Mergeable, so one core or a hundred machines", anchor="middle"),
    note(250, 536, "run it unchanged. Costs a full scan per query.", anchor="middle"),
    note(250, 558, "The recommended path to a few million rows.", anchor="middle"),
    # Approximate.
    band(500, 152, 460, 420, "APPROXIMATE  -  AN INDEX OVER LANCE", "amber"),
    card(540, 196, 380, 74, "build_vector_index(uri)", "index_type='IVF_PQ', metric='L2'"),
    arrow(730, 270, 730, 308, "amber"),
    label(744, 294, "delegates to Lance", size=12),
    card(540, 308, 380, 74, "vector_search(uri, q, k)", "nprobes, refine_factor"),
    arrow(730, 382, 730, 420, "amber"),
    label(744, 406, "one driver call", size=12),
    card(540, 420, 380, 74, "approximate top-k", "k rows and a _distance column"),
    note(730, 518, "Both knobs default to None, so Lance decides.", anchor="middle"),
    note(730, 536, "Raising either probes more of the index:", anchor="middle"),
    note(730, 558, "you buy recall back with time.", anchor="middle"),
    # What approximate costs, stated plainly.
    band(20, 604, 940, 126, "WHAT APPROXIMATE ACTUALLY COSTS YOU", "grey"),
    note(
        490,
        648,
        "The exact path is the one that distributes. The index path runs on the driver, against one Lance dataset, unsharded.",
        anchor="middle",
    ),
    note(
        490,
        670,
        "Nothing here measures the recall you gave up: recall_at_k scores a set you hand it, and no code wires it to the index.",
        anchor="middle",
    ),
    note(
        490,
        700,
        "ds.ml.embed defaults to output_type='tensor', which Lance cannot index. build_vector_index raises rather than mis-index it.",
        anchor="middle",
    ),
]

write("vector_index_search", svg(W, H, "".join(body)))
print("wrote vector_index_search.svg")
