#!/usr/bin/env python3
"""Draw `rag_ingest_query.svg`: the ingest batch job and the query-time path of a RAG system.

Source of truth: `docs/ml/retrieval/rag.md`. Ingest runs load, clean
(`.str.strip_html()`), chunk (`.str.chunk` then `explode`), dedupe (`distinct` then
`ml.drop_near_duplicates`), embed (`ml.embed`) and index (`write.lance` then
`build_vector_index`). Query time embeds the question with the same model, retrieves
(a brute-force `top_k` or `vector_search`), reranks (a cross-encoder narrows 100 to 20
by relevance, then MMR narrows 20 to 5 by diversity), assembles the context with
`array_agg`, and generates with `ml.generate`. The page says to keep the two separate,
and to carry `url` and `chunk_id` through so the answer can cite its sources.

Layout: ingest on the top row, query time on the bottom row, and one dashed curve for
the index the query reads, which is the only thing the two halves share.
"""

from __future__ import annotations

from _authoring import arrow, band, curve, label, note, svg, tint, write

W, H = 980, 562

XS = (36, 280, 524, 768)
CW = 172
TOP_Y, BOT_Y, CH = 74, 318, 80


def row(y: float, cards: list[tuple[str, str]], kind: str, words: tuple[str, ...]) -> list[str]:
    """One row of four tinted cards joined by labeled arrows."""
    out = [tint(x, y, CW, CH, t, s, kind=kind) for x, (t, s) in zip(XS, cards, strict=True)]
    mid = y + CH / 2
    for i, word in enumerate(words):
        x1, x2 = XS[i] + CW + 6, XS[i + 1] - 6
        out += [
            arrow(x1, mid, x2, mid, "blue" if kind == "blue" else "amber"),
            label((x1 + x2) / 2, mid - 10, word, anchor="middle", size=11.5),
        ]
    return out


body = [
    band(16, 20, 948, 180, "INGEST  ·  A BATCH JOB OVER THE CORPUS", "blue"),
    *row(
        TOP_Y,
        [
            ("Load, clean", "str.strip_html()"),
            ("Chunk", "str.chunk, explode"),
            ("Dedupe", "exact, then near"),
            ("Embed, index", "ml.embed, Lance"),
        ],
        "blue",
        ("text", "chunks", "unique"),
    ),
    note(
        490,
        184,
        "Every stage is an engine operator, so the job streams and distributes.",
        anchor="middle",
    ),
    band(16, 264, 948, 214, "QUERY TIME  ·  PER QUESTION", "amber"),
    *row(
        BOT_Y,
        [
            ("Embed question", "the same model"),
            ("Retrieve", "top_k or vector_search"),
            ("Rerank", "100 -> 20 -> 5"),
            ("Generate", "array_agg, ml.generate"),
        ],
        "amber",
        ("vector", "top 100", "top 5"),
    ),
    # The index is the one thing the halves share.
    curve(854, TOP_Y + CH + 6, 640, 262, 380, BOT_Y - 8, "blue"),
    label(560, 234, "the chunk vectors", anchor="middle"),
    note(610, 426, "A cross-encoder keeps 20 by relevance,", anchor="middle"),
    note(610, 444, "then MMR keeps 5 by diversity.", anchor="middle"),
    note(366, 426, "Brute force on a small corpus,", anchor="middle"),
    note(366, 444, "the Lance index at scale.", anchor="middle"),
    band(16, 500, 948, 44, "", "grey"),
    note(
        490,
        527,
        "Carry url and chunk_id from ingest through retrieval, so the answer can cite its sources.",
        anchor="middle",
    ),
]

write("rag_ingest_query", svg(W, H, "".join(body)))
print("wrote rag_ingest_query.svg")
