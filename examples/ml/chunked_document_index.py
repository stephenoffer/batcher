"""Building a chunk-level index from document-level text.

Retrieval works on chunks and reporting works on documents, so the index carries both keys.
Keeping the document id on every chunk is what lets you deduplicate results back to one hit
per document after the search.

    python examples/ml/chunked_document_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    documents = tpch("orders").select("o_orderkey", "o_comment").head(2_000)

    chunked = (
        documents.select(
            document=col("o_orderkey"), chunks=col("o_comment").str.chunk(30, overlap=5)
        )
        .explode("chunks")
        .select("document", text=col("chunks"))
        .with_row_index(name="chunk_id")
    )
    print("chunks:", chunked.count())
    assert chunked.count() >= documents.count()

    # Every chunk knows its document, and every document has at least one chunk.
    assert chunked.n_unique("document") == documents.count()
    assert chunked.n_unique("chunk_id") == chunked.count()

    per_document = chunked.group_by("document").agg(chunks=bt.count())
    assert min(per_document.to_pydict()["chunks"]) >= 1

    # Search at chunk level.
    hits = chunked.filter(col("text").str.contains("final"))
    print("chunk hits:", hits.count())
    assert hits.count() > 0

    # Then collapse to one hit per document, keeping the best chunk.
    best = (
        hits.with_columns(
            rank=bt.row_number().over(partition_by=["document"], order_by=["chunk_id"])
        )
        .filter(col("rank") == 1)
        .select("document", "chunk_id", "text")
    )
    print("document hits:", best.count())
    assert best.count() <= hits.count()
    assert best.n_unique("document") == best.count()

    # Every returned document really contains the term somewhere.
    documents_hit = set(best.to_pydict()["document"])
    sample = list(documents_hit)[:5]
    for key in sample:
        assert (
            documents.filter(col("o_orderkey") == key)
            .filter(col("o_comment").str.contains("final"))
            .count()
            == 1
        )


if __name__ == "__main__":
    main()
