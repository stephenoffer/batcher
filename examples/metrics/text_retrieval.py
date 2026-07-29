"""RAG groundedness: is the answer actually supported by the retrieved context?

These compare an answer column against the context column it was generated from, so they
run over an existing RAG output table with no extra model call. A falling
``fully_grounded_rate`` is the signal that retrieval regressed, not generation.

    python examples/metrics/text_retrieval.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    rag = bt.from_pydict(
        {
            "answer": [
                # Fully supported by its context.
                "The refund window is 30 days.",
                # Invents a number the context never mentions.
                "The refund window is 90 days and shipping is free worldwide.",
                # Quotes the context nearly verbatim.
                "Orders ship within two business days.",
            ],
            "context": [
                "Our refund window is 30 days from delivery.",
                "Our refund window is 30 days from delivery.",
                "Orders ship within two business days of payment.",
            ],
        }
    )

    grounding = rag.select(
        groundedness=bt.answer_groundedness("answer", "context"),
        fully_grounded=bt.fully_grounded_rate("answer", "context"),
        unsupported=bt.unsupported_token_rate("answer", "context"),
        utilization=bt.context_utilization("answer", "context"),
        citations=bt.citation_rate("answer"),
    ).to_pydict()

    print(grounding)

    for name, value in grounding.items():
        assert 0.0 <= value[0] <= 1.0, name
    # Groundedness and unsupported-token rate are complements of each other.
    assert abs(grounding["groundedness"][0] + grounding["unsupported"][0] - 1.0) < 1e-9
    assert grounding["citations"][0] == 0.0

    # Per row, the hallucinating answer scores worst.
    per_row = rag.select(
        g=bt.answer_groundedness("answer", "context").over(partition_by=["answer"])
    ).to_pydict()
    print(per_row)
    assert min(per_row["g"]) < max(per_row["g"])


if __name__ == "__main__":
    main()
