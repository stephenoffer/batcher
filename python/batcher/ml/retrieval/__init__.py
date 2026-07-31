"""Reranking a retrieved candidate set before it reaches a model.

A vector search returns the nearest `k` passages, and nearest is not the same as useful: on a
real corpus several of the `k` are the same passage republished, and the context window ends up
holding one fact four times.

`mmr` is the standard answer — a greedy selection that penalizes a candidate for resembling
what has already been chosen, so the context covers more of the answer space.
"""

from __future__ import annotations

from batcher.ml.retrieval.mmr import mmr_rerank_udf

__all__ = ["mmr_rerank_udf"]
