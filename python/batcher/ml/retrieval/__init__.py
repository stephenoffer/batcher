"""Reranking a retrieved candidate set before it reaches a model.

A vector search returns the nearest `k` passages, and nearest is not the same as useful: on a
real corpus several of the `k` are the same passage republished, and the context window ends up
holding one fact four times.

Two rerankers, answering different halves of that. `rerank` is the accuracy half: a
cross-encoder reads the query and each candidate *together*, which a vector search cannot do
because it embedded the passage before it knew the query. `mmr` is the redundancy half: a
greedy selection that penalizes a candidate for resembling what has already been chosen, using
vectors you already have and no model at all.

They compose in that order — narrow 100 candidates to 20 by relevance, then 20 to 5 by
diversity.
"""

from __future__ import annotations

from batcher.ml.retrieval.mmr import mmr_rerank_udf
from batcher.ml.retrieval.rerank import CrossEncoderScorer, cross_encoder_rerank_udf

__all__ = ["CrossEncoderScorer", "cross_encoder_rerank_udf", "mmr_rerank_udf"]
