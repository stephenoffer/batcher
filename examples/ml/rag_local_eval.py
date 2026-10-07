"""A whole RAG pipeline run locally, then evaluated: chunk, embed, retrieve, generate, score.

Every model in a RAG pipeline is a plug-in point, so a stub encoder and a stub engine stand
in for the real ones and the pipeline runs with no GPU and no service. What it shows is the
data path around the models and the step most pipelines skip: measuring retrieval with
``ds.ml.recall_at_k`` against labelled relevant chunks before trusting any answer.

Swap the stubs for ``ds.ml.embed("<model id>", column=...)`` and
``batcher.ml.vllm_engine("<model id>", chat=True)`` and nothing else changes.

    python examples/ml/rag_local_eval.py
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import batcher as bt
from batcher import col

#: The toy embedding space: one dimension per topic word.
TOPICS = ["cat", "train", "soup"]


def toy_encoder(column: str):
    """A stand-in embedding model: a batch in, the batch plus an ``embedding`` column out."""

    def encode(batch: pa.RecordBatch) -> pa.RecordBatch:
        text = pc.utf8_lower(batch.column(column))
        hits = [pc.cast(pc.match_substring(text, t), pa.float64()) for t in TOPICS]
        vectors = np.stack([h.to_numpy(zero_copy_only=False) for h in hits], axis=1)
        embedding = pa.array(vectors.tolist(), type=pa.list_(pa.float64()))
        return pa.RecordBatch.from_arrays(
            [*batch.columns, embedding], names=[*batch.schema.names, "embedding"]
        )

    return encode


def stub_engine():
    """A stand-in LLM: answers with the first sentence of the context it was given."""
    return lambda prompts: [p.split("Context: ")[1].split(" | ")[0] for p in prompts]


def main() -> None:
    docs = bt.from_pydict(
        {
            "url": ["http://pets", "http://rail", "http://food"],
            "text": [
                "<p>A cat sleeps all day.</p><p>Every cat purrs.</p>",
                "<p>The train leaves at noon.</p><p>A train runs on rails.</p>",
                "<p>Soup is best hot.</p><p>Tomato soup needs salt.</p>",
            ],
        }
    )

    # Ingest: clean, chunk (one sentence per chunk here), keep provenance, embed.
    chunks = (
        docs.select("url", text=col("text").str.strip_html())
        .with_columns(chunk=col("text").str.split(". "))
        .explode("chunk")
        .with_row_index("chunk_id")
        .select("chunk_id", "url", "chunk")
    )
    corpus = chunks.ml.embed(toy_encoder("chunk"), output_columns=[*chunks.columns, "embedding"])

    # Query time: embed the questions with the same encoder and retrieve the top 2 each.
    questions = bt.from_pydict(
        {"qid": [1, 2], "question": ["Does a cat purr?", "When does the train leave?"]}
    )
    queries = questions.ml.embed(
        toy_encoder("question"), output_columns=[*questions.columns, "embedding"]
    )
    hits = corpus.ml.batched_nearest_neighbors(
        queries, query_key="qid", query_column="embedding", corpus_key="chunk_id", k=2
    )

    # Evaluate retrieval against labelled relevant chunks before reading any answer.
    relevant = bt.from_pydict({"qid": [1, 1, 2], "chunk_id": [0, 1, 2]})
    recall = hits.ml.recall_at_k(relevant, query_key="qid", corpus_key="chunk_id")
    print(f"recall@2 = {recall:.3f}")
    assert recall == 1.0, recall

    # Assemble each question's context and generate a grounded answer.
    contexts = (
        hits.join(chunks, on="chunk_id")
        .group_by("qid")
        .agg(context=col("chunk").array_agg())
        .with_columns(context=col("context").list.join(" | "))
    )
    answers = (
        questions.join(contexts, on="qid")
        .ml.generate(
            stub_engine,
            prompt_column="question",
            template="Question: {question}\nContext: {context}",
            output_column="answer",
        )
        .with_columns(grounded=bt.answer_groundedness("answer", "context"))
        .sort("qid")
        .to_pydict()
    )
    for question, answer in zip(answers["question"], answers["answer"], strict=True):
        print(f"{question} -> {answer}")
    assert all(score == 1.0 for score in answers["grounded"]), answers["grounded"]
    assert "cat" in answers["answer"][0].lower() and "train" in answers["answer"][1].lower()


if __name__ == "__main__":
    main()
