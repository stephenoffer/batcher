"""One RAG pipeline, start to finish, over every piece added for AI workloads.

Each function has its own unit tests. What those cannot show is whether the pieces *compose* —
whether the column a chunker produces is the one an embedder expects, whether a reranker's
output still carries the ids a citation check needs, whether a judged column can be aggregated
beside a lexical one. That is where a surface spread across eight modules actually breaks, and
it breaks the first time someone tries to use two of them together.

So this is deliberately one pipeline rather than a suite of assertions: ingest, curate,
decontaminate, chunk, retrieve, rerank, assemble a prompt, generate, judge, and score. The
models are stubs, because what is under test is the plumbing between the stages, not a model.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.ml import (
    QualityThresholds,
    contamination_rate,
    decontaminate,
    llm_score_udf,
    mmr_rerank_udf,
    quality_filter,
    quality_flags,
)

pytestmark = pytest.mark.integration


_DOCUMENTS = [
    # Prose that should survive every filter.
    "Rayleigh scattering explains the colour of the sky. Shorter wavelengths scatter more.",
    "The Eiffel tower stands in Paris and was completed in eighteen eighty nine.",
    # A near-duplicate of the first, for the reranker to collapse.
    "Rayleigh scattering explains why the sky looks blue. Short wavelengths scatter most.",
    # Junk the quality filter should remove.
    "1234 5678 9012 3456",
    "buy now!!!",
    # A verbatim copy of an evaluation question, for decontamination. It has to pass the
    # quality filter to reach that stage, so it is punctuated like a real document.
    "What is the capital of France? It is Paris, the largest city in the country.",
]

_EVAL_QUESTIONS = ["what is the capital of france"]


def _stub_embedding(text: str) -> list[float]:
    """A two-dimensional embedding that puts the sky documents near each other.

    Deterministic and legible on purpose: a real encoder would make the reranking assertion
    depend on a model's weights, which is not what this test is about.
    """
    sky = float("scatter" in text or "sky" in text)
    paris = float("paris" in text.lower() or "eiffel" in text.lower())
    return [sky, paris]


def test_a_rag_pipeline_composes_end_to_end():
    corpus = bt.from_pydict({"doc_id": list(range(len(_DOCUMENTS))), "text": _DOCUMENTS})
    evals = bt.from_pydict({"question": _EVAL_QUESTIONS})

    # --- curate ---------------------------------------------------------------------
    thresholds = QualityThresholds(min_words=5)
    # The flags must agree with the filter, which is what makes a dropped row explainable.
    flagged = quality_flags(corpus, "text", thresholds).to_pydict()
    kept_by_flags = {
        i for i, ok in zip(flagged["doc_id"], flagged["passes_all"], strict=True) if ok
    }
    clean = quality_filter(corpus, "text", thresholds)
    assert set(clean.to_pydict()["doc_id"]) == kept_by_flags
    assert 3 not in kept_by_flags and 4 not in kept_by_flags  # the junk went

    # --- decontaminate --------------------------------------------------------------
    assert contamination_rate(clean, "text", evals, eval_column="question", n=4) > 0
    safe = decontaminate(clean, "text", evals, eval_column="question", n=4)
    assert 5 not in set(safe.to_pydict()["doc_id"])  # the quoted eval question went

    # --- chunk ----------------------------------------------------------------------
    chunks = safe.with_columns(chunk=bt.col("text").str.chunk(60, overlap=10, boundary="word"))
    exploded = chunks.explode("chunk").filter(bt.col("chunk").str.len() > bt.lit(0))
    assert exploded.count() >= safe.count()

    # --- retrieve (a stubbed vector search over the surviving chunks) ----------------
    passages = exploded.to_pydict()["chunk"]
    ids = exploded.to_pydict()["doc_id"]
    hits = bt.from_pydict(
        {
            "question": ["why is the sky blue?"],
            "hits": [passages],
            "hit_ids": [[str(i) for i in ids]],
            "vecs": [[_stub_embedding(p) for p in passages]],
            "scores": [[1.0 - i * 0.01 for i in range(len(passages))]],
        }
    )

    # --- rerank for diversity --------------------------------------------------------
    reranked = hits.ml.map_batches(
        mmr_rerank_udf(
            embedding_column="vecs",
            score_column="scores",
            rerank_columns=("hits", "hit_ids", "scores"),
            k=2,
            lambda_mult=0.6,
        ),
        output_columns=["question", "hits", "hit_ids", "vecs", "scores"],
    )
    picked = reranked.to_pydict()
    assert len(picked["hits"][0]) == 2
    assert len(picked["hit_ids"][0]) == 2  # every reranked column stayed aligned

    # --- retrieval health ------------------------------------------------------------
    health = reranked.agg(
        empty=bt.empty_retrieval_rate("hits"),
        duplicated=bt.duplicate_context_rate("hits"),
        context_tokens=bt.context_token_estimate("hits"),
    ).to_pydict()
    assert health["empty"] == [0.0]
    assert health["context_tokens"][0] > 0

    # --- assemble the prompt ---------------------------------------------------------
    prompted = reranked.with_columns(
        prompt=bt.tagged_fields(
            question=bt.col("question"), context=bt.join_context(bt.col("hits"))
        )
    ).with_columns(
        fits=bt.fits_context("prompt", window=4096, reserve_output=256),
        tokens=bt.prompt_token_estimate(bt.col("prompt")),
    )
    built = prompted.to_pydict()
    assert "<question>" in built["prompt"][0] and "<context>" in built["prompt"][0]
    assert built["fits"] == [True]
    assert built["tokens"][0] > 0

    # --- generate (stub) and judge ---------------------------------------------------
    answered = prompted.with_columns(
        answer=bt.lit("Rayleigh scattering makes the sky blue [1]."),
        gold=bt.lit("The sky is blue because of Rayleigh scattering."),
    )
    judge = lambda: lambda prompts: ["4"] * len(prompts)  # noqa: E731 - a stub engine
    judged = answered.ml.map_batches(
        llm_score_udf(judge, template="Rate 1-5: {answer}"),
        output_columns=[*answered.columns, "score"],
    )

    # --- score, lexically and by the judge, in one aggregate --------------------------
    scored = judged.agg(
        judged_mean=bt.col("score").mean(),
        overlap=bt.ngram_f1("answer", "gold"),
        ordered=bt.rouge_l_f1("answer", "gold"),
        grounded=bt.phrase_groundedness("answer", "gold", n=2),
        injected=bt.instruction_override_rate("answer"),
        leaked=bt.credential_leak_rate("answer"),
    ).to_pydict()
    assert scored["judged_mean"] == [4.0]
    assert 0.0 < scored["overlap"][0] <= 1.0
    assert 0.0 <= scored["ordered"][0] <= scored["overlap"][0]
    assert scored["injected"] == [0.0]
    assert scored["leaked"] == [0.0]

    # --- the citation check the extractor exists for ----------------------------------
    cited = judged.select(
        fabricated=bt.extract_citations("answer").list.set_difference(bt.col("hit_ids"))
    ).to_pydict()
    assert cited["fabricated"][0] == []  # `[1]` names a chunk that was actually retrieved


def test_a_training_corpus_pipeline_composes_end_to_end():
    """The other half: mix, filter, decontaminate, then order for batching."""
    from batcher.ml import length_grouped_order, mix_corpora, padding_waste

    # Lengths vary by an order of magnitude, which is the corpus shape ordering exists for.
    web = bt.from_pydict(
        {"text": [("a web document " * (1 + i % 12)) + "written out properly." for i in range(60)]}
    )
    code = bt.from_pydict({"text": [f"def function_{i}(): return {i} * 2 + 1" for i in range(60)]})

    mixed, report = mix_corpora({"web": web, "code": code}, {"web": 3, "code": 1}, total_rows=40)
    assert report.realized_weights == {"web": 0.75, "code": 0.25}
    assert report.shortfalls == {}

    clean = quality_filter(mixed, "text", QualityThresholds(min_words=4, max_punctuation_ratio=0.4))
    assert clean.count() > 0

    evals = bt.from_pydict({"text": ["a web document number 7 written out properly."]})
    safe = decontaminate(clean, "text", evals, n=5)
    assert safe.count() <= clean.count()

    ordered = length_grouped_order(safe, "text", batch_size=4, megabatch_factor=4, seed=1)
    assert sorted(ordered.to_pydict()["text"]) == sorted(safe.to_pydict()["text"])
    assert padding_waste(ordered, "text", batch_size=4) < padding_waste(safe, "text", batch_size=4)


def test_ordering_a_uniform_corpus_buys_nothing():
    """The caveat the documentation states, asserted rather than left as advice."""
    from batcher.ml import length_grouped_order, padding_waste

    uniform = bt.from_pydict({"text": [f"document number {i:03d} here." for i in range(40)]})
    before = padding_waste(uniform, "text", batch_size=4)
    after = padding_waste(
        length_grouped_order(uniform, "text", batch_size=4, seed=1), "text", batch_size=4
    )
    assert before == pytest.approx(0.0, abs=0.02)
    assert after == pytest.approx(before, abs=0.02)
