"""Profiling a text corpus before deciding what to do with it.

Length distribution, vocabulary size and duplication rate are the three numbers that decide
whether a corpus is worth embedding. Computing them costs one pass and can save a very
expensive one.

    python examples/text_analytics/corpus_statistics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    corpus = tpch("part").select("p_partkey", "p_name").head(20_000)

    lengths = corpus.select(
        characters=col("p_name").str.len_chars(),
        words=col("p_name").str.word_count(),
        tokens=col("p_name").str.estimate_tokens(),
    )
    summary = lengths.agg(
        docs=bt.count(),
        mean_chars=col("characters").mean(),
        p50_chars=bt.median(col("characters")),
        p95_chars=bt.quantile(col("characters"), 0.95),
        max_chars=col("characters").max(),
        total_tokens=col("tokens").sum(),
    ).to_pydict()
    print({name: round(value[0], 2) for name, value in summary.items()})

    assert summary["docs"][0] == corpus.count()
    assert summary["p50_chars"][0] <= summary["p95_chars"][0] <= summary["max_chars"][0]

    # Duplication: how much of the corpus is the same text twice.
    distinct = corpus.n_unique("p_name")
    duplication = 1.0 - distinct / corpus.count()
    print(f"{distinct} distinct of {corpus.count()} ({duplication:.2%} duplicated)")
    assert 0.0 <= duplication < 1.0

    # Vocabulary, from the exploded tokens.
    vocabulary = corpus.select(word=col("p_name").str.split(" ")).explode("word").n_unique("word")
    print("vocabulary:", vocabulary)
    assert 0 < vocabulary < summary["total_tokens"][0]

    # The cost estimate that decides whether to proceed.
    print(f"estimated tokens to embed: {summary['total_tokens'][0]:,}")
    assert summary["total_tokens"][0] > corpus.count()


if __name__ == "__main__":
    main()
