"""Filtering an LLM pretraining corpus with the Gopher document-quality rules.

A web-scraped corpus is mostly not prose. It is navigation menus, SEO keyword spam, listing
pages of truncated teasers, and encoded blobs — and every one of those documents is
*individually unique*, so deduplication does not touch them. The Gopher rules (Rae et al.
2021) are the standard filter, and each is a per-row measure the engine evaluates natively.

The corpus-level counterparts live in `batcher.plan.functions.metrics.text` and answer a
different question: "how repetitive is my dataset" rather than "which documents do I drop".

    python examples/expressions/text_quality_filters.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    # One document of each kind a crawl produces, plus the two shapes a failed extraction
    # leaves behind.
    corpus = bt.from_pydict(
        {
            "url": [
                "good/article",
                "nav/menu",
                "seo/spam",
                "listing/teasers",
                "blob/encoded",
                "empty/page",
                "null/page",
            ],
            "text": [
                "The quick brown fox jumps over the lazy dog. It was a fine day, and the dog "
                "did not mind at all, having slept through most of the afternoon already.",
                "- home\n- about\n- contact\n- products\n- careers\n- press\n- legal\n"
                "- privacy\n- terms\n- blog",
                "cheap flights cheap flights cheap flights cheap flights cheap flights",
                "A story about the...\nAnother piece on...\nAnd a third one...",
                "aGVsbG8gd29ybGQgdGhpcyBpcyBhIGJhc2U2NCBibG9iIHdpdGggbm8gcmVhbCB3b3Jkcw==",
                "",
                None,
            ],
        }
    )

    # --- Every measure at once -------------------------------------------------------
    scored = corpus.with_columns(
        words=col("text").str.word_count(),
        mean_word_len=col("text").str.mean_word_length(),
        alpha=col("text").str.alpha_word_ratio(),
        stopwords=col("text").str.stopword_count(),
        bullets=col("text").str.bullet_line_ratio(),
        ellipses=col("text").str.ellipsis_line_ratio(),
        symbols=col("text").str.symbol_ratio(),
        top_2gram=col("text").str.top_ngram_ratio(2),
        entropy=col("text").str.char_entropy(),
    )
    result = scored.drop("text").to_pydict()
    print(result)

    by_url = dict(zip(result["url"], range(len(result["url"])), strict=True))
    nav, spam, blob, empty, missing = (
        by_url["nav/menu"],
        by_url["seo/spam"],
        by_url["blob/encoded"],
        by_url["empty/page"],
        by_url["null/page"],
    )

    # Each document is caught by a different rule, which is why the set of rules exists
    # rather than any single one.
    assert result["bullets"][nav] == 1.0  # every line is a menu item
    assert result["top_2gram"][spam] > 0.9  # one phrase is nearly the whole page
    assert result["mean_word_len"][blob] > 10  # a base64 blob is one enormous "word"
    assert result["stopwords"][blob] == 0  # and contains no English

    # The two failure shapes report null rather than a number, which matters: a 0.0 would
    # slide under every `<=` threshold and be kept.
    assert result["mean_word_len"][empty] is None
    assert result["mean_word_len"][missing] is None

    # --- The filter ------------------------------------------------------------------
    #
    # Gopher's thresholds, written out. A null on any measure fails the comparison, so the
    # empty and missing documents are dropped without needing a rule of their own.
    gopher = (
        (col("text").str.word_count() >= 20)
        & (col("text").str.mean_word_length() >= 3)
        & (col("text").str.mean_word_length() <= 10)
        & (col("text").str.symbol_ratio() <= 0.1)
        & (col("text").str.alpha_word_ratio() >= 0.8)
        & (col("text").str.stopword_count() >= 2)
        & (col("text").str.bullet_line_ratio() <= 0.9)
        & (col("text").str.ellipsis_line_ratio() <= 0.3)
        & (col("text").str.top_ngram_ratio(2) <= 0.2)
        & (col("text").str.duplicate_ngram_ratio(5) <= 0.15)
    )
    kept = corpus.filter(gopher).to_pydict()
    print(kept["url"])
    assert kept["url"] == ["good/article"]

    # --- Measuring the filter before running it --------------------------------------
    #
    # The measures are ordinary columns, so "how much would each rule remove" is a
    # group-by rather than a series of trial runs.
    # Cast each predicate to an integer before summing: the engine's `sum` is defined on
    # numbers, so counting how often a boolean holds is `cast("int64").sum()`.
    survival = (
        scored.select(
            too_short=(col("words") < 20).cast("int64"),
            bad_word_len=((col("mean_word_len") < 3) | (col("mean_word_len") > 10)).cast("int64"),
            menu=(col("bullets") > 0.9).cast("int64"),
            spammy=(col("top_2gram") > 0.2).cast("int64"),
            not_english=(col("stopwords") < 2).cast("int64"),
        )
        .agg(
            short=bt.col("too_short").sum(),
            wordlen=bt.col("bad_word_len").sum(),
            menus=bt.col("menu").sum(),
            spam=bt.col("spammy").sum(),
            foreign=bt.col("not_english").sum(),
        )
        .to_pydict()
    )
    print(survival)
    # The rules overlap on purpose, and the counts show it: six documents are dropped, but
    # the per-rule counts sum to well over six because most bad documents fail several. That
    # is why the answer to "can I drop a rule" is this table rather than a guess — the menu
    # rule is the only thing catching the menu, while the spam rule is redundant for two of
    # the documents it flags.
    assert survival["menus"] == [1]
    assert survival["spam"][0] >= 1
    assert sum(v[0] for v in survival.values()) > 6

    # --- Entropy catches what the ratios cannot --------------------------------------
    #
    # An encoded blob has ordinary-looking words, lines, and n-grams; what it has is an
    # unusual *character* distribution. Prose sits near 4-5 bits, a blob above it.
    entropies = dict(zip(result["url"], result["entropy"], strict=True))
    assert entropies["blob/encoded"] > entropies["good/article"]
    print({k: round(v, 2) for k, v in entropies.items() if v is not None})


if __name__ == "__main__":
    main()
