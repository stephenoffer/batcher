"""Trim a text corpus to what a model can actually read, before tokenizing it.

Two filters do most of the work on a scraped corpus, and both are cheap enough to run
before anything expensive touches the rows. `filter_by_length` drops the documents that are
too short to carry a signal (a navigation stub, a cookie banner) or so long they would be
truncated anyway. `filter_by_token_budget` answers the different question a fine-tune
actually asks: will this document fit the context window once it is tokenized?

The token estimate is characters divided by `chars_per_token`, which is a planning figure,
not a tokenizer. Use it to shed the documents that cannot possibly fit before paying for a
real tokenizer pass on the rest.

    python examples/quality/trimming_a_text_corpus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt


def main() -> None:
    corpus = bt.from_pydict(
        {
            "doc_id": [1, 2, 3, 4, 5],
            "body": [
                "Home | About | Contact",  # navigation chrome, 22 chars
                "",  # an extraction that produced nothing
                "A short but genuine paragraph about batching data.",  # 50 chars
                "x" * 400,  # long enough to blow a small context window
                "Another real paragraph, comfortably within budget.",  # 50 chars
            ],
        }
    )

    # Keep documents of at least 40 characters. The empty extraction and the navigation
    # chrome go; nothing else does.
    long_enough = corpus.filter_by_length("body", 40)
    kept = long_enough.sort("doc_id").to_pydict()["doc_id"]
    assert kept == [3, 4, 5], kept

    # A ceiling as well as a floor: `max_chars` bounds the other end, so a single scraped
    # page cannot dominate a batch.
    bounded = corpus.filter_by_length("body", 40, 100).sort("doc_id").to_pydict()["doc_id"]
    assert bounded == [3, 5], bounded

    # The budget filter asks the question a context window asks. At the default 4.0
    # characters per token, a 400-character document is ~100 tokens, so a 64-token budget
    # sheds it while the 50-character paragraphs (~13 tokens) survive.
    affordable = long_enough.filter_by_token_budget("body", 64)
    ids = affordable.sort("doc_id").to_pydict()["doc_id"]
    assert ids == [3, 5], ids

    # `chars_per_token` is the knob to move when your tokenizer is denser than the default
    # (code and CJK both are). Tightening it to 1.0 makes the same 50-character paragraphs
    # read as ~50 tokens, which the same 64-token budget still admits.
    dense = long_enough.filter_by_token_budget("body", 64, chars_per_token=1.0)
    assert dense.sort("doc_id").to_pydict()["doc_id"] == [3, 5]

    # ...but a 40-token budget at that density sheds them too, which is the point: the
    # estimate is a planning figure and its parameter changes what survives.
    tight = long_enough.filter_by_token_budget("body", 40, chars_per_token=1.0)
    assert tight.to_pydict()["doc_id"] == [], tight.to_pydict()["doc_id"]

    print(
        f"corpus {corpus.count()} docs -> {long_enough.count()} long enough "
        f"-> {affordable.count()} within a 64-token budget"
    )


if __name__ == "__main__":
    main()
