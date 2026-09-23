"""Scoring generations against references: BLEU, ROUGE, and the n-gram family.

Every metric here is a corpus aggregate: one scan, one number per group, so a million
generations score in the engine and compose with `group_by` to break a score down by prompt
category. The script checks each number against a hand count on data small enough to verify.

Two rules decide the edge cases, and both are asserted below:

- a row with fewer than `n` tokens has no n-gram of order `n`, so it scores 0 at that order
  (BLEU at ``max_n=4`` is 0 for a two-word answer, however right it is);
- a null prediction or reference is scored as the empty string: the row stays in the corpus
  and scores as a miss. Only the exact-match metrics differ, since a null never equals anything.

All the word-level metrics share the SQuAD normalization (lowercase, articles and punctuation
dropped), so treat them as stable in-engine scores for comparing runs, not as numbers to
publish against a paper's table.

    python examples/metrics/text_generation_metrics.py
"""

from __future__ import annotations

import math

import batcher as bt


def reference_scores() -> None:
    """The headline metrics on one exact and one partial generation."""
    runs = bt.from_pydict(
        {
            "prediction": ["the cat sat on the mat", "a quick brown fox"],
            "reference": ["the cat sat on the mat", "the quick brown fox jumps"],
        }
    )
    scores = runs.agg(
        bleu2=bt.bleu("prediction", "reference", max_n=2),
        brevity=bt.brevity_penalty("prediction", "reference"),
        rouge_l=bt.rouge_l_f1("prediction", "reference"),
        rouge_l_p=bt.rouge_l_precision("prediction", "reference"),
        rouge_l_r=bt.rouge_l_recall("prediction", "reference"),
        p1=bt.ngram_precision("prediction", "reference", n=1),
        r1=bt.ngram_recall("prediction", "reference", n=1),
        f1=bt.ngram_f1("prediction", "reference", n=1),
    ).to_pydict()
    print(scores)
    # Row 2 normalizes to "quick brown fox" against "quick brown fox jumps": every unigram and
    # bigram it emits is right, but it is one token short, so the brevity penalty is e^(1-4/3).
    short = math.exp(1 - 4 / 3)
    assert scores["brevity"][0] == (1.0 + short) / 2
    assert math.isclose(scores["bleu2"][0], (1.0 + short) / 2)
    assert scores["p1"][0] == 1.0
    assert math.isclose(scores["r1"][0], (1.0 + 3 / 4) / 2)
    assert math.isclose(scores["rouge_l_r"][0], (1.0 + 3 / 4) / 2)
    assert scores["rouge_l_p"][0] == 1.0


def short_rows_have_no_higher_order_ngrams() -> None:
    """A two-word match has no 3-gram or 4-gram, so 4-gram BLEU is 0; lower `max_n` for it."""
    short = bt.from_pydict({"p": ["cat sat"], "r": ["cat sat"]})
    got = short.agg(
        b4=bt.bleu("p", "r"),
        b2=bt.bleu("p", "r", max_n=2),
        p3=bt.ngram_precision("p", "r", n=3),
        distinct3=bt.distinct_ngram_ratio("p", n=3),
    ).to_pydict()
    print(got)
    assert got == {"b4": [0.0], "b2": [1.0], "p3": [0.0], "distinct3": [0.0]}


def degeneration_and_copying() -> None:
    """Distinct-n catches a looping model; novelty catches one copying its source."""
    outputs = bt.from_pydict(
        {
            "text": ["go on go on go on", "alpha beta gamma delta epsilon"],
            "source": ["unrelated words here", "alpha beta gamma delta epsilon zeta"],
        }
    )
    got = outputs.agg(
        distinct2=bt.distinct_ngram_ratio("text", n=2),
        novelty=bt.ngram_novelty("text", "source", n=4),
    ).to_pydict()
    print(got)
    # Row 1 has 2 distinct bigrams out of 5; row 2 has 4 out of 4.
    assert math.isclose(got["distinct2"][0], (2 / 5 + 1.0) / 2)
    # Row 1 shares no 4-gram with its source (novelty 1); row 2 is copied verbatim (0).
    assert got["novelty"][0] == 0.5


def nulls_score_as_empty() -> None:
    """A missing prediction or reference counts, as a miss, in every overlap metric."""
    with_nulls = bt.from_pydict(
        {"p": ["cat sat", None, "cat sat"], "r": ["cat sat", "cat sat", None]}
    )
    as_empty = bt.from_pydict({"p": ["cat sat", "", "cat sat"], "r": ["cat sat", "cat sat", ""]})
    metrics = {
        "bleu": lambda: bt.bleu("p", "r", max_n=2),
        "rouge": lambda: bt.rouge_l_f1("p", "r"),
        "brevity": lambda: bt.brevity_penalty("p", "r"),
    }
    for name, metric in metrics.items():
        got = with_nulls.agg(m=metric()).to_pydict()["m"][0]
        want = as_empty.agg(m=metric()).to_pydict()["m"][0]
        print(f"{name:>8}: {got:.4f}")
        assert got == want
    # BLEU and ROUGE-L agree: only the first of the three rows scores.
    assert math.isclose(with_nulls.agg(b=bt.bleu("p", "r", max_n=2)).to_pydict()["b"][0], 1 / 3)
    # Exact match keeps SQL's rule instead: a null never matches, and it still counts.
    em = bt.from_pydict({"p": ["a", None], "r": ["a", None]}).agg(e=bt.exact_match("p", "r"))
    assert em.to_pydict() == {"e": [0.5]}


def per_category() -> None:
    """Every metric composes with `group_by`, so one scan scores every slice."""
    evals = bt.from_pydict(
        {
            "task": ["qa", "qa", "summary", "summary"],
            "p": ["paris", "berlin", "the cat sat down", "a dog ran"],
            "r": ["paris", "paris", "the cat sat down today", "the cat sat"],
        }
    )
    got = (
        evals.group_by("task")
        .agg(rouge=bt.rouge_l_f1("p", "r"), recall=bt.ngram_recall("p", "r"))
        .sort("task")
        .to_pydict()
    )
    print(got)
    assert got["task"] == ["qa", "summary"]
    assert got["rouge"][0] == 0.5


def main() -> None:
    reference_scores()
    short_rows_have_no_higher_order_ngrams()
    degeneration_and_copying()
    nulls_score_as_empty()
    per_category()


if __name__ == "__main__":
    main()
