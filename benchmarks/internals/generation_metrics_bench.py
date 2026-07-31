"""What the generation metrics cost, and which one is not like the others.

The clipped-overlap metrics (BLEU, ROUGE-N, novelty) are linear in the two texts' lengths;
ROUGE-L is quadratic, because a longest-common-subsequence has no cheaper form. That is stated
in `list.lcs_length`'s documentation, and this is the measurement behind the statement — the
number that tells you whether scoring whole documents with it is affordable on your corpus.

It compares each metric against the cheapest thing the engine can do over the same column (a
token count), so the reported figure is the metric's own cost rather than the scan's. There is
no cross-engine comparison here: DuckDB has no BLEU, and an implementation written in SQL for
the occasion would measure the SQL, not the engine.

This benchmark is also what found the shared normalization to be the dominant term: it cost
ninety times a bare `len` over the same column, and every word-level metric paid it once per
text column. Folding those five expressions into one kernel (`str.squad_normalize`) made it
4.7x faster with byte-identical output. Re-run this after touching that path.

Run:
    python benchmarks/internals/generation_metrics_bench.py [rows] [tokens_per_row]
"""

from __future__ import annotations

import random
import sys
import time
from collections.abc import Callable

import batcher as bt

#: Repeats per measurement. The linear metrics are fast enough that one sample is noise.
REPEATS = 3

_VOCAB = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and", "then",
    "runs", "away", "into", "forest", "where", "nobody", "can", "find", "it", "again",
]  # fmt: skip


def build(rows: int, tokens: int, seed: int = 0) -> bt.Dataset:
    """A corpus of generated/reference pairs that genuinely overlap.

    The pairs share a prefix and diverge, which is what a real generation does. Two independent
    random texts would make every overlap zero and let a short-circuiting implementation look
    fast for the wrong reason.
    """
    rng = random.Random(seed)

    def sentence() -> list[str]:
        return [rng.choice(_VOCAB) for _ in range(tokens)]

    answers, golds = [], []
    for _ in range(rows):
        base = sentence()
        answers.append(" ".join(base))
        # Keep the first half, resample the rest: a real partial match.
        golds.append(" ".join(base[: tokens // 2] + sentence()[tokens // 2 :]))
    return bt.from_pydict({"answer": answers, "gold": golds})


def _time(label: str, build_metric: Callable[[], object], ds: bt.Dataset) -> float:
    """Milliseconds for one aggregate over the corpus, best of `REPEATS`."""
    best = float("inf")
    for _ in range(REPEATS):
        started = time.perf_counter()
        ds.agg(m=build_metric()).to_pydict()
        best = min(best, (time.perf_counter() - started) * 1000.0)
    print(f"  {label:<28} {best:8.1f} ms")
    return best


def main() -> None:
    """Time every generation metric over one corpus and report them together."""
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    ds = build(rows, tokens)
    print(f"generation metrics over {rows:,} pairs of ~{tokens} tokens (best of {REPEATS})\n")

    baseline = _time("token count (baseline)", lambda: bt.total_token_estimate("answer"), ds)
    linear = {
        "ngram_precision (n=1)": lambda: bt.ngram_precision("answer", "gold"),
        "ngram_f1 (n=4)": lambda: bt.ngram_f1("answer", "gold", n=4),
        "bleu (max_n=4)": lambda: bt.bleu("answer", "gold"),
        "ngram_novelty (n=4)": lambda: bt.ngram_novelty("answer", "gold"),
        "distinct_ngram_ratio": lambda: bt.distinct_ngram_ratio("answer"),
    }
    for label, build_metric in linear.items():
        _time(label, build_metric, ds)

    print()
    quadratic = _time("rouge_l_f1 (quadratic)", lambda: bt.rouge_l_f1("answer", "gold"), ds)

    print(
        f"\nrouge_l_f1 costs {quadratic / max(baseline, 1e-9):.1f}x the baseline scan at "
        f"{tokens} tokens per row. That ratio grows with the token count, which is the "
        f"reason to truncate or score per sentence rather than per document."
    )


if __name__ == "__main__":
    main()
