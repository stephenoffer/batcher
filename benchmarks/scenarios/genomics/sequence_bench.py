"""Throughput of the `.seq` genomics kernels and the `.str` document-quality filters.

Two jobs, and the first is the one that matters most.

**Validate a claim.** `.str.word_count()` was `regexp_count(r"\\S+")` and is now a native
single-pass scan. That change was made *for speed*, and a speed change asserted without a
measurement is exactly what `.claude/rules/performance.md` forbids. This measures both
spellings over the same column so the claim is a number rather than an argument.

**Find the bottlenecks.** The rest of the kernels are timed per megabase so the expensive
ones are identifiable rather than guessed at. The k-mer family is the one to watch: it emits
a `List<Utf8>` and therefore allocates per k-mer, which was a deliberate choice (a list column
composes with `explode`, `n_unique`, and `array_intersect`; a packed `u64` composes with
nothing) and is worth knowing the price of.

There is no cross-engine comparison here on purpose. DuckDB and Polars have no counterpart to
these kernels, so a "ratio vs DuckDB" column would be a made-up number. What is comparable is
the *old* spelling against the new, and a per-byte rate against the rest of the engine.

Run:
    python benchmarks/scenarios/genomics/sequence_bench.py
    python benchmarks/scenarios/genomics/sequence_bench.py --rows 200000
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import batcher as bt
from batcher import col
from harness import bench

# A fixed seed so two runs of this script are comparable; the shapes below are chosen to
# look like real data rather than to flatter any kernel.
_RNG = random.Random(20240805)

_BASES = "ACGT"
_WORDS = [
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "a",
    "lazy",
    "dog",
    "while",
    "the",
    "cat",
    "sat",
    "on",
    "its",
    "mat",
    "and",
    "considered",
    "whether",
    "any",
    "of",
    "this",
    "was",
    "worth",
    "the",
    "effort",
    "involved",
]


def _reads(n: int, length: int = 150) -> bt.Dataset:
    """A short-read table: sequence plus an equally long quality string."""
    seqs = ["".join(_RNG.choice(_BASES) for _ in range(length)) for _ in range(min(n, 2000))]
    quals = ["".join(chr(33 + _RNG.randint(2, 40)) for _ in range(length)) for _ in seqs]
    # Repeat the sampled block up to `n` so generation stays fast while the scan stays honest
    # — every row is still a distinct string object to the engine.
    reps = (n // len(seqs)) + 1
    return bt.from_pydict({"seq": (seqs * reps)[:n], "qual": (quals * reps)[:n]})


def _documents(n: int, words: int = 120) -> bt.Dataset:
    """A text corpus with the shapes a web crawl produces."""
    docs = []
    for _ in range(min(n, 2000)):
        body = " ".join(_RNG.choice(_WORDS) for _ in range(words))
        docs.append(body)
    reps = (n // len(docs)) + 1
    return bt.from_pydict({"text": (docs * reps)[:n]})


def _mb(ds: bt.Dataset, column: str) -> float:
    """Megabytes of text in `column`, for a per-byte rate rather than a per-row one."""
    total = ds.select(n=col(column).str.len().sum()).to_pydict()["n"][0]
    return total / 1e6


def _row(label: str, ms: float, mb: float) -> None:
    rate = mb / (ms / 1000.0) if ms > 0 else float("inf")
    print(f"  {label:<34} {ms:9.1f} ms   {rate:8.1f} MB/s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200_000)
    args = ap.parse_args()

    print(f"engine: {bt.versions()['engine_profile']}   rows: {args.rows:,}\n")

    # --- The claim: native word_count vs the regex it replaced -----------------------
    docs = _documents(args.rows).collect()
    docs = bt.from_arrow(docs)
    text_mb = _mb(docs, "text")
    print(f"document corpus: {text_mb:.1f} MB of text")

    native = bench(lambda: docs.select(v=col("text").str.word_count()).collect())
    regex = bench(lambda: docs.select(v=col("text").str.regexp_count(r"\S+")).collect())
    _row("word_count (native scan)", native, text_mb)
    _row("word_count (regexp_count, the old one)", regex, text_mb)
    speedup = regex / native if native > 0 else float("nan")
    print(f"  -> native is {speedup:.1f}x the old spelling\n")

    # Correctness before timing: the two must agree, or the number above is meaningless.
    a = docs.select(v=col("text").str.word_count()).to_pydict()["v"]
    b = docs.select(v=col("text").str.regexp_count(r"\S+")).to_pydict()["v"]
    assert a == b, "the two spellings disagree; the timing above is not a comparison"

    # --- Document-quality filters ----------------------------------------------------
    print("document quality (per row, over the same corpus)")
    for label, expr in [
        ("mean_word_length", col("text").str.mean_word_length()),
        ("symbol_ratio", col("text").str.symbol_ratio()),
        ("alpha_word_ratio", col("text").str.alpha_word_ratio()),
        ("stopword_count", col("text").str.stopword_count()),
        ("duplicate_line_ratio", col("text").str.duplicate_line_ratio()),
        ("char_entropy", col("text").str.char_entropy()),
        ("top_ngram_ratio(2)", col("text").str.top_ngram_ratio(2)),
        ("duplicate_ngram_ratio(5)", col("text").str.duplicate_ngram_ratio(5)),
    ]:
        _row(label, bench(lambda e=expr: docs.select(v=e).collect()), text_mb)

    # The whole Gopher filter, which is what a corpus pipeline actually runs.
    gopher = (
        (col("text").str.word_count() >= 50)
        & (col("text").str.mean_word_length() >= 3)
        & (col("text").str.mean_word_length() <= 10)
        & (col("text").str.symbol_ratio() <= 0.1)
        & (col("text").str.alpha_word_ratio() >= 0.8)
        & (col("text").str.stopword_count() >= 2)
        & (col("text").str.bullet_line_ratio() <= 0.9)
        & (col("text").str.top_ngram_ratio(2) <= 0.2)
    )
    _row("the whole Gopher filter", bench(lambda: docs.filter(gopher).collect()), text_mb)

    # --- Sequence kernels ------------------------------------------------------------
    reads = bt.from_arrow(_reads(args.rows).collect())
    seq_mb = _mb(reads, "seq")
    print(f"\nread table: {seq_mb:.1f} MB of sequence")
    for label, expr in [
        ("reverse_complement", col("seq").seq.reverse_complement()),
        ("gc_content", col("seq").seq.gc_content()),
        ("base_counts", col("seq").seq.base_counts()),
        ("translate", col("seq").seq.translate()),
        ("max_homopolymer", col("seq").seq.max_homopolymer()),
        ("count_motif(GAATTC)", col("seq").seq.count_motif("GAATTC")),
        ("melting_temp", col("seq").seq.melting_temp()),
        ("kmers(21)", col("seq").seq.kmers(21)),
        ("canonical_kmers(21)", col("seq").seq.canonical_kmers(21)),
        ("minimizers(21, 10)", col("seq").seq.minimizers(21, 10)),
    ]:
        _row(label, bench(lambda e=expr: reads.select(v=e).collect()), seq_mb)

    print("\nFASTQ quality (per row, over the quality column)")
    qual_mb = _mb(reads, "qual")
    for label, expr in [
        ("mean_quality", col("qual").seq.mean_quality()),
        ("expected_errors", col("qual").seq.expected_errors()),
        ("phred_quality (List<Int32>)", col("qual").seq.phred_quality()),
    ]:
        _row(label, bench(lambda e=expr: reads.select(v=e).collect()), qual_mb)

    # --- Assembly aggregates ---------------------------------------------------------
    print("\nassembly aggregates (grouped over contig lengths)")
    lengths = bt.from_pydict(
        {
            "asm": [f"s{i % 100}" for i in range(args.rows)],
            "len": [_RNG.randint(500, 5_000_000) for _ in range(args.rows)],
        }
    ).collect()
    lengths = bt.from_arrow(lengths)
    for label, stat in [
        ("n50", lambda c: c.n50()),
        ("l50", lambda c: c.l50()),
        ("aun", lambda c: c.aun()),
        ("median (for reference)", lambda c: c.median()),
    ]:
        ms = bench(lambda s=stat: lengths.group_by("asm").agg(v=s(col("len"))).collect())
        print(f"  {label:<34} {ms:9.1f} ms   {args.rows / (ms / 1000.0) / 1e6:8.1f} M rows/s")


if __name__ == "__main__":
    main()
