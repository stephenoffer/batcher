# Assembly: N50, N90, L50, and auN

An assembler hands back a pile of contigs. How good is the assembly? Total sequence does not
say — two assemblies of the same genome hold the same bases whether they are in five pieces or
five thousand. What distinguishes them is *contiguity*, and the statistics that measure it are
base-weighted rather than item-weighted.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/genomics_assembly_stats.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/genomics_assembly_stats.py
```

## Why these are not quantiles

This is the one thing to take from the page. {py:meth}`.n50() <batcher.AggExpr>` is not
`median(length)`, and reaching for the median instead is the mistake the statistic exists to
prevent.

A median weighs every contig equally. An assembly of one 10 Mb chromosome plus a thousand
500 bp fragments has a median contig length of **500** — a number that describes the debris
and says nothing about the chromosome. N50 weighs by *base*: it is the length `L` such that
contigs at least `L` long hold half the assembly's sequence. The same assembly has an N50 of
**10 Mb**, which is the answer a biologist wants.

The engine has both, and they disagree by four orders of magnitude on that input. That is not
a rounding difference; they are answering different questions.

## The four statistics

| Statistic | What it is | Reading it |
|---|---|---|
| {py:meth}`.n50() <batcher.AggExpr>` | The length at which the running total, longest-first, reaches half the assembly | Higher is better |
| {py:meth}`.n90() <batcher.AggExpr>` | The same at 90% | Higher is better; never above N50 |
| {py:meth}`.l50() <batcher.AggExpr>` | The **count** of contigs needed to reach half | Lower is better |
| {py:meth}`.aun() <batcher.AggExpr>` | Area under the Nx curve, `sum(l²)/sum(l)` | Higher is better |

**N is a length, L is a count.** That is the pair people mix up, and the types say so:
`n50`/`n90`/`aun` come back Float64, `l50` comes back Int64. No assembly has 3.5 contigs.

## Why auN exists

N50 is a *step* function of the length distribution. A single contig crossing the halfway mark
moves it discontinuously, so two assemblies that are genuinely close can swap rank on what
amounts to a rounding difference — which makes N50 a poor number to rank on, however good it
is to report.

auN integrates over every threshold instead of picking one, so it is continuous in the lengths.
It is also the cheapest of the four, needing no sort at all: it is exactly `sum(l²)/sum(l)`,
the base-weighted mean length.

## They merge, which is why they scale

All four reuse the same per-group value-list state the median does, so
`combine(partial(p₁), partial(p₂))` reproduces the single-node answer exactly. That is not a
detail: a cohort's assemblies are produced per sample and summarized with a group-by over a
shuffle, and a statistic that could not merge would cap the metric at one machine — returning
a *different number* on a cluster rather than an error.

The test suite pins it across `collect()`, `collect(spill=True)`, and `iter_batches()` past a
morsel boundary, and with a scale-invariance check: duplicating every contig leaves N50, N90
and auN unchanged and exactly doubles L50.

## Requirements and limitations

- **Null, negative, and non-finite lengths are excluded**, not summed. A negative length would
  cancel real sequence out of the total every statistic divides by, quietly lowering all four.
- **A group with no usable length is null**, not zero — so an empty assembly fails a
  `n50() >= 1000` threshold instead of sliding under it.
- These read a *length column*. From a FASTA that is one expression:
  `bt.col("sequence").str.len()`.
- There is no `l90()`. It is rarely cited, and the four here cover what is. If you need it,
  the shape is the same walk at a different threshold.

## See also

- {doc}`/cookbook/expressions/genomics/files`: reading the FASTA these lengths come from.
- {doc}`/cookbook/expressions/genomics/sequences`: the per-contig measures.
- {doc}`/api/relational/expressions`: the full aggregate reference.
