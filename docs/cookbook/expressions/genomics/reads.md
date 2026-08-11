# Reads: quality, sketching, and primer design

Three things happen to a run of sequencing reads before any analysis: the bad ones are dropped, the rest are sketched so they can be compared, and the oligos that produced them are checked. Each is a column expression, so a run of hundreds of millions of reads stays a scan.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/genomics_reads.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/genomics_reads.py
```

## Decoding quality, and stating the offset

A FASTQ quality string encodes one Phred score per base as one ASCII character. The offset is the one thing the bytes cannot tell you: Sanger and Illumina 1.8+ encode `Q + 33`, the older Illumina 1.3 to 1.7 pipelines encoded `Q + 64`, and the two character ranges overlap. Sniffing it would be a guess, and a wrong guess shifts every score by 31, turning a Q40 base into Q9.

So it is a parameter, defaulting to 33. A character below the offset decodes to a *negative* score rather than being clamped to zero, because a negative score is the unmistakable signature of the wrong choice and clamping would hide it.

```{important}
If your reads came off an instrument before about 2011, pass `offset=64`. Every score in the run is wrong by 31 otherwise, and nothing downstream will complain.
```

## Filtering on expected errors, not on a mean

{py:meth}`mean_quality <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.mean_quality>` is the "average quality" every FASTQ tool reports, and it is what a `mean_quality >= 20` filter means. It is also the wrong quantity to filter on, and the script shows why: a read of nine Q40 bases and one Q2 base still has a mean above 32, while it is expected to contain more than one miscalled base.

{py:meth}`expected_errors <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.expected_errors>` is `sum(10 ** (-Q / 10))`, the number of bases the read is expected to get wrong. It is additive over bases, so one Q2 base contributes as much as sixty Q20 bases, which is exactly what a mean cannot see. `expected_errors() < 1.0` is the USEARCH and VSEARCH `fastq_maxee` criterion, and it corresponds to an actual claim about the data rather than to a summary statistic.

## Sketching with k-mers and minimizers

{py:meth}`canonical_kmers <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.canonical_kmers>` folds each window with its reverse complement and keeps the lexicographically smaller one. A read and the same fragment sequenced from the other strand therefore produce the same k-mer table, which is what makes two reads comparable at all. It is the rule Jellyfish, KMC, and minimap2 all use, so a table built here is comparable with one built by those tools.

{py:meth}`minimizers <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.minimizers>` keeps only the lexicographically smallest canonical k-mer of each window of `window` consecutive k-mers, collapsing consecutive repeats. The result is roughly `2/(window+1)` of the k-mers, which is what makes it a sketch rather than a re-encoding.

The reason a sketch is safe to compare is a guarantee rather than a heuristic: two sequences sharing a substring of length `window + k - 1` are **guaranteed** to share a minimizer. So an `array_intersect` between two rows' sketches cannot miss a real overlap, which is what seed-and-extend alignment is built on.

The output is a list column rather than packed integers, deliberately, because a list composes with the vocabulary the engine already has. `explode` plus `group_by` is a k-mer frequency table. `.list.n_unique()` is a cardinality estimate. `.list.intersect()` between two rows is a shared-substring count. A packed `u64` encoding would be faster per k-mer and would compose with nothing.

## Designing primers

Screening candidate oligos is a filter over a computed column, which is the shape this engine is for. The alternative — a per-row call into a design library over millions of candidates — is a control-plane row loop.

{py:meth}`melting_temp <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.melting_temp>` uses the SantaLucia (1998) unified nearest-neighbour model, the one primer3 and Biopython's `Tm_NN` default to, at 50 mM Na+ and 500 nM total strand. Those conditions are part of the answer: melting temperature is not a property of a sequence alone, and halving the concentration moves it by several degrees.

A nearest-neighbour model is used rather than the Wallace rule or a GC-percentage formula because those two read a sequence as a bag of bases. `GCGCGC` and `GGGCCC` get the same answer from them despite stacking very differently, and the error is several degrees — enough to put a primer outside its annealing window.

A sequence containing any character outside `ACGT` yields null rather than an approximate temperature. An ambiguity code has no defined stacking energy, and reporting a specific number the data does not support would be worse than reporting nothing.

## Requirements and limitations

- {py:meth}`melting_temp <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.melting_temp>` is fixed at 50 mM Na+ and 500 nM strand. If your buffer differs materially, treat the output as a ranking rather than as an absolute temperature.
- {py:meth}`molecular_weight <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.molecular_weight>` reports a **single** strand. Double it for a duplex.
- The degenerate alphabets have no defined mass, so {py:meth}`molecular_weight <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.molecular_weight>` accepts only `dna`, `rna`, and `protein`, and raises rather than returning a column of nulls.
- `k` and `window` are capped at 256. Assembly uses 21 to 127 and alignment seeds 15 to 31, so this binds on nothing real, but it does catch a `k` that was meant to be a window.

## See also

- {doc}`/cookbook/expressions/genomics/sequences`: strands, composition, translation, and motifs.
- {doc}`/cookbook/expressions/nested/lists_set_operations`: the list vocabulary a sketch comparison uses.
- {doc}`/api/relational/expression-accessors`: every {py:class}`.seq <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace>` method, tabulated.
