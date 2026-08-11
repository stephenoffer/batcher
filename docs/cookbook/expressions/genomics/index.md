# Genomics

This section holds runnable recipes for the {py:class}`.seq <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace>` accessor, the column language for biological sequences.

A genomics pipeline is a string pipeline with a different alphabet. That sounds like it should make {py:class}`.str <batcher.plan.expr_ir.namespaces.strings._StrNamespace>` sufficient, and it is the reason this namespace exists rather than not existing. Reverse-complement is one pass over a byte table, not `reverse` composed with `translate`. Codon translation reads three bases at a time and has no substring spelling that isn't a row loop. GC content written as `len(replace(s, 'A', '')) / len(s)` allocates two strings per row and silently counts the `N` bases you meant to exclude, which turns an assembly gap into an AT-rich region.

Everything here runs per-base in Rust over whole columns, so a scan over a reference genome or a run of sequencing reads never materializes a sequence in Python.

| Recipe | Covers |
|---|---|
| {doc}`/cookbook/expressions/genomics/files` | Reading and writing FASTA and FASTQ, and what each reader refuses |
| {doc}`/cookbook/expressions/genomics/sequences` | Strands, composition, codon translation, and IUPAC motif search |
| {doc}`/cookbook/expressions/genomics/reads` | FASTQ quality filtering, k-mer sketching, and primer design |
| {doc}`/cookbook/expressions/genomics/intervals` | BED, GFF/GTF, and VCF as joinable tables, and the coordinate trap |
| {doc}`/cookbook/expressions/genomics/assembly` | N50, N90, L50, and auN — judging an assembly, as mergeable aggregates |

## Case, and why it is not uniform

The transforms preserve case and the measures fold it. That asymmetry is deliberate and it is the Biopython convention: lowercase is how every reference genome marks soft-masked repeats, so a transform that upper-cased would destroy the mask, while a measure that respected it would report a repeat-rich contig as mostly-unknown.

The sketching functions ({py:meth}`kmers <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.kmers>`, {py:meth}`canonical_kmers <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.canonical_kmers>`, {py:meth}`minimizers <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.minimizers>`) fold case for a third reason: a k-mer table that treated `acgt` and `ACGT` as different strings would split every repeat-adjacent count in two.

## What is not here

Coordinates are the other half, and the three formats that carry them ({doc}`BED, GFF, and VCF </cookbook/expressions/genomics/intervals>`) deliberately do not normalize between each other's conventions — BED is 0-based half-open, GFF and VCF are 1-based inclusive, and a silent shift is the field's most common off-by-one.

There is no aligner and no variant caller. Those are iterative algorithms over a whole genome index rather than per-row column work, and wrapping one behind an expression would misrepresent what it costs. What this namespace gives you is the layer underneath: the per-sequence measures a filter thresholds on, and the sketches ({py:meth}`minimizers <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.minimizers>`) that a seed-and-extend aligner is built on top of.

## See also

- {doc}`/user-guide/transform/columns/expression-accessors`: the guide to the accessor namespaces generally.
- {doc}`/api/relational/expression-accessors`: every method on every namespace, tabulated.
- {doc}`/cookbook/expressions/strings/index`: the general-purpose text vocabulary these sit beside.

```{toctree}
:hidden:

files
sequences
reads
intervals
assembly
```
