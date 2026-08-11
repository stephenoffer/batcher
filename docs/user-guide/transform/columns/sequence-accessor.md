# The sequence accessor

This page describes {py:class}`.seq <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace>`, the accessor that reads a text column as a biological
sequence: DNA, RNA, protein, or a FASTQ quality string.

It is a separate namespace from {py:class}`.str <batcher.plan.expr_ir.namespaces.strings._StrNamespace>` rather than more string methods, because the
operations are genuinely different rather than merely specialized. Reverse-complement is
one pass over a byte table, not `reverse` composed with `translate`. Codon translation
reads three bases at a time, which has no substring spelling that is not a per-row loop.
And GC content written as `len(replace(s, 'A', '')) / len(s)` allocates two strings per row
*and* silently counts the `N` bases you meant to exclude.

```python
import batcher as bt
```

## Strands, composition, and coding

{py:class}`.seq <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace>` reads a text column as a biological sequence: DNA, RNA, protein, or a
FASTQ quality string. It is a separate namespace rather than more `.str` methods because
the operations are genuinely different. Reverse-complement is one pass over a byte table,
not `reverse` composed with `translate`. Codon translation reads three bases at a time,
which has no substring spelling that isn't a per-row loop.

```python
contigs = bt.from_pydict({"dna": ["ATGGCCTAA", "atgnnntaa"]})
out = contigs.with_columns(
    rc=bt.col("dna").seq.reverse_complement(),
    gc=bt.col("dna").seq.gc_content(),
    protein=bt.col("dna").seq.translate(to_stop=True),
)
print(out.to_pydict())
# {'dna': ['ATGGCCTAA', 'atgnnntaa'], 'rc': ['TTAGGCCAT', 'ttannncat'], 'gc': [0.4444444444444444, 0.16666666666666666], 'protein': ['MA', 'MX']}
```

Two details there are the whole reason this namespace is worth having. The transform
preserved the second row's lower case, because lowercase is how a reference genome marks
soft-masked repeats and upper-casing would destroy the mask. And `gc_content` excluded the
three `N` bases from its denominator rather than counting them as non-GC: the second row
reports one G among the six bases that are actually known, not one among nine. Counting the
gap would have made it look more AT-rich than the data supports.

Quality strings decode through the same namespace. The ASCII offset is a parameter rather
than something sniffed from the bytes, because the Sanger (`Q+33`) and legacy Illumina
(`Q+64`) ranges overlap and a wrong guess shifts every score by 31.

```python
reads = bt.from_pydict({"qual": ["IIIIIIIIII", "IIIIIIIII#"]})
out = reads.with_columns(
    mean_q=bt.col("qual").seq.mean_quality(),
    errors=bt.col("qual").seq.expected_errors(),
)
print(out.select("mean_q", "errors").to_pydict())
# {'mean_q': [40.0, 36.2], 'errors': [0.0010000000000000002, 0.6318573444801933]}
```

The second read's mean is still high while it is expected to contain a wrong base. That is
why {py:meth}`expected_errors <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.expected_errors>` rather than {py:meth}`mean_quality <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.mean_quality>` is the quantity to filter on: it is
additive over bases, so one terrible base outweighs sixty good ones, which a mean cannot
see.

The rest of the namespace covers composition ({py:meth}`base_counts <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.base_counts>`, {py:meth}`gc_skew <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.gc_skew>`,
{py:meth}`max_homopolymer <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.max_homopolymer>`), sketching ({py:meth}`kmers <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.kmers>`, {py:meth}`canonical_kmers <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.canonical_kmers>`,
{py:meth}`minimizers <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.minimizers>`), physical properties ({py:meth}`melting_temp <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.melting_temp>`,
{py:meth}`molecular_weight <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.molecular_weight>`, {py:meth}`gravy <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.gravy>`, {py:meth}`isoelectric_point <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.isoelectric_point>`),
and IUPAC-degenerate motif search ({py:meth}`find_motif <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.find_motif>`, {py:meth}`count_motif <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.count_motif>`).
See {doc}`/cookbook/expressions/genomics/index` for those worked through.

## See also

- {doc}`/cookbook/expressions/genomics/index`: the same namespace worked through on real sequence and read data, as runnable scripts.
- {doc}`/user-guide/transform/columns/expression-accessors`: the general-purpose accessors this sits beside.
- {doc}`/api/relational/expression-accessors`: every `.seq` method, tabulated.
