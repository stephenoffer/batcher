# Sequences: strands, composition, and coding

Four things are done to a nucleotide column before anything else: read the other strand, measure what it is made of, translate it, and search it for a site. Each is one expression, evaluated per base in Rust.

Two of these have a trap that only shows up on real data. {py:meth}`reverse_complement <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.reverse_complement>` must complement the IUPAC ambiguity codes as IUPAC defines them, which is invisible on a test of pure ACGT and wrong on any variant call. {py:meth}`gc_content <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.gc_content>` must exclude ambiguous bases from its denominator, or a run of `N` reads as an AT-rich region instead of as no data.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/genomics_sequences.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/genomics_sequences.py
```

## Reading the other strand

{py:meth}`reverse_complement <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.reverse_complement>` is the most-used operation in genomics, because a sequencing read maps to either strand and anything comparing two sequences has to normalize for that. It preserves case, so a soft-masked repeat comes back soft-masked, and it leaves any character that is not a nucleotide code exactly where it was.

Applying it twice is the identity, which is worth using as a check when you are unsure whether a column has already been flipped.

## Measuring composition

{py:meth}`base_counts <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.base_counts>` yields every count in one pass, as a struct, so asking for the `N` count costs nothing beyond asking for the `A` count. Project the one you want with `.struct.field("n")`.

The `other` field is the useful one for data quality: it counts every byte that is neither one of the five bases nor `N`, so `other > 0` is a "this row is not clean sequence" predicate that finds a header line, a quality string, or a sentinel that leaked into the wrong column.

{py:meth}`gc_skew <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.gc_skew>` is `(G - C) / (G + C)`. Its sign flips at a bacterial chromosome's replication origin and again at the terminus, which is how an origin is located in a newly assembled genome. Computed over a sliding window rather than a whole contig, it is a signal rather than a summary.

## Translating

{py:meth}`translate <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.translate>` uses NCBI genetic code table 1 and accepts DNA or RNA. Three of its choices are worth knowing before you rely on the output:

- A codon containing any ambiguous base becomes `X`. Resolving `GTN` to valine would be correct, but resolving it in general needs the whole degenerate table, and picking one of four would put a specific residue where the data supports none.
- A trailing partial codon is dropped rather than padded. Two leftover bases encode nothing, and padding them would fabricate a residue.
- `to_stop=True` ends the protein at the first stop codon, excluding the stop. That is what "the protein this ORF encodes" means; the default runs to the end and marks every stop with `*`.

A six-frame translation is the three `frame` values, plus the same three over {py:meth}`reverse_complement <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.reverse_complement>`.

## Searching for a site

{py:meth}`find_motif <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.find_motif>` and {py:meth}`count_motif <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.count_motif>` take a motif in the IUPAC degenerate alphabet, so `GGWWTT` matches all four of `GGAATT`, `GGATTT`, `GGTATT`, and `GGTTTT`.

Matching is defined on *sets of bases* rather than on text, and that is what makes this different from a regular expression over the literal characters. Ambiguity works in both directions: an `N` in the reference is consistent with every pattern base, so it matches. A character-class regex gets that half wrong.

Matches overlap, so `AA` occurs three times in `AAAA`. That is the biologically meaningful count — tandem repeats and overlapping binding sites are real — and it is what separates this from a replace-and-measure spelling, which counts only non-overlapping occurrences. Positions are 1-based, matching every genome browser, GFF file, and VCF record you would compare them against.

A motif containing a character that is not an IUPAC code raises rather than matching nothing, because a column of empty lists would hide the typo.

## See also

- {doc}`/cookbook/expressions/genomics/reads`: quality filtering, sketching, and primer design.
- {doc}`/cookbook/expressions/strings/index`: the general-purpose text vocabulary.
- {doc}`/api/relational/expression-accessors`: every {py:class}`.seq <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace>` method, tabulated.
