# The .seq namespace

This page is the reference for `.seq`, the accessor a biological sequence column carries: DNA, RNA, protein, or a FASTQ quality string held as text. Reach it as {py:obj}`col("dna").seq <batcher.plan.expr_ir.core.Expr.seq>`.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.sequence

.. autoclass:: _SeqNamespace
   :no-members:
```

## Strands and translation

Validate a sequence, read its opposite strand, or transcribe and translate it.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.sequence

.. autosummary::
   :toctree: generated
   :nosignatures:

   _SeqNamespace.is_valid
   _SeqNamespace.complement
   _SeqNamespace.reverse_complement
   _SeqNamespace.transcribe
   _SeqNamespace.back_transcribe
   _SeqNamespace.translate
```

## Composition and physical properties

Count bases and compute a sequence's GC content, weight, and melting temperature.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.sequence

.. autosummary::
   :toctree: generated
   :nosignatures:

   _SeqNamespace.base_counts
   _SeqNamespace.gc_content
   _SeqNamespace.gc_skew
   _SeqNamespace.max_homopolymer
   _SeqNamespace.molecular_weight
   _SeqNamespace.melting_temp
   _SeqNamespace.isoelectric_point
   _SeqNamespace.gravy
```

## K-mers and motifs

Break a sequence into k-mers or minimizers, and find degenerate motifs in it.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.sequence

.. autosummary::
   :toctree: generated
   :nosignatures:

   _SeqNamespace.kmers
   _SeqNamespace.canonical_kmers
   _SeqNamespace.minimizers
   _SeqNamespace.find_motif
   _SeqNamespace.count_motif
```

## Read quality

Decode a FASTQ quality string and summarize it.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.sequence

.. autosummary::
   :toctree: generated
   :nosignatures:

   _SeqNamespace.phred_quality
   _SeqNamespace.mean_quality
   _SeqNamespace.expected_errors
```

## See also

- {doc}`index`: the other accessor namespaces, and which column kind each one attaches to.
- {doc}`/api/relational/expression-accessors`: the same methods with a runnable example per namespace.
- {doc}`/api/symbols/expression-methods`: the {py:obj}`Expr <batcher.plan.expr_ir.core.Expr>` these namespaces hang off.
- {doc}`/user-guide/transform/columns/sequence-accessor`: the guide these methods are the reference for.
- {doc}`/cookbook/expressions/genomics/index`: runnable recipes over reads, intervals, and assemblies.
