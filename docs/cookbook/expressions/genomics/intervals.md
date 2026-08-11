# Intervals: BED, GFF, and VCF as joinable tables

Three formats carry the coordinate half of genomics: BED holds intervals, GFF and GTF hold annotations, and VCF holds variants. Reading them into tables is what turns "which variants fall in a coding exon" into a join and a filter rather than a script — and a join optimizes, streams, and distributes like any other query.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/genomics_intervals.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/genomics_intervals.py
```

## The coordinate trap

BED is **0-based and half-open**. GFF and VCF are **1-based and inclusive**. A BED interval `chr1 0 100` and a GFF feature `chr1 1 100` describe the same hundred bases and share no number.

Nothing in this engine converts between them. Each reader reports exactly what its file says, and a comparison across formats needs the conversion written down:

```python
# docs: skip
# A BED interval [start, end) covers 1-based positions start+1 .. end.
beds_1based = beds.with_columns(start_1=bt.col("start") + 1, end_1=bt.col("end"))
```

That is deliberate and it is the most important thing on this page. A silent normalization would make an interval disagree with the file it came from and with every other tool in the pipeline, and the resulting off-by-one is invisible: every row still has plausible coordinates. Writing the conversion out is the only way it stays reviewable.

## Reading BED

{py:meth}`bt.read.bed <batcher.api.io_namespace.reader.Reader.bed>` names the columns the file's width implies. BED3 yields `chrom`, `start`, `end`; wider files add `name`, `score`, `strand`, and then the BED12 block columns, in the specification's order. The width comes from the first data line.

`track` and `browser` lines are skipped wherever they appear, which the specification allows between data blocks and not only at the top.

A directory mixing BED3 and BED6 files produces files with different schemas. That is a general problem with a general answer: read it with `schema_mode="union"`.

## Reading GFF and GTF

{py:meth}`bt.read.gff <batcher.api.io_namespace.reader.Reader.gff>` reads both dialects through one source, because they differ only in how the ninth column encodes its attributes.

That column arrives as **raw text**. Parsing it here would mean guessing the dialect — a `.gff` extension is used for both, and the encodings are not reliably distinguishable — or producing a `Map` whose keys differ per row and per feature type. The honest shape is the text plus the engine's own string vocabulary:

```python
# docs: skip
# GFF3
bt.col("attributes").str.regexp_extract(r"ID=([^;]+)", 1)
# GTF
bt.col("attributes").str.regexp_extract(r'gene_id "([^"]+)"', 1)
```

`.` reads as null in every optional column, so an absent score is a null rather than the string `"."` on a float column.

## Reading VCF

{py:meth}`bt.read.vcf <batcher.api.io_namespace.reader.Reader.vcf>` takes its column list partly from the file: the eight fixed columns are the specification's, and everything after them comes from the `#CHROM` header line. A sites-only VCF yields eight columns; a joint-called cohort yields those plus `format` and one column per sample.

The specification's names are lower-cased to match every other column in the engine. A **sample** name is not: it identifies a library and gets matched against a manifest, so folding its case could quietly merge two cohorts.

`INFO` and the genotype columns are raw text for the same reason GFF's attributes are — their keys are declared per file in the `##INFO` and `##FORMAT` metadata and differ per row:

```python
# docs: skip
bt.col("info").str.regexp_extract(r"AF=([0-9.]+)", 1).cast("float64")
bt.col("info").str.contains("DB")
```

A multi-allelic `alt` such as `T,A` stays in one field. Splitting it would multiply rows and change what a record means; do it explicitly with `.str.split` and `explode` when you want that.

## Overlap is a join

Once these are tables, an overlap query is a join on the contig plus a predicate on the coordinates:

```python
# docs: skip
in_target = (
    variants.join(regions, left_on="chrom", right_on="chrom", how="inner")
    .filter((bt.col("pos") >= bt.col("start_1")) & (bt.col("pos") <= bt.col("end_1")))
)
```

This is why the readers carry no interval logic of their own. The engine already has the join, the filter pushdown, the optimizer, and the distributed execution; an interval operation bolted onto a reader would have none of them.

## Requirements and limitations

- **VCF is read-only.** A valid VCF needs a `##` metadata block declaring every `INFO` and `FORMAT` key its records use, and that cannot be reconstructed from the columns alone. Writing one without it would produce a file that parses and that no caller can interpret, so the sink is deliberately absent. Write Parquet, or BED if you only need the intervals.
- **Splits are whole files** for all three. None has a byte-addressable record boundary: a `#` is legal inside a GFF attribute and a VCF `INFO` field, so a line found from a random offset is not necessarily a record start. These formats are normally delivered per chromosome or per cohort shard, which is where the parallelism comes from.
- **BED writes the leading run** of standard columns only. BED is positional, so a table with `chrom/start/end/strand` but no `name` writes BED3 — a gap cannot be expressed without shifting every later field.

## Surviving a bad corpus

Every reader here honours `on_error="skip"`, which drops an unreadable file and carries on
rather than aborting the scan. A corpus at scale always contains a few bad members — a
truncated upload, an interrupted write — and losing a 10,000-file run to one of them is the
wrong default.

```{warning}
The granularity is **per file, not per record**. One malformed line loses the whole file it
is in, including every good record before it. For a 50 GB FASTQ, where a truncated final
record is the commonest corruption there is, that is an expensive trade — convert to Parquet
once, or split the run into shards, if it is the wrong one for you.
```

## See also

- {doc}`/cookbook/expressions/genomics/files`: reading and writing FASTA and FASTQ.
- {doc}`/cookbook/expressions/genomics/sequences`: the per-sequence measures these coordinates point at.
- {doc}`/user-guide/analyze/index`: joins, group-by, and windows generally.
