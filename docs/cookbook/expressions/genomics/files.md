# Files: reading FASTA and FASTQ

A genomics pipeline starts at a file, and both of the formats it starts at need a real reader rather than a text read plus string work.

FASTA wraps one record's sequence across as many lines as the writer felt like, so the row boundary is the `>` header and not the newline. Read line-wise and a two-line record becomes two rows that both look like sequence, with nothing in the output to say so.

FASTQ's quality string is one character per base. A file where the sequence and quality lengths disagree is corrupt, and a reader that tolerates it produces a column where every score is attributed to the wrong base — which no downstream filter can detect.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/genomics_files.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/genomics_files.py
```

## Reading FASTA

{py:meth}`bt.read.fasta <batcher.api.io_namespace.reader.Reader.fasta>` gives one row per record as `{id, description, sequence}`, with the sequence lines re-joined. The header is split on its first run of whitespace, the NCBI convention: `>chr1 Homo sapiens chromosome 1` is the sequence named `chr1` described as the rest. A header carrying no description yields an empty string rather than null, because the description is present and empty — a different fact from a header the reader could not parse.

All five conventional suffixes are recognized: `.fasta`, `.fa`, `.faa`, `.fna`, and `.ffn`. A directory mixing them reads in one call, which matters because that is how NCBI publishes an assembly.

Reading streams. That matters more here than for most formats, because a single FASTA file is routinely a whole genome and one human chromosome is a quarter of a gigabyte in a single record.

## Reading FASTQ

{py:meth}`bt.read.fastq <batcher.api.io_namespace.reader.Reader.fastq>` gives one row per read as `{id, description, sequence, quality}`. `.fastq` and `.fq` are both recognized.

The quality column arrives as **text**, not as decoded integers, and that is deliberate. The ASCII offset is not recoverable from the bytes: Sanger and Illumina 1.8+ encode `Q + 33`, the older Illumina pipelines encoded `Q + 64`, and the two character ranges overlap. A reader that decoded here would be guessing, and a wrong guess shifts every score by 31. Decode it with {py:meth}`.seq.phred_quality <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.phred_quality>`, {py:meth}`.seq.mean_quality <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.mean_quality>`, or {py:meth}`.seq.expected_errors <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace.expected_errors>` once you know which encoding the run used.

```{important}
Pass `offset=64` for reads from an instrument older than about 2011. Nothing downstream will complain if you don't, and every score in the run will be wrong by 31.
```

## What the reader refuses

A FASTQ record whose sequence and quality strings differ in length raises, as does one missing its `@` or `+` marker, and a file that ends mid-record. These are refusals rather than skips because each one means the file is not what it claims to be, and the alternative is a plausible-looking column of misattributed scores.

The same check runs on the way out: {py:meth}`ds.write.fastq <batcher.api.io_namespace.writer.Writer.fastq>` refuses a row whose two strings disagree, rather than writing a file for the next reader to misinterpret.

## Parallelism

Splits are whole files for both formats. Neither has a byte-addressable record boundary: a FASTA record is found by scanning for a `>` at the start of a line, and a FASTQ `@` is also a legal quality character, so a byte-range split could land mid-record with nothing to say it had.

That is rarely the binding constraint, because a sequencing run arrives as many files — per lane, per sample, per mate — and a reference assembly as one file per chromosome. Where a single enormous file really is the input, convert it to Parquet once and read that.

## Writing back out

{py:meth}`ds.write.fasta <batcher.api.io_namespace.writer.Writer.fasta>` needs `id` and `sequence` columns and uses `description` for the rest of the header when present. Sequences are wrapped at 60 characters, the width the NCBI and UniProt reference files use, so a file written here is byte-comparable with the corpus it came from.

{py:meth}`ds.write.fastq <batcher.api.io_namespace.writer.Writer.fastq>` needs `id`, `sequence`, and `quality`. Both round-trip, so a filtered run is still a file the rest of a toolchain reads.

## See also

- {doc}`/cookbook/expressions/genomics/reads`: quality filtering, sketching, and primer design.
- {doc}`/cookbook/expressions/genomics/sequences`: strands, composition, translation, and motifs.
- {doc}`/user-guide/moving-data/reading-data`: the reader surface generally.
