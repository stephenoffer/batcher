"""Genomics file formats — sequences, reads, intervals, annotations, and variants.

Its own category, next to `robotics`, for the same reason that one exists: these formats share
a domain rather than a shape. A FASTA record is line-wrapped text, a FASTQ record is four fixed
lines, and BED/GFF/VCF are comment-carrying TSVs — but they are read together, joined together,
and reasoned about in one coordinate space, so keeping them in one place is what makes that
coordinate space statable once.

**The coordinate conventions differ between them, and nothing here normalizes that.** BED is
0-based half-open; GFF and VCF are 1-based inclusive. Silently converting would make a record
disagree with the file it came from and with every other tool in a pipeline, so each reader
reports what its file says and the difference is documented where a reader will meet it.

Once read, these are ordinary tables: an interval overlap is the engine's range join, a
per-contig summary is a group-by, and a per-base measure is a `.seq` expression.
"""

from __future__ import annotations

from batcher.io.formats.genomics.bed import BED_COLUMNS, BedSink, BedSource
from batcher.io.formats.genomics.fasta import FASTA_SCHEMA, FastaSink, FastaSource
from batcher.io.formats.genomics.fastq import FASTQ_SCHEMA, FastqSink, FastqSource
from batcher.io.formats.genomics.gff import GFF_SCHEMA, GffSink, GffSource
from batcher.io.formats.genomics.vcf import VCF_FIXED_COLUMNS, VcfSource

__all__ = [
    "BED_COLUMNS",
    "FASTA_SCHEMA",
    "FASTQ_SCHEMA",
    "GFF_SCHEMA",
    "VCF_FIXED_COLUMNS",
    "BedSink",
    "BedSource",
    "FastaSink",
    "FastaSource",
    "FastqSink",
    "FastqSource",
    "GffSink",
    "GffSource",
    "VcfSource",
]
