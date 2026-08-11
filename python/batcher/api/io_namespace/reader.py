"""The `bt.read` namespace — typed, per-format dataset readers.

`bt.read` is a single callable object: call it for path autodetection
(``bt.read("s3://b/*.parquet")``) or use a typed method per format
(``bt.read.parquet(path)``, ``bt.read.delta(uri, version=3)``,
``bt.read.kafka(topic="events")``). The methods are thin, typed wrappers over the
generic dispatch (`session.read`/`read_table`); format implementations live in
`io/formats/` and register into the `SOURCES` registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher.api.io_namespace._discovery import (
    PathLike,
    namespace_dir,
    namespace_repr,
    unknown_attribute,
)
from batcher.api.session import read as _read
from batcher.api.session import read_table as _read_table

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from batcher.api.dataset import Dataset

__all__ = ["Reader", "read"]


def _decode(ds: Dataset, fn_name: str, **kwargs: Any) -> Dataset:
    """Append decoded media tensors to `ds` via the named `batcher.ml.decode` helper."""
    from batcher.ml import decode

    return getattr(decode, fn_name)(ds, **kwargs)


class Reader:
    """The `bt.read` namespace: callable for autodetect, typed methods per format.

    ``bt.read(path)`` infers the format from the URI scheme or file extension;
    ``bt.read.<format>(...)`` is the explicit, discoverable spelling. File/object
    formats take a path; catalog/SQL/NoSQL/streaming sources take their own
    connector arguments. A path may be a string, a `pathlib.Path`, or a list of
    either. Every file format also accepts ``columns=`` and ``n_rows=`` (spelled
    ``usecols=``/``nrows=`` if you come from pandas), whatever else it takes.

    Examples:
        .. doctest::

            >>> import batcher as bt, tempfile, os
            >>> p = os.path.join(tempfile.mkdtemp(), "t.parquet")
            >>> _ = bt.from_pydict({"x": [1, 2]}).write.parquet(p)
            >>> bt.read.parquet(p).sort("x").to_pydict()
            {'x': [1, 2]}
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """List the formats this namespace reads, grouped by family."""
        return namespace_repr(self, "bt.read")

    def __dir__(self) -> list[str]:
        """Every format method, so tab-completion shows the readable formats."""
        return namespace_dir(self)

    def __getattr__(self, name: str) -> Any:
        """Answer a misspelled format with a suggestion instead of a bare `AttributeError`.

        Only ever reached on a miss, so it cannot shadow a real method. A `_`-prefixed
        name still raises `AttributeError` — `copy`, `pickle`, and IPython probe for those
        and require a miss to look like a miss.
        """
        raise unknown_attribute(self, "bt.read", name)

    def __call__(
        self, path: PathLike | Sequence[PathLike], *, format: str | None = None, **opts: Any
    ) -> Dataset:
        r"""Read a file/object-store dataset, dispatching on `format` or the path.

        With no `format`, it is inferred from the URI scheme (``delta://``…) or the
        file extension. ``bt.read("s3://b/*.parquet")`` → Parquet;
        ``bt.read("data/", format="csv")``. `path` may be a string, a `pathlib.Path`,
        or a list of either — a list is read as one relation, its format taken from
        the first entry.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> p = os.path.join(tempfile.mkdtemp(), "t.csv")
                >>> _ = open(p, "w").write("a,b\n1,2\n")
                >>> bt.read(p).to_pydict()
                {'a': [1], 'b': [2]}
        """
        return _read(path, format=format, **opts)

    def table(self, format: str, *args: Any, **opts: Any) -> Dataset:
        """Read any registered non-file source by name (escape hatch).

        ``bt.read.table("delta", "s3://bucket/table", version=3)``. The typed
        methods below wrap this for the common backends.

        Args:
            format: Registered source name to dispatch to (e.g. ``"delta"``).
            args: Positional arguments forwarded to that source.
            opts: Connector options (connection, credentials, query) as keywords.

        Returns:
            A lazy `Dataset` over the named source.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.table("delta", "s3://bucket/table", version=3)  # doctest: +SKIP
        """
        return _read_table(format, *args, **opts)

    # --- File / object-store formats (path-addressed) ----------------------
    def parquet(self, path: PathLike, **opts: Any) -> Dataset:
        """Read a Parquet file, directory, or glob (e.g. ``d/*.parquet``).

        Kyber pushes column projection and row-group predicates into the read, so a
        filtered/projected query touches only the needed columns and row groups.

        Args:
            path: A Parquet file, directory, or glob to read.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` over the Parquet source.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> import pyarrow as pa, pyarrow.parquet as pq
                >>> p = os.path.join(tempfile.mkdtemp(), "t.parquet")
                >>> pq.write_table(pa.table({"a": [1], "b": [2]}), p)
                >>> bt.read.parquet(p).to_pydict()
                {'a': [1], 'b': [2]}
        """
        return _read(path, format="parquet", **opts)

    def parquet_dataset(self, path: PathLike, **opts: Any) -> Dataset:
        """Read a (Hive-)partitioned Parquet dataset directory.

        Partition columns are recovered from the directory layout, and projection plus
        predicate pushdown (including partition pruning) are applied per fragment.

        Args:
            path: Root directory of the partitioned Parquet dataset.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` over the partitioned Parquet dataset.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> import pyarrow as pa, pyarrow.parquet as pq
                >>> root = os.path.join(tempfile.mkdtemp(), "pds")
                >>> pq.write_to_dataset(
                ...     pa.table({"x": [1, 2], "part": ["a", "b"]}),
                ...     root,
                ...     partition_cols=["part"],
                ... )
                >>> bt.read.parquet_dataset(root).select("x").sort("x").to_pydict()
                {'x': [1, 2]}
        """
        return _read(path, format="parquet_dataset", **opts)

    def csv(self, path: PathLike, **opts: Any) -> Dataset:
        r"""Read a CSV file, directory, or glob (e.g. ``d/*.csv``).

        The header row and column types are auto-inferred; column projection is pushed
        into the read and a single large file is split into newline-aligned byte ranges
        for parallel parsing.

        Types are inferred from the file's **first block**, which is what pyarrow's
        streaming reader commits to and what DuckDB and Polars sample. A column that is
        integral for a million rows and then holds ``"N/A"`` was therefore inferred wrong,
        and the read fails with a message naming this. Pass ``schema=`` to declare the
        types when inference cannot reach them — every read path (whole-file, streaming,
        and each byte-range split) is pinned to the advertised schema, so they cannot
        disagree with it or with each other.

        A row whose field count disagrees with the header is a different failure, and it
        aborts the file rather than the value. ``on_bad_lines="skip"`` (or ``"warn"``) drops
        such a row and keeps the rest, which is what a corpus at scale needs — reaching for
        ``on_error="skip"`` there discards every good row in the file to be rid of one bad
        line. Dropped rows are counted as ``malformed_rows_total`` on the metrics export.

        Args:
            path: A CSV file, directory, or glob to read.
            opts: Format-specific reader options forwarded to the source — notably
                ``schema`` (a `pyarrow.Schema` declaring the column types), ``on_error``
                (drop an unreadable *file*), and ``on_bad_lines`` (drop a malformed *row*:
                ``"error"``, ``"warn"``, or ``"skip"``).

        Returns:
            A lazy `Dataset` over the CSV source.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> p = os.path.join(tempfile.mkdtemp(), "t.csv")
                >>> _ = open(p, "w").write("a,b\n1,2\n")
                >>> bt.read.csv(p).to_pydict()
                {'a': [1], 'b': [2]}
        """
        return _read(path, format="csv", **opts)

    def json(self, path: PathLike, **opts: Any) -> Dataset:
        r"""Read newline-delimited JSON: a file, directory, or glob.

        One JSON object per line; column types are inferred from the records.

        A line that is not JSON at all aborts the file. ``on_bad_lines="skip"`` (or
        ``"warn"``) drops that record and keeps the rest, which is what an event or log
        corpus needs — reaching for ``on_error="skip"`` there discards every good record in
        the file. A record that *parses* but does not fit the inferred types is deliberately
        not dropped: that is inference having been shown too little, answered by ``schema=``.

        Args:
            path: A JSON file, directory, or glob to read.
            opts: Format-specific reader options forwarded to the source — notably
                ``on_error`` (drop an unreadable *file*) and ``on_bad_lines`` (drop an
                unparseable *record*: ``"error"``, ``"warn"``, or ``"skip"``).

        Returns:
            A lazy `Dataset` over the JSON source.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> p = os.path.join(tempfile.mkdtemp(), "t.json")
                >>> _ = open(p, "w").write('{"a": 1, "b": 2}\n')
                >>> bt.read.json(p).to_pydict()
                {'a': [1], 'b': [2]}
        """
        return _read(path, format="json", **opts)

    def orc(self, path: PathLike, **opts: Any) -> Dataset:
        """Read ORC file(s) — file, directory, or glob — with column projection pushed in.

        Args:
            path: An ORC file, directory, or glob to read.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` over the ORC source.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> import pyarrow as pa, pyarrow.orc as orc
                >>> p = os.path.join(tempfile.mkdtemp(), "t.orc")
                >>> orc.write_table(pa.table({"a": [1], "b": [2]}), p)
                >>> bt.read.orc(p).to_pydict()
                {'a': [1], 'b': [2]}
        """
        return _read(path, format="orc", **opts)

    def arrow(self, path: PathLike, **opts: Any) -> Dataset:
        """Read Arrow/Feather IPC file(s) — file, directory, or glob — zero-copy into the engine.

        Args:
            path: An Arrow/Feather file, directory, or glob to read.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` over the Arrow/Feather source.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> import pyarrow as pa, pyarrow.feather as fe
                >>> p = os.path.join(tempfile.mkdtemp(), "t.arrow")
                >>> fe.write_feather(pa.table({"a": [1], "b": [2]}), p)
                >>> bt.read.arrow(p).to_pydict()
                {'a': [1], 'b': [2]}
        """
        return _read(path, format="arrow", **opts)

    def avro(self, path: PathLike, **opts: Any) -> Dataset:
        """Read Avro file(s): a file, directory, or glob.

        Needs the optional extra: ``pip install 'batcher-engine[avro]'``.

        Args:
            path: An Avro file, directory, or glob to read.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` over the Avro source.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.avro("data/events.avro")  # doctest: +SKIP
        """
        return _read(path, format="avro", **opts)

    def fasta(self, path: PathLike, **opts: Any) -> Dataset:
        """Read FASTA file(s) as rows of ``{id, description, sequence}``.

        One row per record, with the sequence lines re-joined — a FASTA writer wraps a
        sequence across as many lines as it likes, and the row boundary is the ``>`` header,
        not the newline. The header is split on its first whitespace into `id` and
        `description`, the NCBI convention. Any of ``.fasta``, ``.fa``, ``.faa``, ``.fna``,
        and ``.ffn`` are recognized, so a directory mixing them reads in one call.

        Reading streams, so a reference genome held in a handful of enormous records never
        materializes whole.

        Args:
            path: A FASTA file, directory, or glob to read.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` over the FASTA source.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.fasta("genomes/GRCh38.fa")  # doctest: +SKIP
                >>> gc = ds.select(bt.col("sequence").seq.gc_content())  # doctest: +SKIP
        """
        return _read(path, format="fasta", **opts)

    def fastq(self, path: PathLike, **opts: Any) -> Dataset:
        """Read FASTQ file(s) as rows of ``{id, description, sequence, quality}``.

        One row per sequencing read. ``.fastq`` and ``.fq`` are both recognized.

        The quality string arrives as **text**, not as decoded scores, because the ASCII
        offset is not recoverable from the bytes — Sanger and Illumina 1.8+ encode ``Q+33``
        and the older pipelines ``Q+64``, and the ranges overlap. Decode it through the
        ``.seq`` accessor (``phred_quality``, ``mean_quality``, ``expected_errors``) once you
        know which encoding the run used.

        A record whose sequence and quality strings differ in length raises: the quality
        string is one character per base, so a mismatch would attribute every score
        downstream to the wrong base.

        Args:
            path: A FASTQ file, directory, or glob to read.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` over the FASTQ source.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> reads = bt.read.fastq("run/*.fq.gz")  # doctest: +SKIP
                >>> clean = reads.filter(  # doctest: +SKIP
                ...     bt.col("quality").seq.expected_errors() < 1.0
                ... )
        """
        return _read(path, format="fastq", **opts)

    def bed(self, path: PathLike, **opts: Any) -> Dataset:
        """Read BED interval file(s), with the standard column names for the file's width.

        BED3 yields ``chrom/start/end``; wider files add ``name``, ``score``, ``strand`` and
        the BED12 block columns in the specification's order. The width is read from the
        file's first data line. Browser ``track`` and ``browser`` lines are skipped.

        ``.bed``, ``.bed.gz``, and ``.bedgraph`` are recognized.

        .. important::
           BED coordinates are **0-based and half-open**, unlike GFF and VCF which are
           1-based and inclusive. They are read exactly as written — nothing here normalizes
           between the two conventions, because a silent shift would make an interval
           disagree with its own file.

        Args:
            path: A BED file, directory, or glob to read.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` over the BED source.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> exons = bt.read.bed("annotations/exons.bed")  # doctest: +SKIP
        """
        return _read(path, format="bed", **opts)

    def gff(self, path: PathLike, **opts: Any) -> Dataset:
        """Read GFF3 or GTF annotation file(s) as nine-column rows.

        Both dialects read through one source: they differ only in how the ninth column
        encodes its attributes, and that column arrives as raw text rather than parsed, since
        the dialect is not reliably recoverable from the file. Pull a key out with the string
        vocabulary, e.g. ``col("attributes").str.regexp_extract(r"ID=([^;]+)", 1)``.

        ``.gff``, ``.gff3``, ``.gtf`` and their gzipped forms are recognized. Coordinates are
        1-based and inclusive; ``.`` reads as null in every optional column.

        Args:
            path: A GFF/GTF file, directory, or glob to read.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` over the annotation source.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> genes = bt.read.gff("gencode.v44.gtf")  # doctest: +SKIP
        """
        return _read(path, format="gff", **opts)

    def vcf(self, path: PathLike, **opts: Any) -> Dataset:
        """Read VCF variant file(s), with one column per sample when genotypes are present.

        The eight fixed columns are ``chrom``, ``pos``, ``id``, ``ref``, ``alt``, ``qual``,
        ``filter``, ``info``. Sample names come from the file's ``#CHROM`` header, so a
        joint-called cohort yields ``format`` plus one string column per sample. ``.`` reads
        as null.

        ``INFO`` and the genotype columns arrive as raw text, because their keys are declared
        per file and differ per row; extract with the string vocabulary, e.g.
        ``col("info").str.regexp_extract(r"AF=([0-9.]+)", 1).cast("float64")``.

        Args:
            path: A VCF file, directory, or glob to read.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` over the variant source.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> variants = bt.read.vcf("cohort.vcf")  # doctest: +SKIP
                >>> passing = variants.filter(bt.col("filter") == "PASS")  # doctest: +SKIP
        """
        return _read(path, format="vcf", **opts)

    def lance(self, path: PathLike, **opts: Any) -> Dataset:
        """Read a Lance dataset (columnar ML format) by directory path.

        Needs the optional extra: ``pip install 'batcher-engine[lance]'``.

        Args:
            path: Directory path of the Lance dataset.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` over the Lance dataset.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.lance("data/embeddings.lance")  # doctest: +SKIP
        """
        return _read(path, format="lance", **opts)

    def excel(self, path: PathLike, **opts: Any) -> Dataset:
        """Read Excel workbook(s) — a file, directory, or glob — via python-calamine.

        Needs the optional extra: ``pip install 'batcher-engine[excel]'``.

        Args:
            path: An Excel workbook file, directory, or glob to read.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` over the Excel source.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.excel("report.xlsx")  # doctest: +SKIP
        """
        return _read(path, format="excel", **opts)

    def xml(self, path: PathLike, **opts: Any) -> Dataset:
        """Read XML file(s) — a file, directory, or glob — into columnar rows.

        Needs the optional extra: ``pip install 'batcher-engine[xml]'``.

        Args:
            path: An XML file, directory, or glob to read.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` over the XML source.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.xml("data/records.xml")  # doctest: +SKIP
        """
        return _read(path, format="xml", **opts)

    def msgpack(self, path: PathLike, **opts: Any) -> Dataset:
        """Read MessagePack file(s) — a file, directory, or glob — into columnar rows.

        The counterpart of `ds.write.msgpack`. Without it, MessagePack was the one format
        this engine could write by name and not read by name: `ds.write.msgpack(p)` succeeded,
        `bt.read.msgpack(p)` raised `FormatError: Unknown format 'msgpack'`, and the reader
        the registry already held was reachable only through the untyped
        ``bt.read(p, format="msgpack")``.

        Needs the optional extra: ``pip install 'batcher-engine[msgpack]'``.

        Args:
            path: A MessagePack file, directory, or glob to read.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` over the MessagePack source.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.msgpack("data/events.msgpack")  # doctest: +SKIP
        """
        return _read(path, format="msgpack", **opts)

    def logs(self, path: PathLike, **opts: Any) -> Dataset:
        """Read line-delimited log file(s) as rows, one raw line per row by default.

        Pass ``pattern=`` to extract fields with a grok pattern instead.

        Args:
            path: A log file, directory, or glob.
            pattern: Optional grok pattern; named captures become columns.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` over the log source.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.logs(  # doctest: +SKIP
                ...     "/var/log/app/*.log",
                ...     pattern="%{IP:client} %{WORD:method} %{URIPATHPARAM:path}",
                ... )
        """
        return _read(path, format="logs", **opts)

    def text(self, path: PathLike, **opts: Any) -> Dataset:
        r"""Read text file(s) as rows, one row per line by default.

        Args:
            path: A text file, directory, or glob.
            mode: ``"line"`` for one row per line, or ``"file"`` for whole-file rows.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` over the text source.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> p = os.path.join(tempfile.mkdtemp(), "t.txt")
                >>> _ = open(p, "w").write("hello\nworld\n")
                >>> bt.read.text(p).select("line_number", "text").to_pydict()
                {'line_number': [1, 2], 'text': ['hello', 'world']}
        """
        return _read(path, format="text", **opts)

    def warc(self, path: PathLike, **opts: Any) -> Dataset:
        r"""Read web-archive (WARC) file(s) as one row per record.

        The front door of a web-scale text corpus: Common Crawl and every major crawler
        ship WARC. Filter to ``warc_type == "response"``, then take the payload apart with
        the string accessors — ``.str.strip_html()``, ``.str.chunk()`` — without leaving
        the engine.

        The named WARC headers become typed columns; every other header is carried as JSON
        in ``warc_headers``, so a crawl's own extension fields stay reachable through
        ``.json``. ``.warc.gz`` is read transparently, including the per-record gzip
        members a crawler normally writes.

        Args:
            path: A ``.warc`` / ``.warc.gz`` file, directory, or glob.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` over the WARC records.

        Examples:
            .. doctest::

                >>> import batcher as bt, os, tempfile
                >>> body = b"<html>hi</html>"
                >>> rec = (
                ...     b"WARC/1.0\r\nWARC-Type: response\r\n"
                ...     b"WARC-Target-URI: https://example.com/\r\n"
                ...     b"Content-Length: %d\r\n\r\n" % len(body)
                ... ) + body + b"\r\n\r\n"
                >>> p = os.path.join(tempfile.mkdtemp(), "c.warc")
                >>> _ = open(p, "wb").write(rec)
                >>> bt.read.warc(p).select("warc_type", "warc_target_uri").to_pydict()
                {'warc_type': ['response'], 'warc_target_uri': ['https://example.com/']}
        """
        return _read(path, format="warc", **opts)

    def binary(self, path: PathLike, **opts: Any) -> Dataset:
        """Read whole files as ``{uri, bytes, size, mime}`` rows.

        The entry point for custom/multimodal decoding of arbitrary file(s).

        Args:
            path: A file, directory, or glob to read as whole-file rows.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` of whole-file rows.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> p = os.path.join(tempfile.mkdtemp(), "b.bin")
                >>> _ = open(p, "wb").write(b"abc")
                >>> bt.read.binary(p).select("bytes", "size", "mime").to_pydict()
                {'bytes': [b'abc'], 'size': [3], 'mime': ['application/octet-stream']}
        """
        return _read(path, format="binary", **opts)

    def documents(self, path: PathLike, *, password: str | None = None, **opts: Any) -> Dataset:
        """Read PDF document(s) — a file, directory, or glob — as extracted text rows.

        Needs the optional extra: ``pip install 'batcher-engine[pdf]'``.

        Extracting the text is skipped entirely when the ``text`` column is not projected,
        so ``select("path", "page")`` and ``count()`` read the page tree and stop. Laying a
        page out into reading order is most of the cost of this reader.

        Args:
            path: A PDF file, directory, or glob to read.
            password: The user password for an encrypted corpus. A PDF encrypted for
                *permissions* only opens without it.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` of extracted document text rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.documents("docs/*.pdf")  # doctest: +SKIP
        """
        return _read(path, format="documents", password=password, **opts)

    def numpy(self, path: PathLike, **opts: Any) -> Dataset:
        """Read NumPy ``.npy``/``.npz`` file(s) — file, directory, or glob — as tensor rows.

        Args:
            path: A NumPy ``.npy``/``.npz`` file, directory, or glob to read.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` of tensor rows.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os, numpy as np
                >>> p = os.path.join(tempfile.mkdtemp(), "t.npy")
                >>> np.save(p, np.array([1, 2, 3]))
                >>> bt.read.numpy(p).to_pydict()
                {'data': [1, 2, 3]}
        """
        return _read(path, format="numpy", **opts)

    def point_cloud(self, path: PathLike, **opts: Any) -> Dataset:
        """Read LiDAR / point-cloud file(s) — ``.pcd`` / ``.ply`` / raw ``.bin`` — as points.

        The native robotics / autonomous-driving point-cloud formats, with no third-party
        dependency. Each file is one frame; every point becomes a row with a column per
        field (``x``/``y``/``z``/``intensity``/…) plus a ``frame`` column naming the source
        file — so region cropping, ground-plane removal, and voxel binning are native
        operators, and a directory of sweeps stays separable via ``group_by("frame")``.

        Args:
            path: a point-cloud file, directory, or glob.
            opts: ``columns`` (field names for a raw ``.bin`` layout, default
                ``("x", "y", "z", "intensity")``), ``dtype`` (``.bin`` element type),
                and ``frame_column`` (the appended source-file column; ``None`` to omit).

        Returns:
            A lazy `Dataset` of point rows.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os, numpy as np
                >>> p = os.path.join(tempfile.mkdtemp(), "sweep.bin")
                >>> np.array([[1, 2, 3, 0.5]], dtype=np.float32).tofile(p)
                >>> bt.read.point_cloud(p, frame_column=None).to_pydict()
                {'x': [1.0], 'y': [2.0], 'z': [3.0], 'intensity': [0.5]}
        """
        return _read(path, format="point_cloud", **opts)

    def mcap(self, path: PathLike, **opts: Any) -> Dataset:
        """Read MCAP robot / vehicle log(s) — the ROS 2 and ADAS recording format — as messages.

        One log multiplexes every sensor as timestamped messages on named topics, so a row
        is a *message*: ``{topic, log_time, publish_time, sequence, schema_name,
        message_encoding, data}``. Payloads stay encoded in ``data``, so a query that wants
        ``/gps`` never pays to deserialize ``/camera``.

        The container is indexed, which this reader exploits: ``row_count`` and the
        ``log_time`` bounds come from the summary without reading a message, and a filter on
        ``topic`` or ``log_time`` is pushed into the reader as a seek. Naming `topics` up
        front does the same thing explicitly.

        The multi-sensor idiom follows: filter to the topics you want, then `join_asof` them
        on ``log_time`` to align sensors sampled at different rates.

        Args:
            path: an ``.mcap`` file, directory, or glob.
            opts: ``topics`` (restrict the read to these topic names) and ``on_error``
                (``"raise"``, or ``"skip"`` to drop an unreadable log and carry on).

        Returns:
            A lazy `Dataset` of message rows.

        Examples:
            .. doctest::

                >>> import batcher as bt  # doctest: +SKIP
                >>> from batcher import col  # doctest: +SKIP
                >>> log = bt.read.mcap("s3://drives/2026-07-18/")  # doctest: +SKIP
                >>> imu = log.filter(col("topic") == "/imu")  # doctest: +SKIP
                >>> lidar = log.filter(col("topic") == "/lidar/top")  # doctest: +SKIP
                >>> # Align the 100 Hz IMU onto each 10 Hz LiDAR sweep.
                >>> aligned = lidar.join_asof(imu, on="log_time")  # doctest: +SKIP
        """
        return _read(path, format="mcap", **opts)

    def mdf(self, path: PathLike, **opts: Any) -> Dataset:
        """Read ASAM MDF4 (``.mf4``) vehicle measurement(s) — CAN/LIN and sensor channels.

        MDF is what automotive OEMs and test fleets log to. A file holds several *channel
        groups*, each with its own sampling raster, so this reads **long format** — one row
        per signal sample, ``{signal, timestamp, value, unit}`` — giving one schema for the
        whole file however many rasters it has. Widening it into one table would mean
        resampling (inventing data) or nulls at every raster boundary; resampling stays an
        explicit choice you make.

        ``timestamp`` is absolute, so a measurement as-of joins against an MCAP log from the
        same drive. ``value`` is ``float64``; non-numeric diagnostic channels are skipped and
        `signals()` lists what is actually readable.

        Args:
            path: an ``.mf4`` file, directory, or glob.
            opts: ``signals`` (restrict to these channel names — the usual case, since a
                measurement carries thousands) and ``on_error``.

        Returns:
            A lazy `Dataset` of signal samples.

        Examples:
            .. doctest::

                >>> import batcher as bt  # doctest: +SKIP
                >>> from batcher import col  # doctest: +SKIP
                >>> speed = ["VehicleSpeed"]
                >>> m = bt.read.mdf("s3://fleet/drive.mf4", signals=speed)  # doctest: +SKIP
                >>> hard_braking = m.filter(col("value") < 5.0)  # doctest: +SKIP
        """
        return _read(path, format="mdf", **opts)

    def webdataset(self, path: PathLike, **opts: Any) -> Dataset:
        """Read WebDataset ``.tar`` shard(s), grouping each sample's member files into one row.

        Args:
            path: A ``.tar`` shard file, directory, or brace-expansion glob.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` with one row per WebDataset sample.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.webdataset("s3://bucket/shards/{000..099}.tar")  # doctest: +SKIP
        """
        return _read(path, format="webdataset", **opts)

    def tfrecord(self, path: PathLike, **opts: Any) -> Dataset:
        """Read TFRecord file(s) — the Waymo Open Dataset / TFDS / RLDS container format.

        Each length-prefixed, CRC-checked record becomes a row in a ``record`` binary
        column (the raw serialized payload — commonly a ``tf.train.Example`` protobuf);
        decode it downstream with a `map_batches`. CRC verification uses ``crc32c`` when
        installed. Reads records with no TensorFlow dependency.

        Args:
            path: a ``.tfrecord``/``.tfrecords`` file, directory, or glob.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` with one row per record.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.tfrecord("gs://waymo_open_dataset/*.tfrecord")  # doctest: +SKIP
        """
        return _read(path, format="tfrecord", **opts)

    def hdf5(self, path: PathLike, **opts: Any) -> Dataset:
        """Read HDF5 file(s) — a file, directory, or glob — with datasets as columns.

        Needs the optional extra: ``pip install 'batcher-engine[hdf5]'``.

        Args:
            path: An HDF5 file, directory, or glob to read.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` over the HDF5 source.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.hdf5("data/measurements.h5")  # doctest: +SKIP
        """
        return _read(path, format="hdf5", **opts)

    def zarr(self, path: PathLike, **opts: Any) -> Dataset:
        """Read a Zarr store (chunked n-dimensional arrays) by path.

        Needs the optional extra: ``pip install 'batcher-engine[zarr]'``.

        Args:
            path: Path or URI of the Zarr store.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` over the Zarr store.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.zarr("s3://bucket/array.zarr")  # doctest: +SKIP
        """
        return _read(path, format="zarr", **opts)

    # --- Multimodal --------------------------------------------------------
    def images(
        self,
        path: PathLike,
        *,
        decode: bool = False,
        size: tuple[int, int] | None = None,
        **opts: Any,
    ) -> Dataset:
        """List image file(s) as ``{uri, bytes, size, mime}`` + header-metadata rows.

        ``decode=True`` (or passing ``size=``) appends an ``image`` (H, W, 3) uint8
        tensor column; decoding needs the optional extra:
        ``pip install 'batcher-engine[image]'``.

        Args:
            path: An image file, directory, or glob.
            decode: If true, append the decoded ``image`` tensor column.
            size: ``(height, width)`` to resize decoded images to; implies ``decode``.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` of image rows, optionally with a decoded tensor column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.images(  # doctest: +SKIP
                ...     "s3://bucket/images/*.jpg", decode=True, size=(224, 224)
                ... )
        """
        ds = _read(path, format="images", **opts)
        return _decode(ds, "image_tensor_dataset", size=size) if (decode or size) else ds

    def audio(
        self,
        path: PathLike,
        *,
        decode: bool = False,
        sample_rate: int | None = None,
        **opts: Any,
    ) -> Dataset:
        """List audio file(s) + header-metadata rows.

        ``decode=True`` appends a ``waveform`` ``list<float32>`` column via soundfile,
        optionally resampled; decoding needs the optional extra:
        ``pip install 'batcher-engine[audio]'``.

        Args:
            path: An audio file, directory, or glob.
            decode: If true, append the decoded ``waveform`` column.
            sample_rate: Target sample rate in Hz to resample to when decoding.
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` of audio rows, optionally with a decoded waveform column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.audio(  # doctest: +SKIP
                ...     "data/clips/*.wav", decode=True, sample_rate=16000
                ... )
        """
        ds = _read(path, format="audio", **opts)
        return _decode(ds, "audio_dataset", sample_rate=sample_rate) if decode else ds

    def video(
        self,
        path: PathLike,
        *,
        decode: bool = False,
        size: tuple[int, int] | None = None,
        num_frames: int = 8,
        **opts: Any,
    ) -> Dataset:
        """List video file(s) + header-metadata rows.

        ``decode=True`` (or passing ``size=``) appends a ``frames`` (num_frames, H, W, 3)
        uint8 tensor column. An engine built with the ``video`` feature decodes it in the
        data plane; otherwise the decode falls back to PyAV, which needs the optional
        extra ``pip install 'batcher-engine[video]'``. Either way a clip that will not
        decode is null rather than a black frame.

        Args:
            path: A video file, directory, or glob.
            decode: If true, append the decoded ``frames`` tensor column.
            size: ``(height, width)`` to resize decoded frames to; implies ``decode``.
            num_frames: Number of frames to sample per video (default 8).
            opts: Format-specific reader options forwarded to the source.

        Returns:
            A lazy `Dataset` of video rows, optionally with a decoded frames column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.video(  # doctest: +SKIP
                ...     "s3://bucket/clips/*.mp4", decode=True, num_frames=16
                ... )
        """
        ds = _read(path, format="video", **opts)
        if not (decode or size):
            return ds
        return _decode(ds, "video_dataset", size=size, num_frames=num_frames)

    # --- Lakehouse ---------------------------------------------------------
    def delta(
        self,
        table_uri: str,
        *,
        version: int | None = None,
        timestamp: str | datetime | None = None,
        stream: bool = False,
        starting_version: int = 0,
        **opts: Any,
    ) -> Dataset:
        """Read a Delta Lake table by URI, defaulting to its latest version.

        Needs the optional extra: ``pip install 'batcher-engine[delta]'``.

        Args:
            table_uri: Path/URI of the Delta table root.
            version: Time-travel to this table version (exclusive with ``timestamp``).
            timestamp: Time-travel to the version current as of this ISO timestamp.
            stream: Read the table as an unbounded stream of new commits (Spark
                ``readStream``) instead of a snapshot — see `delta_stream`. Requires
                ``delta.enableChangeDataFeed = true`` on the table.
            starting_version: When streaming, the first version to read from (default 0).
            opts: Connector options passed through to the Delta source.

        Returns:
            A lazy `Dataset` over the Delta table (a snapshot, or a stream if ``stream``).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.delta("s3://bucket/delta/events", version=3)  # doctest: +SKIP
        """
        if stream:
            return _read_table("delta_stream", table_uri, starting_version=starting_version, **opts)
        return _read_table("delta", table_uri, version=version, timestamp=timestamp, **opts)

    def read_change_feed(
        self,
        table_uri: str,
        *,
        starting_version: int | None = None,
        ending_version: int | None = None,
        starting_timestamp: str | datetime | None = None,
        ending_timestamp: str | datetime | None = None,
        **opts: Any,
    ) -> Dataset:
        """Read a Delta table's Change Data Feed (Databricks ``readChangeFeed``).

        Yields row-level changes — ``_change_type`` (insert/update/delete),
        ``_commit_version``, ``_commit_timestamp`` plus the data columns. Requires
        ``delta.enableChangeDataFeed = true`` on the table.

        Naming **any** bound (``ending_version``, ``starting_timestamp``,
        ``ending_timestamp``) makes this a *bounded* read of that window, which you can
        collect, count, join, and merge into a target — the shape an incremental ETL step
        wants. With no bound it is an unbounded stream of new commits, for
        `write.stream`-style continuous processing.

        Args:
            table_uri: Path/URI of the Delta table root.
            starting_version: First commit version to read changes from. Defaults to
                version 0, unless ``starting_timestamp`` names the start instead.
            ending_version: Last commit version to read, inclusive. Bounds the read.
            starting_timestamp: Read changes from the first commit at or after this time,
                instead of a version. Bounds the read. Ignored if ``starting_version`` is
                also given, which is delta-rs's precedence.
            ending_timestamp: Read changes up to the last commit at or before this time.
                Bounds the read.
            opts: Connector options passed through to the Delta source.

        Returns:
            A lazy `Dataset` of row-level change records — bounded if any bound was given.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> changes = bt.read.read_change_feed(  # doctest: +SKIP
                ...     "s3://bucket/delta/events", starting_version=10, ending_version=20
                ... )
                >>> inserts = changes.filter(  # doctest: +SKIP
                ...     bt.col("_change_type") == "insert"
                ... )
                >>> inserts.count()  # doctest: +SKIP
                42
        """
        bounds = (ending_version, starting_timestamp, ending_timestamp)
        if all(b is None for b in bounds):
            return _read_table(
                "delta_stream",
                table_uri,
                starting_version=0 if starting_version is None else starting_version,
                change_feed=True,
                **opts,
            )
        return _read_table(
            "delta_cdf",
            table_uri,
            starting_version=starting_version,
            ending_version=ending_version,
            starting_timestamp=starting_timestamp,
            ending_timestamp=ending_timestamp,
            **opts,
        )

    def iceberg(
        self,
        identifier: str,
        *,
        catalog: str | None = None,
        snapshot_id: int | None = None,
        **opts: Any,
    ) -> Dataset:
        """Read an Iceberg table by catalog identifier (e.g. ``"db.table"``).

        Needs the optional extra: ``pip install 'batcher-engine[iceberg]'``.

        Args:
            identifier: Table identifier within the catalog.
            catalog: Named catalog to resolve against (defaults to the configured one).
            snapshot_id: Time-travel to this Iceberg snapshot id.
            opts: Connector options passed through to the Iceberg source.

        Returns:
            A lazy `Dataset` over the Iceberg table.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.iceberg("db.events", catalog="prod")  # doctest: +SKIP
        """
        return _read_table("iceberg", identifier, catalog=catalog, snapshot_id=snapshot_id, **opts)

    def hudi(self, table_uri: str, **opts: Any) -> Dataset:
        """Read an Apache Hudi table by URI (read-only, snapshot query).

        Needs the optional extra: ``pip install 'batcher-engine[hudi]'``.

        Args:
            table_uri: Path/URI of the Hudi table root.
            opts: Connector options passed through to the Hudi source.

        Returns:
            A lazy `Dataset` over the Hudi table snapshot.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.hudi("s3://bucket/hudi/events")  # doctest: +SKIP
        """
        return _read_table("hudi", table_uri, **opts)

    def delta_sharing(self, url: str, **opts: Any) -> Dataset:
        """Read a Delta Sharing table by ``<profile>#<share>.<schema>.<table>`` URL.

        Needs the optional extra: ``pip install 'batcher-engine[delta-sharing]'``.

        Args:
            url: The ``<profile>#<share>.<schema>.<table>`` sharing URL.
            opts: Connector options passed through to the Delta Sharing source.

        Returns:
            A lazy `Dataset` over the shared table.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.delta_sharing("config.share#share.schema.table")  # doctest: +SKIP
        """
        return _read_table("delta_sharing", url, **opts)

    # --- SQL / warehouses --------------------------------------------------
    def sql(
        self,
        query: str | None = None,
        *,
        uri: str | None = None,
        connection: Any = None,
        **opts: Any,
    ) -> Dataset:
        """Read any SQL database from a standard connection URI, in a single submission.

        The URI vocabulary is SQLAlchemy's — the same string that works in
        ``sqlalchemy.create_engine``, pandas' ``read_sql``, or an existing
        ``$DATABASE_URL`` — and Batcher routes it to whichever backend can serve it:
        ADBC for databases with an Arrow-native driver (PostgreSQL, SQLite, DuckDB,
        Snowflake, BigQuery, FlightSQL), ConnectorX for the rest (MySQL, SQL Server,
        Oracle, Redshift, Trino). You do not choose the backend; the scheme does.

        Kyber pushes the projection and the filter into the SQL that is actually
        submitted, so the database does the work and only the surviving rows and columns
        cross the wire. `schema()` costs nothing — it uses a zero-row probe rather than
        running your query a second time.

        Pass ``password="env:PGPASSWORD"`` rather than embedding a password in the URI:
        the reference is what travels to workers, and it becomes a secret only where the
        connection is opened. For a database with no Arrow-native driver at all, use
        ``bt.read.table("dbapi", module="psycopg", ...)``.

        A single query is a single stream, however large the table. To extract in
        parallel, name an indexed numeric column and its approximate range —
        ``partition_on="id", lower_bound=1, upper_bound=1_000_000, num_partitions=8`` —
        and the read becomes that many independent queries, one per worker. The spelling
        is Spark's JDBC reader's, and so are the semantics: **the bounds are cut points,
        not filters.** Rows outside them are still read (the first partition is unbounded
        below, the last unbounded above, and NULL keys ride in the first), so approximate
        bounds cost skew rather than rows.

        You can also hand it a connection you already have, exactly as
        ``pandas.read_sql(query, con)`` does — a PEP 249 connection or a SQLAlchemy
        ``Engine``/``Connection``, which is unwrapped for you. That reads through the
        DB-API path, so it stays on this process and cannot be partitioned; `uri=` is
        what scales out.

        Args:
            query: SQL text to execute, or ``None`` when reading via ``table=``.
            uri: A connection URI, e.g. ``"postgresql://user@host:5432/mydb"``.
            connection: An already-open PEP 249 connection or SQLAlchemy handle.
                Mutually exclusive with `uri`.
            opts: Further options — ``table=``, ``password=``, the partitioning
                keywords above, or any driver-specific keyword.

        Returns:
            A lazy `Dataset` over the query or table result.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.sql(  # doctest: +SKIP
                ...     "SELECT * FROM events WHERE country = 'US'",
                ...     uri="postgresql://localhost:5432/app",
                ... )

                >>> ds = bt.read.sql(  # doctest: +SKIP
                ...     "SELECT * FROM orders",
                ...     uri="postgresql://svc@warehouse:5432/shop",
                ...     password="env:PGPASSWORD",
                ... )
        """
        # Bound by name, not positionally: `ADBCSource`'s first field is `driver`, so a
        # positional `query` silently became the driver name. Same bug as `bigquery` below.
        if connection is not None:
            if uri is not None:
                from batcher._internal.errors import BackendError

                raise BackendError(
                    "pass either uri= or connection=, not both: they name two different "
                    "databases with no way to tell which you meant."
                )
            # `pandas.read_sql(query, con)` spelled here. A live connection can only be
            # used by this process, so it goes to the DB-API source, which reads on one
            # worker and refuses to be partitioned.
            return _read_table("dbapi", query=query, connection=connection, **opts)
        if uri is not None:
            from batcher.io.formats.sql.uri import parse_uri

            if parse_uri(uri).backend == "connectorx":
                # ConnectorX takes credentials *inside* its URI, so it gets the original
                # string rather than the password-stripped one `parse_uri` returns. It has
                # no separate password channel, and silently dropping one would leave the
                # user with an unauthenticated connection attempt and no idea why.
                if "password" in opts:
                    from batcher._internal.errors import BackendError

                    raise BackendError(
                        f"the {parse_uri(uri).scheme!r} backend has no separate password "
                        "channel: embed the credential in the URI "
                        "(mysql://user:pw@host/db), or keep the whole URI in a secret and "
                        "pass the reference directly to the backend — "
                        "bt.read.table('connectorx', query=..., conn_uri='env:MYSQL_URL'). "
                        "The reference cannot go in uri= here, which must be a real URI."
                    )
                return _read_table("connectorx", query=query, conn_uri=uri, **opts)
            return _read_table("adbc", query=query, uri=uri, **opts)
        return _read_table("adbc", query=query, **opts)

    def snowflake(self, query: str, **opts: Any) -> Dataset:
        """Read the result of a Snowflake SQL query, fetching result chunks in parallel as Arrow.

        Connection credentials are passed as keyword options.

        Args:
            query: SQL text to execute against Snowflake.
            opts: Connection credentials (account, user, warehouse, …) as keywords.

        Returns:
            A lazy `Dataset` over the Snowflake query result.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.snowflake(  # doctest: +SKIP
                ...     "SELECT * FROM sales.orders",
                ...     account="acme",
                ...     user="bob",
                ...     warehouse="wh",
                ... )
        """
        return _read_table("snowflake", query, **opts)

    def databricks(self, table: str, **opts: Any) -> Dataset:
        """Read a Databricks/Unity Catalog table by name.

        Uses credential vending to read the underlying Delta files directly.

        Args:
            table: Fully qualified Unity Catalog table name (``catalog.schema.table``).
            opts: Connection and credential options passed as keywords.

        Returns:
            A lazy `Dataset` over the Databricks table.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.databricks("main.sales.orders")  # doctest: +SKIP
        """
        return _read_table("databricks", table, **opts)

    def bigquery(self, query: str | None = None, **opts: Any) -> Dataset:
        """Read BigQuery via the Storage Read API as parallel Arrow streams.

        Supply a SQL ``query`` positionally, or ``table=`` to read a whole table.

        Args:
            query: SQL text to execute, or ``None`` when reading via ``table=``.
            opts: Project, credentials, and ``table=`` options passed as keywords.

        Returns:
            A lazy `Dataset` over the BigQuery query or table result.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.bigquery(  # doctest: +SKIP
                ...     "SELECT * FROM `project.dataset.events`", project="my-project"
                ... )
        """
        # Bind `query` by name. `BigQuerySource`'s first field is `project`, so passing it
        # positionally sent the SQL text into `project` — and a caller who also passed
        # `project=` (as they must) got "multiple values for argument 'project'" instead of
        # a query.
        if query is not None:
            opts["query"] = query
        return _read_table("bigquery", **opts)

    def clickhouse(self, query: str, **opts: Any) -> Dataset:
        """Read the result of a ClickHouse SQL query over the Arrow-native interface.

        Connection details are passed as keyword options.

        Args:
            query: SQL text to execute against ClickHouse.
            opts: Connection details (host, port, credentials) passed as keywords.

        Returns:
            A lazy `Dataset` over the ClickHouse query result.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.clickhouse(  # doctest: +SKIP
                ...     "SELECT * FROM events", host="localhost"
                ... )
        """
        return _read_table("clickhouse", query, **opts)

    # --- NoSQL -------------------------------------------------------------
    def mongo(self, **opts: Any) -> Dataset:
        """Read a MongoDB collection Arrow-natively via pymongoarrow.

        Pass connection, database, collection, and any query/projection as keyword options.

        Args:
            opts: Connection, ``database=``, ``collection=``, and query/projection keywords.

        Returns:
            A lazy `Dataset` over the MongoDB collection.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.mongo(  # doctest: +SKIP
                ...     uri="mongodb://localhost:27017",
                ...     database="app",
                ...     collection="events",
                ... )
        """
        return _read_table("mongo", **opts)

    def cassandra(self, **opts: Any) -> Dataset:
        """Read a Cassandra/Scylla table, fanning out across token-range splits for parallelism.

        Pass connection, keyspace, and table as keyword options.

        Args:
            opts: Connection (``contact_points=``), ``keyspace=``, and ``table=`` keywords.

        Returns:
            A lazy `Dataset` over the Cassandra table.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.cassandra(  # doctest: +SKIP
                ...     contact_points=["127.0.0.1"],
                ...     keyspace="app",
                ...     table="events",
                ... )
        """
        return _read_table("cassandra", **opts)

    def dynamodb(self, **opts: Any) -> Dataset:
        """Read a DynamoDB table using native parallel scan segments.

        Pass the table name and AWS connection options as keywords.

        Args:
            opts: ``table=`` name and AWS connection options passed as keywords.

        Returns:
            A lazy `Dataset` over the DynamoDB table.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.dynamodb(table="events", region="us-east-1")  # doctest: +SKIP
        """
        return _read_table("dynamodb", **opts)

    def elasticsearch(self, **opts: Any) -> Dataset:
        """Read an Elasticsearch index via ES|QL Arrow output (or a sliced scroll fallback).

        Pass the host, index, and query as keyword options.

        Args:
            opts: ``host=``, ``index=``, and query options passed as keywords.

        Returns:
            A lazy `Dataset` over the Elasticsearch index.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.elasticsearch(  # doctest: +SKIP
                ...     host="http://localhost:9200", index="events"
                ... )
        """
        return _read_table("elasticsearch", **opts)

    # --- Streaming ---------------------------------------------------------
    def kafka(
        self,
        topic: str,
        *,
        bootstrap_servers: str = "localhost:9092",
        group: str = "batcher",
        **opts: Any,
    ) -> Dataset:
        """Read a Kafka topic as an unbounded streaming source.

        Needs the optional extra: ``pip install 'batcher-engine[kafka]'``.

        Args:
            topic: The Kafka topic to consume.
            bootstrap_servers: Broker address(es), comma-separated.
            group: Consumer group id (offsets are committed under it).
            opts: Further consumer options passed through: ``partitions=`` to pin
                topic-partitions, ``max_offsets_per_trigger=`` and
                ``max_bytes_per_trigger=`` (the Spark spellings of ``poll_size`` and
                ``poll_bytes``) to bound one micro-batch by message count and by payload
                size, ``starting_offsets=`` (``"earliest"``, ``"latest"``, or a
                ``{partition: offset}`` map) for where a first run begins,
                ``fail_on_data_loss=`` for whether aged-out offsets stop the query,
                ``include_headers=True`` to add a ``headers`` column of
                ``array<struct<key:string,value:binary>>``, and anything else the
                ``confluent-kafka`` consumer accepts.

        Returns:
            A lazy `Dataset` streaming from the Kafka topic.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.kafka(  # doctest: +SKIP
                ...     "events", bootstrap_servers="localhost:9092"
                ... )
        """
        return _read_table("kafka", topic, bootstrap_servers=bootstrap_servers, group=group, **opts)

    def kinesis(self, stream_name: str, *, region: str = "us-east-1", **opts: Any) -> Dataset:
        """Read an AWS Kinesis stream as an unbounded source.

        Needs the optional extra: ``pip install 'batcher-engine[kinesis]'``.

        Args:
            stream_name: The Kinesis stream to consume.
            region: AWS region the stream lives in.
            opts: Further options (e.g. ``iterator_type=``) passed through.

        Returns:
            A lazy `Dataset` streaming from the Kinesis stream.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.kinesis("events", region="us-east-1")  # doctest: +SKIP
        """
        return _read_table("kinesis", stream_name, region=region, **opts)

    def pulsar(
        self,
        topic: str,
        *,
        service_url: str = "pulsar://localhost:6650",
        subscription: str = "batcher",
        **opts: Any,
    ) -> Dataset:
        """Read an Apache Pulsar topic as an unbounded streaming source.

        Needs the optional extra: ``pip install 'batcher-engine[pulsar]'``.

        Args:
            topic: The Pulsar topic to consume.
            service_url: The Pulsar broker service URL.
            subscription: The subscription name to consume under.
            opts: Further options passed through to the source.

        Returns:
            A lazy `Dataset` streaming from the Pulsar topic.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.pulsar("events")  # doctest: +SKIP
        """
        return _read_table(
            "pulsar", topic, service_url=service_url, subscription=subscription, **opts
        )

    def pubsub(self, topic: str, **opts: Any) -> Dataset:
        """Read a Google Cloud Pub/Sub subscription as an unbounded source.

        Needs the optional extra: ``pip install 'batcher-engine[pubsub]'``.

        Args:
            topic: The Pub/Sub subscription to consume.
            opts: Further options (e.g. ``project=``) passed through to the source.

        Returns:
            A lazy `Dataset` streaming from the Pub/Sub subscription.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.pubsub("projects/p/subscriptions/s")  # doctest: +SKIP
        """
        return _read_table("pubsub", topic, **opts)

    def eventhubs(
        self, topic: str, *, connection_str: str = "", consumer_group: str = "$Default", **opts: Any
    ) -> Dataset:
        """Read an Azure Event Hubs stream as an unbounded source.

        Uses the AMQP client (the ``eventhubs`` extra); Event Hubs also exposes a
        Kafka endpoint, so `read.kafka` works against it without the extra.

        Args:
            topic: The Event Hub name to consume.
            connection_str: The Event Hubs namespace connection string.
            consumer_group: The consumer group to read under.
            opts: Further options passed through to the source.

        Returns:
            A lazy `Dataset` streaming from the Event Hub.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.eventhubs(  # doctest: +SKIP
                ...     "events", connection_str="Endpoint=sb://..."
                ... )
        """
        return _read_table(
            "eventhubs", topic, connection_str=connection_str, consumer_group=consumer_group, **opts
        )

    def files_incremental(self, path: PathLike, file_format: str, **opts: Any) -> Dataset:
        """Incrementally discover and read newly arrived files under `path`.

        A Databricks Auto Loader analog: tracks already-seen files across runs. Under a
        streaming trigger the directory is re-listed every cadence, so a query started
        against it keeps ingesting whatever lands later.

        Backpressure is two bounds that compose, and a large backlog usually wants both:
        ``max_files_per_trigger=`` caps how many new files one micro-batch takes, and
        ``max_bytes_per_trigger=`` caps their total size. A file count alone is a poor
        proxy for the memory a batch needs when files range from kilobytes to gigabytes.

        Args:
            path: Directory or glob to watch for new files.
            file_format: Underlying format of those files (e.g. ``"parquet"``, ``"json"``).
            opts: ``state_dir=`` for the durable seen-file store,
                ``max_files_per_trigger=`` / ``max_bytes_per_trigger=`` for backpressure,
                and any further options forwarded to the underlying file reader.

        Returns:
            A lazy `Dataset` streaming rows from newly arrived files.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.files_incremental(  # doctest: +SKIP
                ...     "s3://bucket/incoming/", "parquet"
                ... )
        """
        return _read_table("files_incremental", path, file_format, **opts)

    def rate(self, rows_per_second: int = 1, **opts: Any) -> Dataset:
        """Generate ``(timestamp, value)`` rows at `rows_per_second` (Spark `rate`).

        A dev/benchmark source. Pass ``num_rows=`` to bound it (and ``pace=False`` to
        emit without the one-second cadence).

        Args:
            rows_per_second: Number of rows to emit per second.
            opts: ``num_rows=`` / ``pace=`` and other generator options as keywords.

        Returns:
            A lazy `Dataset` of generated ``(timestamp, value)`` rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.rate(  # doctest: +SKIP
                ...     rows_per_second=100, num_rows=1000, pace=False
                ... )
        """
        return _read_table("rate", rows_per_second, **opts)

    def rate_micro_batch(self, rows_per_batch: int = 1, **opts: Any) -> Dataset:
        """Generate exactly `rows_per_batch` rows per micro-batch (Spark ``rate-micro-batch``).

        The difference from `rate` is the unit, and it is the whole point. `rate` promises
        rows per *second*, so how many land in a micro-batch depends on how long the
        previous one took — which makes it useless as a benchmark input, because the thing
        being measured changes the input. This promises rows per *batch*, so a run is
        reproducible and a comparison between two builds is a comparison.

        Args:
            rows_per_batch: Rows in every micro-batch.
            opts: ``num_rows=`` to bound it, ``start_timestamp=`` and
                ``advance_ms_per_batch=`` to shape the event-time column so a windowed
                query over it is deterministic too.

        Returns:
            A lazy `Dataset` of generated ``(timestamp, value)`` rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> demo = bt.read.rate_micro_batch(3, num_rows=9)
                >>> [b.num_rows for b in demo.iter_batches()]
                [3, 3, 3]
        """
        return _read_table("rate_micro_batch", rows_per_batch, **opts)

    def socket(self, host: str = "localhost", port: int = 9999, **opts: Any) -> Dataset:
        """Read newline-delimited text from a TCP socket (Spark `socket`; dev only).

        Args:
            host: Hostname of the TCP source (default ``"localhost"``).
            port: TCP port to connect to (default 9999).
            opts: Additional socket-source options passed as keywords.

        Returns:
            A lazy `Dataset` streaming lines from the socket.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.socket(host="localhost", port=9999)  # doctest: +SKIP
        """
        return _read_table("socket", host, port, **opts)


read = Reader()
