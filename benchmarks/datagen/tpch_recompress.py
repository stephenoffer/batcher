"""Copy the public TPC-H Parquet into our own bucket, recompressed, with nothing else changed.

The public copy (`s3://ray-benchmark-data/tpch/parquet`) is Snappy with ~122k-row row
groups. A distributed scan of it is bound by each node's network link, not by S3 or the
CPU (measured on 16-core nodes: ~1.1 GB/s per node, the link rate), so the one lever on a
cold read is the number of bytes. Zstd level 3 with 1M-row row groups stores a `lineitem`
file in 178 MB instead of 295 MB, and the columns a scan actually projects shrink by about
a third (q1's seven: ~90 -> ~58 MB per file), while decoding no slower.

**Only the encoding changes.** Every file keeps its name, its row order, its schema (the
positional ``columnNN`` names and the trailing empty column included), so each engine reads
the copy through exactly the code that reads the original -- Ray's release scripts need only
their base URI changed -- and the file-to-key-range layout the original has is preserved.
Comparing engines is only fair when they read the same copy; say which one a number used.

A file already present at the destination with the source's row count is skipped, so a run
interrupted by a cluster restart resumes where it stopped.

Run:
    python benchmarks/datagen/tpch_recompress.py --scales 1,10,100,1000 \
        --dest "$ANYSCALE_ARTIFACT_STORAGE/tpch_zstd"
"""

from __future__ import annotations

import argparse
import os
import time

SOURCE = "s3://ray-benchmark-data/tpch/parquet"
TABLES = ("region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem")
ROW_GROUP_ROWS = 1 << 20
ZSTD_LEVEL = 3


def _bare(uri: str) -> str:
    return uri.removeprefix("s3://")


def _convert(src: str, dst: str) -> tuple[str, int, int, str]:
    """Rewrite one file; returns (dst, rows, bytes written, "copied" | "skipped")."""
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq

    fs = pafs.S3FileSystem(region="us-west-2")
    rows = pq.ParquetFile(_bare(src), filesystem=fs).metadata.num_rows
    try:
        done = pq.ParquetFile(_bare(dst), filesystem=fs).metadata.num_rows
        if done == rows:
            return dst, rows, 0, "skipped"
    except (OSError, FileNotFoundError):
        pass
    table = pq.read_table(_bare(src), filesystem=fs)
    with fs.open_output_stream(_bare(dst)) as out:
        pq.write_table(
            table,
            out,
            compression="zstd",
            compression_level=ZSTD_LEVEL,
            row_group_size=ROW_GROUP_ROWS,
        )
    written = pq.ParquetFile(_bare(dst), filesystem=fs).metadata
    if written.num_rows != rows:
        raise RuntimeError(f"{dst}: wrote {written.num_rows} rows, source has {rows}")
    return dst, rows, fs.get_file_info(_bare(dst)).size, "copied"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scales", default="1,10,100,1000")
    p.add_argument("--tables", default=",".join(TABLES))
    p.add_argument("--source", default=SOURCE)
    p.add_argument("--dest", default=f"{os.environ.get('ANYSCALE_ARTIFACT_STORAGE', '')}/tpch_zstd")
    args = p.parse_args()
    if not args.dest.startswith("s3://"):
        raise SystemExit("--dest must be an s3:// URI")

    import pyarrow.fs as pafs
    import ray

    ray.init(address="auto", logging_level="ERROR", log_to_driver=False)
    convert = ray.remote(num_cpus=1, memory=3 << 30, max_retries=3)(_convert)
    fs = pafs.S3FileSystem(region="us-west-2")
    jobs = []
    for sf in args.scales.split(","):
        for table in args.tables.split(","):
            listed = fs.get_file_info(pafs.FileSelector(_bare(f"{args.source}/sf{sf}/{table}")))
            for info in listed:
                if info.path.endswith(".parquet"):
                    name = info.path.rsplit("/", 1)[1]
                    jobs.append((f"s3://{info.path}", f"{args.dest}/sf{sf}/{table}/{name}"))
    # Largest first, so the long lineitem files are not the tail of the run.
    jobs.sort(key=lambda j: ("lineitem" not in j[0], "orders" not in j[0]))
    print(f"{len(jobs)} files -> {args.dest}", flush=True)
    t0 = time.time()
    refs = [convert.remote(src, dst) for src, dst in jobs]
    copied = skipped = written = 0
    for i, ref in enumerate(refs, 1):
        _dst, _rows, nbytes, status = ray.get(ref)
        copied += status == "copied"
        skipped += status == "skipped"
        written += nbytes
        if i % 200 == 0 or i == len(refs):
            print(
                f"{i}/{len(refs)} done, {written / 1e9:.1f} GB written, {time.time() - t0:.0f}s",
                flush=True,
            )
    gb, secs = written / 1e9, time.time() - t0
    print(f"copied {copied}, already present {skipped}, {gb:.1f} GB in {secs:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
