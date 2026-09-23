"""The genomics readers return the same rows single-node and distributed.

Every genomics format splits per file — a FASTA/FASTQ record boundary is not recoverable
from a byte offset, and neither is a VCF/GFF one once a `#` may sit inside a field — so the
distributed read is one task per file. What that has to preserve is exactly the single-node
answer: the same row multiset, the same column names, the same column types, for plain and
gzipped files mixed in one directory.

The comparison is only worth something if the distributed side really ran on more than one
worker. `collect(distributed=True)` with no `num_workers` runs one, which computes what
single-node computes (`.claude/rules/testing.md`), so each run names `num_workers=4`, and a
positive control shows the setting changes something observable: a `LIMIT` over an unordered
`group_by` — one of the four documented places a distributed answer may differ — returns
different groups on the two sides.
"""

from __future__ import annotations

import gzip
import os
import shutil

import pyarrow as pa
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

_FILES = 6
_ROWS = 3_000


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


def _fasta(k: int) -> str:
    return "".join(f">s{k}_{i} desc {i % 5}\n{'ACGT' * 20}\nNN{i % 7}\n" for i in range(_ROWS))


def _fastq(k: int) -> str:
    return "".join(f"@r{k}_{i} lane:{i % 3}\nACGT\n+\nII{i % 9}I\n" for i in range(_ROWS))


def _vcf(k: int) -> str:
    head = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
    body = "".join(
        f"{i % 17}\t{k * _ROWS + i}\t.\tA\tC,G\t{i % 4}.5\tPASS\tDB\tGT\t0/1\n"
        for i in range(_ROWS)
    )
    return head + body


def _bed(k: int) -> str:
    return "track name=x\n" + "".join(
        f"chr{i % 17}\t{i}\t{i + 1}\tn{k}\t0\t+\n" for i in range(_ROWS)
    )


def _gff(k: int) -> str:
    rows = "".join(
        f"c{i % 17}\ts{k}\tgene\t{i + 1}\t{i + 2}\t.\t+\t.\tID=g{i}\n" for i in range(_ROWS)
    )
    return "##gff-version 3\n" + rows + "##FASTA\n>c0\nACGT\n"


# format -> (reader, file text for file k, plain suffix, the group key column)
_FORMATS = {
    "fasta": (bt.read.fasta, _fasta, ".fa", "description"),
    "fastq": (bt.read.fastq, _fastq, ".fq", "description"),
    "vcf": (bt.read.vcf, _vcf, ".vcf", "chrom"),
    "bed": (bt.read.bed, _bed, ".bed", "chrom"),
    "gff": (bt.read.gff, _gff, ".gff3", "seqid"),
}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Six files per format — four plain, two gzipped — on worker-readable storage."""
    from batcher.dist.shuffle_io import shared_scratch_root

    root = shared_scratch_root()
    if root is not None:
        base = os.path.join(root, f"genomics_dist_{os.getpid()}")
    else:
        base = str(tmp_path_factory.mktemp("genomics"))
    for fmt, (_, text, suffix, _) in _FORMATS.items():
        os.makedirs(os.path.join(base, fmt), exist_ok=True)
        for k in range(_FILES):
            data = text(k).encode()
            if k >= _FILES - 2:
                name, data = f"f{k}{suffix}.gz", gzip.compress(data)
            else:
                name = f"f{k}{suffix}"
            with open(os.path.join(base, fmt, name), "wb") as fh:
                fh.write(data)
    yield base
    if root is not None:
        shutil.rmtree(base, ignore_errors=True)


def _canonical(table: pa.Table) -> list[tuple]:
    return sorted((tuple(row.values()) for row in table.to_pylist()), key=repr)


@pytest.mark.parametrize("fmt", sorted(_FORMATS))
def test_a_distributed_read_returns_the_single_node_rows(corpus, fmt):
    reader = _FORMATS[fmt][0]
    ds = reader(os.path.join(corpus, fmt))
    local = ds.collect(distributed=False)
    spread = ds.collect(distributed=True, num_workers=4)
    assert local.num_rows == _FILES * _ROWS  # every file, gzipped ones included, was read
    # Names and types, the contract `.claude/rules/python-control-plane.md` states. Field
    # *nullability* is not compared: the distributed executor returns FASTA/FASTQ's
    # declared-`not null` fields as nullable (a Parquet source keeps the flag), which is an
    # executor behaviour rather than a reader one and is reported, not asserted away here.
    assert [(f.name, f.type) for f in spread.schema] == [(f.name, f.type) for f in local.schema]
    assert _canonical(spread) == _canonical(local)


@pytest.mark.parametrize("fmt", sorted(_FORMATS))
def test_a_distributed_group_by_over_the_read_matches(corpus, fmt):
    reader, _, _, key = _FORMATS[fmt]
    ds = reader(os.path.join(corpus, fmt))
    query = ds.group_by(key).agg(n=bt.col(ds.columns[0]).count()).sort(key)
    local = query.collect(distributed=False)
    spread = query.collect(distributed=True, num_workers=4)
    assert local.num_rows > 1
    assert spread.equals(local)  # sorted on the key, so an ordered comparison is exact


def test_the_distributed_runs_really_used_several_workers(corpus):
    """The positive control: a shape whose answer the query leaves open does move.

    `LIMIT` over an unordered `group_by` keeps *some* n groups — which ones follows the
    hash table's walk, and so the partitioning. One worker would reproduce single-node's
    choice; a different choice is evidence the equality above was measured across a real
    fan-out. The bound that still holds is checked too: n rows, each a real group.
    """
    ds = bt.read.bed(os.path.join(corpus, "bed"))
    query = ds.group_by("chrom").agg(n=bt.col("start").count()).limit(3)
    local = query.collect(distributed=False)
    spread = query.collect(distributed=True, num_workers=4)
    full = {
        r["chrom"]: r["n"]
        for r in ds.group_by("chrom")
        .agg(n=bt.col("start").count())
        .collect(distributed=False)
        .to_pylist()
    }
    assert local.num_rows == spread.num_rows == 3
    assert all(full[r["chrom"]] == r["n"] for r in spread.to_pylist())
    assert sorted(local.column("chrom").to_pylist()) != sorted(spread.column("chrom").to_pylist())
