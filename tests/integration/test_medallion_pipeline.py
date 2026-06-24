"""Workstream I — end-to-end medallion pipeline (bronze → silver → gold).

Composes the streaming features into the Databricks medallion architecture using
Parquet + the Auto-Loader incremental file source (no external dependency): a source
stream lands in **bronze**, an incremental read cleans/derives into **silver**, and
an incremental read aggregates into **gold** — every stage checkpointed for
exactly-once. (Transactional silver `MERGE`/SCD via Delta is an enhancement on top;
here the layer chaining itself is proven end to end.)
"""

from __future__ import annotations

import collections

import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.integration


def _run_pipeline(d) -> dict[int, int]:
    """Run bronze→silver→gold over `d` and return the gold {bucket: total}."""

    def p(*a: str) -> str:
        return str(d.joinpath(*a))

    # Bronze: a rate stream lands as Parquet, append-only, checkpointed.
    bt.read.rate(5, num_rows=20, pace=False).with_columns(bucket=col("value") % 4).write(
        p("bronze"), format="parquet", trigger=bt.Trigger.available_now(), checkpoint=p("ck_b")
    ).await_termination()

    # Silver: incrementally read new bronze files, filter/clean, land as Parquet.
    bt.read.files_incremental(p("bronze"), "parquet", state_dir=p("seen_s")).filter(
        col("value") >= 4
    ).write(
        p("silver"), format="parquet", trigger=bt.Trigger.available_now(), checkpoint=p("ck_s")
    ).await_termination()

    # Gold: incrementally read silver, aggregate, materialize (complete output).
    bt.read.files_incremental(p("silver"), "parquet", state_dir=p("seen_g")).group_by("bucket").agg(
        total=col("value").sum(), n=col("value").count()
    ).write.memory(
        "gold", trigger=bt.Trigger.available_now(), output_mode="complete", checkpoint=p("ck_g")
    ).await_termination()

    g = bt.read_memory("gold").to_pydict()
    if "bucket" not in g:  # a fully-drained rerun emits no new gold rows
        return {}
    return dict(zip(g["bucket"], g["total"], strict=True))


def test_medallion_bronze_silver_gold(tmp_path):
    got = _run_pipeline(tmp_path)
    # Oracle: values 0..19; silver keeps value>=4; gold sums value per bucket=value%4.
    oracle: dict[int, int] = collections.defaultdict(int)
    for v in range(4, 20):
        oracle[v % 4] += v
    assert got == dict(oracle)
    # Layer files are physically present and chain correctly.
    assert bt.read(str(tmp_path / "bronze"), format="parquet").count() == 20
    assert bt.read(str(tmp_path / "silver"), format="parquet").count() == 16  # value >= 4


def test_medallion_rerun_reprocesses_nothing(tmp_path):
    # Re-running the whole pipeline against the same checkpoints reprocesses nothing
    # new (each layer's source resumes from its committed position), so the persisted
    # bronze/silver layers are unchanged — exactly-once across the medallion.
    _run_pipeline(tmp_path)
    bronze1 = bt.read(str(tmp_path / "bronze"), format="parquet").count()
    silver1 = bt.read(str(tmp_path / "silver"), format="parquet").count()
    _run_pipeline(tmp_path)
    assert bt.read(str(tmp_path / "bronze"), format="parquet").count() == bronze1 == 20
    assert bt.read(str(tmp_path / "silver"), format="parquet").count() == silver1 == 16
