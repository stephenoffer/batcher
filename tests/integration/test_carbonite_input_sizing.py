"""The out-of-core route must depend on the data's size, not on the file format.

The in-memory path resolves every source to Arrow *before* the engine runs, so a scan-heavy
query's dominant memory term is the input, not the operator state. Carbonite guards that
with `input_exceeds_budget`, fed by `projected_input_bytes` — and that estimate used to
require an **exact** row count.

Parquet carries one in its footer. CSV and JSON do not. So the identical query over the
identical rows was protected in one format and unbounded in the other, and the difference
was invisible: the CSV query simply ran, until the day the file was large enough that it
did not. The pressure signal cannot rescue it either, because the pressure check happens
*before* the input is read, when the process is by construction still empty.

An estimated row count is the right input here. The two failure modes are not symmetric:
over-estimating routes a query out of core and costs latency, having no estimate leaves it
in memory and costs the process.
"""

from __future__ import annotations

import csv
import json

import pytest

import batcher as bt
import batcher.dist.spill as spill_mod
from batcher.api.orchestration.sizing import projected_input_bytes
from batcher.config import Config, MemoryConfig, config_context

pytestmark = pytest.mark.integration

_ROWS = 400_000
# 400k rows of (string key, int64) resolve to ~17 MiB of Arrow, so a 16 MiB envelope is
# genuinely exceeded while keeping the fixture small enough to build quickly.
_ENVELOPE = 16 << 20


@pytest.fixture
def routed_out_of_core(monkeypatch):
    """Whether the query took the out-of-core executor. Returns a one-key dict."""
    seen = {"n": 0}
    real = spill_mod.spill_collect

    def traced(*args, **kwargs):
        seen["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(spill_mod, "spill_collect", traced)
    return seen


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """The same rows written as CSV, JSON, and Parquet."""
    d = tmp_path_factory.mktemp("input_sizing")
    csv_path = d / "data.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "v"])
        for i in range(_ROWS):
            w.writerow([f"key{i % 2000}", i])

    json_path = d / "data.jsonl"
    with json_path.open("w") as f:
        for i in range(_ROWS):
            f.write(json.dumps({"k": f"key{i % 2000}", "v": i}) + "\n")

    parquet_path = d / "data.parquet"
    bt.read.csv(str(csv_path)).write.parquet(str(parquet_path))
    return {"csv": str(csv_path), "json": str(json_path), "parquet": str(parquet_path)}


def _reader(fmt: str, path: str):
    return {"csv": bt.read.csv, "json": bt.read.json, "parquet": bt.read.parquet}[fmt](path)


def test_a_footerless_format_can_size_its_own_input(files) -> None:
    """The estimate must be non-zero for a format that declares no exact row count.

    `0` is the "I cannot tell" answer, and `input_exceeds_budget` reads it as "no evidence
    of not fitting" — which is how the guard came to be silently format-dependent.
    """
    for fmt in ("csv", "json", "parquet"):
        sources = list(_reader(fmt, files[fmt])._sources)
        estimate = projected_input_bytes(sources, {})
        assert estimate > 0, f"{fmt} sources cannot size themselves, so the guard is blind"


def test_the_estimate_is_in_the_right_order_of_magnitude(files) -> None:
    """An estimate only has to be right enough to make the routing decision."""
    parquet = projected_input_bytes(list(_reader("parquet", files["parquet"])._sources), {})
    for fmt in ("csv", "json"):
        estimate = projected_input_bytes(list(_reader(fmt, files[fmt])._sources), {})
        assert parquet / 20 <= estimate <= parquet * 20, (
            f"{fmt} estimate {estimate} is not comparable to parquet's {parquet}"
        )


@pytest.mark.parametrize("fmt", ["csv", "json", "parquet"])
def test_every_format_routes_out_of_core_under_a_tight_envelope(
    fmt, files, routed_out_of_core
) -> None:
    """The routing decision must follow the data, not the format's metadata richness."""
    cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=_ENVELOPE))
    with config_context(cfg):
        table = _reader(fmt, files[fmt]).group_by("k").agg(total=bt.col("v").sum()).collect()

    assert table.num_rows == 2000
    assert routed_out_of_core["n"] >= 1, (
        f"{fmt} resolved its whole input in memory under a {_ENVELOPE >> 20} MiB envelope"
    )


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_the_out_of_core_answer_is_unchanged(fmt, files) -> None:
    """Routing is a memory strategy; the rows must be identical either way."""
    with config_context(Config()):
        roomy = _reader(fmt, files[fmt]).group_by("k").agg(total=bt.col("v").sum()).collect()
    tight = Config().replace(memory=MemoryConfig(max_memory_bytes=_ENVELOPE))
    with config_context(tight):
        bounded = _reader(fmt, files[fmt]).group_by("k").agg(total=bt.col("v").sum()).collect()

    assert bounded.sort_by("k").to_pydict() == roomy.sort_by("k").to_pydict()


def test_a_roomy_envelope_still_keeps_the_in_memory_fast_path(files, routed_out_of_core) -> None:
    """The estimate must not route a query that comfortably fits."""
    with config_context(Config().replace(memory=MemoryConfig(max_memory_bytes=8 << 30))):
        table = _reader("csv", files["csv"]).group_by("k").agg(total=bt.col("v").sum()).collect()
    assert table.num_rows == 2000
    assert routed_out_of_core["n"] == 0, "a query that fits was pushed out of core"
