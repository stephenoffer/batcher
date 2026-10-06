"""A ported write keeps the source engine's save mode, or stops the migration where it cannot.

Batcher's file writes default to `mode="overwrite"`. Spark's `DataFrameWriter` defaults to
`errorifexists`, and Daft's and Ray Data's file writes default to append. A codemod that spells a
write as `ds.write.<format>(...)` with no `mode=` therefore inherits Batcher's default and turns
a job that used to refuse an existing destination (or add to it) into one that deletes it. Every
ported write must either carry an explicit `mode=` that means what the source meant, or be left
in the source engine's spelling with a marker, which fails on a Batcher `Dataset` instead of
writing anything.
"""

from __future__ import annotations

import re
import textwrap

import pytest

pytest.importorskip("libcst")

from batcher.migrate.engines import tables
from batcher.migrate.templates import Signature
from batcher.migrate.translate import translate

_SPARK = """
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet("in")
"""
_DAFT = 'import daft\ndf = daft.read_parquet("in")\n'
_RAY = 'import ray\nds = ray.data.read_parquet("in")\n'
_PRELUDES = {"pyspark": (_SPARK, "df.write."), "daft": (_DAFT, "df."), "ray_data": (_RAY, "ds.")}


def _port(engine: str, body: str) -> tuple[list[str], list[object]]:
    prelude, _ = _PRELUDES[engine]
    code, report = translate(textwrap.dedent(prelude) + textwrap.dedent(body), engine)
    lines = [ln for ln in code.splitlines() if ln and not ln.lstrip().startswith("#")]
    return lines, report.sites  # type: ignore[attr-defined]


def _marked(sites: list[object], spelling: str) -> bool:
    return any(s.spelling == spelling and s.action == "marked" for s in sites)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('df.write.parquet("a")\n', 'df.write.parquet("a", mode="error")'),
        ('df.write.text("d")\n', 'df.write.text("d", mode="error")'),
        ('df.write.xml("e")\n', 'df.write.xml("e", mode="error")'),
        ('df.write.csv("g")\n', 'df.write.csv("g", mode="error")'),
        # The parquet template declines `compression=`; the call left as written still runs on
        # Batcher, so it must not run with Batcher's default.
        (
            'df.write.parquet("p", compression="snappy")\n',
            'df.write.parquet("p", compression="snappy", mode="error")',
        ),
        ('df.write.saveAsTable("t")\n', 'df.write.table("t", mode="error")'),
    ],
)
def test_a_spark_write_without_a_mode_ports_to_errorifexists(body: str, expected: str) -> None:
    lines, _ = _port("pyspark", body)
    assert lines[-1] == expected


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('df.write.mode("append").parquet("b")\n', 'df.write.parquet("b", mode="append")'),
        ('df.write.parquet("c", mode="overwrite")\n', 'df.write.parquet("c", mode="overwrite")'),
        ('df.write.csv("g", mode="ignore")\n', 'df.write.csv("g", mode="ignore")'),
        # A mode passed by position is still the caller's; no default is stacked on top.
        ('df.write.csv("g", "append")\n', 'df.write.csv("g", "append")'),
        ('df.write.json("h", mode="append")\n', 'df.write.json("h", mode="append")'),
        ('df.write.saveAsTable("t", mode="overwrite")\n', 'df.write.table("t", mode="overwrite")'),
    ],
)
def test_an_explicit_spark_mode_is_preserved(body: str, expected: str) -> None:
    lines, _ = _port("pyspark", body)
    assert lines[-1] == expected


@pytest.mark.parametrize(
    ("body", "spelling", "kept"),
    [
        # `insertInto` appends to an existing table and `writeTo` is a builder: a 1:1
        # `write.table(name)` would run a write at once, under a different mode.
        ('df.write.insertInto("t")\n', "DataFrameWriter.insertInto", "insertInto"),
        ('df.writeTo("t").append()\n', "DataFrame.writeTo", "writeTo"),
        ('df.writeTo("t").createOrReplace()\n', "DataFrame.writeTo", "writeTo"),
    ],
)
def test_a_spark_write_whose_intent_cannot_be_spelled_is_blocked(
    body: str, spelling: str, kept: str
) -> None:
    lines, sites = _port("pyspark", body)
    assert kept in lines[-1]
    assert _marked(sites, spelling)
    assert ".write.table(" not in lines[-1]


@pytest.mark.parametrize(
    ("body", "spelling"),
    [
        ('df.write_parquet("a")\n', "DataFrame.write_parquet"),
        ('df.write_csv("b")\n', "DataFrame.write_csv"),
        ('df.write_deltalake("c")\n', "DataFrame.write_deltalake"),
        ("df.write_iceberg(tbl)\n", "DataFrame.write_iceberg"),
        ('df.write_lance("u")\n', "DataFrame.write_lance"),
    ],
)
def test_a_daft_write_without_a_mode_is_blocked(body: str, spelling: str) -> None:
    lines, sites = _port("daft", body)
    assert lines[-1] == body.strip()
    assert _marked(sites, spelling)


def test_an_explicit_daft_mode_is_preserved() -> None:
    lines, _ = _port("daft", 'df.write_deltalake("c", mode="overwrite")\n')
    assert lines[-1] == 'df.write.delta("c", mode="overwrite")'
    lines, _ = _port("daft", 'df.write_deltalake("c", mode="append")\n')
    assert lines[-1] == 'df.write.delta("c", mode="append")'


@pytest.mark.parametrize(
    ("body", "spelling"),
    [
        ('ds.write_parquet("a")\n', "Dataset.write_parquet"),
        ('ds.write_numpy("c", column="x")\n', "Dataset.write_numpy"),
        ('ds.write_lance("u")\n', "Dataset.write_lance"),
        ('ds.write_iceberg("t")\n', "Dataset.write_iceberg"),
        ('ds.write_tfrecords("p")\n', "Dataset.write_tfrecords"),
        # Ray spells a mode as its own `SaveMode` enum; nothing proves what Batcher makes of it.
        ('ds.write_lance("u", mode=SaveMode.APPEND)\n', "Dataset.write_lance"),
    ],
)
def test_a_ray_write_without_a_literal_mode_is_blocked(body: str, spelling: str) -> None:
    lines, sites = _port("ray_data", body)
    assert lines[-1] == body.strip()
    assert _marked(sites, spelling)


def test_an_explicit_ray_mode_is_preserved() -> None:
    lines, _ = _port("ray_data", 'ds.write_iceberg("t", mode="append")\n')
    assert lines[-1] == 'ds.write.iceberg("t", mode="append")'


_WRITE_CALL = re.compile(r"\.write\.(\w+)\(")


@pytest.mark.parametrize("engine", sorted(_PRELUDES))
def test_no_ported_write_runs_on_batcher_without_an_explicit_mode(engine: str) -> None:
    """Every registry write, called the plainest way, ports with `mode=` or not at all."""
    t = tables(engine)
    _, head = _PRELUDES[engine]
    surface = {"pyspark": "DataFrameWriter", "daft": "DataFrame", "ray_data": "Dataset"}[engine]
    writer = set(t.batcher_params["Dataset.write"])
    calls = []
    for (row_surface, name), row in sorted(t.rows.items()):
        if row_surface != surface or not any(b.startswith("Dataset.write.") for b in row.batcher):
            continue
        tokens = t.params.get(surface, {}).get(name)
        if tokens is None:
            continue
        sig = Signature.parse(tokens)
        required = [n for n, has_default in sig.positional if not has_default]
        calls.append(f"{head}{name}({', '.join(repr(f'v{i}') for i in range(len(required)))})")
    assert len(calls) >= 5, "the sweep found too few writes to mean anything"
    lines, _ = _port(engine, "".join(f"{c}\n" for c in calls))
    runnable = [ln for ln in lines if (m := _WRITE_CALL.search(ln)) and m.group(1) in writer]
    assert runnable or engine != "pyspark", "the sweep proves nothing if no write was ported"
    for line in runnable:
        assert "mode=" in line, f"{engine}: {line!r} runs with Batcher's default save mode"
