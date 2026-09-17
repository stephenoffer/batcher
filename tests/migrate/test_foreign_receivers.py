"""The foreign directions rewrite only what the inference proves belongs to the source engine.

Both halves matter, as in the canonical rename. A missed rewrite leaves foreign code behind; a
wrong one changes a pandas call, a Python list method, or the body of a user function that is
handed a foreign frame, and the migrated script silently computes something else.
"""

from __future__ import annotations

import textwrap

import pytest

pytest.importorskip("libcst")

from batcher.migrate.translate import translate


def _translate(source: str, engine: str) -> tuple[str, object]:
    return translate(textwrap.dedent(source), engine)


def test_pandas_and_python_objects_sharing_a_name_are_left_alone() -> None:
    out, _ = _translate(
        """
        import pandas as pd
        import polars as pl

        frame = pl.DataFrame({"a": [1]})
        table = pd.DataFrame({"a": [1]})
        kept = frame.head(3)
        other = table.head(3)
        names = ["a"]
        names.sort()
        """,
        "polars",
    )
    assert "kept = frame.limit(3)" in out
    assert "other = table.head(3)" in out
    assert "names.sort()" in out


def test_each_import_style_seeds_the_engine() -> None:
    out, _ = _translate(
        """
        from pyspark.sql.functions import col, upper
        from pyspark.sql import functions as F
        import pyspark.sql.functions as G

        a = upper(col("x"))
        b = F.upper(F.col("x"))
        c = G.upper(G.col("x"))
        """,
        "pyspark",
    )
    assert out.count('bt.col("x").str.upper()') == 3
    assert "pyspark" not in out


def test_a_receiver_the_inference_cannot_type_is_marked_not_guessed() -> None:
    out, report = _translate(
        """
        import daft

        def load():
            return make_frame()

        frame = load()
        out = frame.where(daft.col("x") > 1)
        """,
        "daft",
    )
    assert 'frame.where(bt.col("x") > 1)' in out  # the expression inside is still translated
    assert "# batcher-migrate: Daft `.where` is called on a receiver" in out
    assert any(s.action == "marked" and s.spelling == ".where" for s in report.sites)


def test_code_inside_a_lambda_keeps_the_source_engine() -> None:
    out, _ = _translate(
        """
        import polars as pl

        lf = pl.LazyFrame({"x": [1]})
        batched = lf.map_batches(lambda frame: frame.with_columns(pl.col("x") + 1))
        """,
        "polars",
    )
    assert 'lambda frame: frame.with_columns(pl.col("x") + 1)' in out
    assert "import polars as pl" in out  # still needed, so still imported


def test_a_spark_session_and_window_spec_are_absorbed_and_removed() -> None:
    out, _ = _translate(
        """
        from pyspark.sql import SparkSession, Window
        from pyspark.sql import functions as F

        spark = SparkSession.builder.appName("x").getOrCreate()
        df = spark.read.parquet("p")
        w = Window.partitionBy("g")
        out = df.withColumn("n", F.count("*").over(w))
        """,
        "pyspark",
    )
    assert "SparkSession" not in out and "Window" not in out and "spark" not in out
    assert 'out = df.with_columns(n=bt.count().over(partition_by=["g"]))' in out
