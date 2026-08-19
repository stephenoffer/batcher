"""The ``from_*`` constructors, against the frameworks they convert from.

``bt.from_duckdb``, ``from_spark``, ``from_dask``, ``from_torch`` and ``from_tf`` are the
front door for anyone arriving with data already in another system, and none of them had a
test. They are also exactly the surface where a silent defect is expensive: a conversion
that drops the last partition, loses a null, or widens a column type produces a dataset
that looks right and is not.

So each test asserts three things rather than one: the **values** match the source frame,
the **column names** match, and the **types** are the ones Batcher's documented boundary
normalization produces (narrow integers and ``float32`` widen once at the FFI edge, so a
``torch.float32`` tensor arrives as ``double`` -- that is the contract, and pinning it here
is what stops a future change to the boundary from silently altering every conversion).

Frameworks absent from the environment skip rather than fail; the point of the module is
that the ones present are actually exercised.
"""

from __future__ import annotations

import os
import shutil

import pytest

import batcher as bt

pytestmark = pytest.mark.differential

#: Deliberately includes a null, a negative, and a string column, since a conversion that
#: goes through a dense numeric buffer loses exactly those.
ROWS = {"i": [1, 2, 3, -4], "f": [1.5, None, 3.25, -0.5], "s": ["a", "b", None, "d"]}


def _schema(ds: bt.Dataset) -> dict[str, str]:
    """Column name to Arrow type name, for comparing shapes rather than values."""
    schema = ds.schema
    return {name: str(schema.field(name).type) for name in schema.names}


def test_from_duckdb_accepts_a_relation_a_connection_and_a_query():
    """All three documented spellings must land on the same rows."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE t (i BIGINT, f DOUBLE, s VARCHAR)")
    con.executemany(
        "INSERT INTO t VALUES (?, ?, ?)",
        [(ROWS["i"][k], ROWS["f"][k], ROWS["s"][k]) for k in range(4)],
    )
    want = con.execute("SELECT i, f, s FROM t ORDER BY i").fetchall()

    from_relation = bt.from_duckdb(con.sql("SELECT i, f, s FROM t"))
    from_query = bt.from_duckdb(con, "SELECT i, f, s FROM t")
    for ds in (from_relation, from_query):
        got = ds.sort("i").to_pydict()
        assert sorted(got) == ["f", "i", "s"]
        assert list(zip(got["i"], got["f"], got["s"], strict=True)) == want
        assert _schema(ds) == {"i": "int64", "f": "double", "s": "string"}


def test_from_duckdb_preserves_nulls_rather_than_filling_them():
    """The failure mode a dense conversion has: a null becomes a zero or an empty string."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    got = bt.from_duckdb(
        con.sql("SELECT * FROM (VALUES (1, NULL), (NULL, 'x')) AS v(i, s)")
    ).to_pydict()
    assert None in got["i"]
    assert None in got["s"]
    assert 0 not in got["i"], "a null integer must not arrive as zero"
    assert "" not in got["s"], "a null string must not arrive as empty"


def test_from_torch_converts_a_tensor_a_tuple_and_a_dataset():
    """Every documented input shape, with the leading axis read as the row axis."""
    torch = pytest.importorskip("torch")

    flat = bt.from_torch(torch.tensor([1.0, 2.0, 3.0])).to_pydict()
    assert flat == {"data": [1.0, 2.0, 3.0]}, "a 1-D tensor is a scalar column"

    # An (n, dim) tensor is n rows of a fixed-size list -- the convention
    # `bt.from_numpy` documents for embeddings, not one row holding a matrix.
    matrix = bt.from_torch(torch.tensor([[1.0, 2.0], [3.0, 4.0]])).to_pydict()
    assert matrix == {"data": [[1.0, 2.0], [3.0, 4.0]]}

    pair = bt.from_torch((torch.tensor([1, 2, 3]), torch.tensor([0.5, 1.5, 2.5])))
    assert pair.to_pydict() == {"col_0": [1, 2, 3], "col_1": [0.5, 1.5, 2.5]}

    stacked = torch.utils.data.TensorDataset(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([0, 1])
    )
    assert bt.from_torch(stacked).to_pydict() == {
        "col_0": [[1.0, 2.0], [3.0, 4.0]],
        "col_1": [0, 1],
    }


def test_from_torch_and_from_numpy_agree_on_a_per_row_vector():
    """One convention for a vector feature, whichever constructor produced it."""
    torch = pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")
    rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    through_torch = bt.from_torch(torch.tensor(rows))
    through_numpy = bt.from_numpy(numpy.array(rows))
    assert through_torch.to_pydict() == through_numpy.to_pydict()
    assert _schema(through_torch) == _schema(through_numpy), (
        "the two doors into the engine must give a vector feature the same type"
    )


def test_from_torch_widens_float32_to_double_at_the_boundary():
    """The documented normalization, pinned so a boundary change cannot pass unnoticed.

    Narrow numerics widen once at the FFI edge (Int8/16/32 to Int64, Float16/32 to
    Float64). A ``torch.float32`` tensor therefore arrives as ``double``, and the values
    must be the ``float32`` values exactly -- widening must not resample.
    """
    torch = pytest.importorskip("torch")
    values = [1.5, 2.25, -0.125]
    ds = bt.from_torch(torch.tensor(values, dtype=torch.float32))
    assert set(_schema(ds).values()) == {"double"}
    got = next(iter(ds.to_pydict().values()))
    assert got == values, "widening changed a value that float32 represents exactly"

    narrow = bt.from_torch(torch.tensor([1, 2, 3], dtype=torch.int32))
    assert set(_schema(narrow).values()) == {"int64"}


def test_from_tf_materializes_a_tf_data_dataset():
    """A ``tf.data.Dataset`` of scalar features, in order, with its rows intact."""
    tf = pytest.importorskip("tensorflow")
    source = tf.data.Dataset.from_tensor_slices({"i": [1, 2, 3], "f": [0.5, 1.5, 2.5]})
    got = bt.from_tf(source).to_pydict()
    assert sorted(got) == ["f", "i"]
    assert got["i"] == [1, 2, 3], "the conversion must not reorder or drop rows"
    assert got["f"] == [0.5, 1.5, 2.5]


def test_from_tf_accepts_a_vector_feature():
    """A per-row vector becomes a list column, as it does through every other door.

    This is the shape an embedding or a feature vector arrives in, and before the rank
    rules were shared with ``from_numpy`` it raised ``ArrowInvalid: only handle
    1-dimensional arrays`` -- so ``from_tf`` accepted scalar features only.
    """
    tf = pytest.importorskip("tensorflow")
    source = tf.data.Dataset.from_tensor_slices({"x": [[1.0, 2.0], [3.0, 4.0]], "y": [0, 1]})
    got = bt.from_tf(source).to_pydict()
    assert got["x"] == [[1.0, 2.0], [3.0, 4.0]]
    assert got["y"] == [0, 1]


def test_from_tf_refuses_a_batched_dataset_and_says_what_to_do():
    """One element is one row, so a batched dataset would make each batch a row.

    Every value would still be present and every row count would be wrong, which is the
    worst kind of conversion bug. It is refused with a typed error naming ``.unbatch()``
    rather than reshaped, and rather than surfacing the raw ``pyarrow`` failure this
    produced several frames below the public API.
    """
    tf = pytest.importorskip("tensorflow")
    from batcher import PlanError

    source = tf.data.Dataset.from_tensor_slices({"i": list(range(7))}).batch(3)
    with pytest.raises(PlanError, match="unbatch"):
        bt.from_tf(source)
    assert bt.from_tf(source.unbatch()).to_pydict()["i"] == list(range(7))


def test_from_spark_collects_a_dataframe_through_arrow():
    """A Spark ``DataFrame``, compared against the rows Spark itself reports."""
    pytest.importorskip("pyspark")
    if not (os.environ.get("JAVA_HOME") or shutil.which("java")):
        pytest.skip("PySpark needs a JVM, and this environment has none")
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("batcher-interop-test")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    try:
        frame = spark.createDataFrame(
            [(ROWS["i"][k], ROWS["f"][k], ROWS["s"][k]) for k in range(4)], ["i", "f", "s"]
        )
        want = sorted((row["i"], row["f"], row["s"]) for row in frame.collect())
        got = bt.from_spark(frame).sort("i").to_pydict()
        assert sorted(got) == ["f", "i", "s"]
        assert sorted(zip(got["i"], got["f"], got["s"], strict=True)) == want
        assert None in got["f"] and None in got["s"], "nulls must survive the collect"
    finally:
        spark.stop()


def test_from_dask_streams_one_batch_per_partition():
    """The partition-to-batch mapping the docstring promises, and no lost partition."""
    dask = pytest.importorskip("dask.dataframe")
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(ROWS)
    ddf = dask.from_pandas(frame, npartitions=3)
    ds = bt.from_dask(ddf)
    got = ds.sort("i").to_pydict()
    assert sorted(got["i"]) == sorted(ROWS["i"]), "a partition was dropped"
    assert len(list(ds.iter_batches())) >= 1


def test_every_available_constructor_produces_a_queryable_dataset():
    """A conversion is only useful if the result plans and executes like any other.

    Each converted dataset is put through a filter, a projection and an aggregate, because
    a constructor that returns something merely *shaped* like a ``Dataset`` -- one whose
    schema is unknown, or whose plan cannot be optimized -- passes every value comparison
    above and fails the first real query.
    """
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    made = {"from_duckdb": bt.from_duckdb(con.sql("SELECT 1 AS i UNION ALL SELECT 5"))}

    torch = pytest.importorskip("torch", reason="torch absent")
    made["from_torch"] = bt.from_torch(torch.tensor([1.0, 5.0]))

    for label, ds in made.items():
        column = ds.schema.names[0]
        total = ds.filter(bt.col(column) > 0).agg(n=bt.col(column).count()).to_pydict()
        assert total["n"] == [2], f"{label} did not survive a filter and aggregate"
        assert ds.schema.names, f"{label} produced a dataset with no known schema"
