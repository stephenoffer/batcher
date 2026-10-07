"""Qdrant, Pinecone, Milvus and Turbopuffer, end to end through ``bt.read`` / ``ds.write``.

Each client library is replaced by the recording fake in `_vector_fakes`, so these tests pin
the requests each connector sends, how it maps the frame onto the store's own model (ids,
named vectors, metadata, primary keys, reserved attributes), and what it reads back. They do
not prove a live service accepts the requests: `tests/integration/live/` holds the smoke tests
that do, skipped until their credentials are set.
"""

from __future__ import annotations

import pickle

import pyarrow as pa
import pytest
from _vector_fakes import install_milvus, install_pinecone, install_qdrant, install_turbopuffer

import batcher as bt
from batcher._internal.errors import MissingDependencyError, PlanError
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.vector import (
    MilvusSink,
    MilvusSource,
    PineconeSink,
    PineconeSource,
    QdrantSink,
    QdrantSource,
    TurbopufferSink,
    TurbopufferSource,
    VectorWriteError,
    contract,
)

pytestmark = pytest.mark.io

ROWS = {
    "id": [1, 2, 3],
    "embedding": [[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]],
    "title": ["a", None, "c"],
}


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract.time, "sleep", lambda _s: None)


@pytest.mark.parametrize(
    ("name", "source", "sink"),
    [
        ("qdrant", QdrantSource, QdrantSink),
        ("pinecone", PineconeSource, PineconeSink),
        ("milvus", MilvusSource, MilvusSink),
        ("turbopuffer", TurbopufferSource, TurbopufferSink),
    ],
)
def test_each_store_is_registered_both_ways(name, source, sink) -> None:
    assert SOURCES.get(name) is source
    assert SINKS.get(name) is sink


@pytest.mark.parametrize(
    ("call", "extra"),
    [
        (lambda: bt.read.qdrant("docs", location=":memory:"), "qdrant"),
        (lambda: bt.read.pinecone("docs", api_key="k"), "pinecone"),
        (lambda: bt.read.milvus("docs", uri="./m.db"), "milvus"),
        (lambda: bt.read.turbopuffer("docs", region="r", api_key="k"), "turbopuffer"),
    ],
)
def test_a_missing_client_names_the_extra(call, extra) -> None:
    with pytest.raises(MissingDependencyError) as raised:
        call()
    assert f"batcher-engine[{extra}]" in raised.value.install


# --- Qdrant ---------------------------------------------------------------------------


def test_qdrant_upserts_points_with_stable_ids_and_reads_them_back(monkeypatch) -> None:
    store = install_qdrant(monkeypatch)
    frame = {"id": ["doc-1", "doc-2"], "embedding": [[0.5, 1.0], [1.5, 2.0]], "n": [1, 2]}
    manifest = bt.from_pydict(frame).write.qdrant("docs", location=":memory:", metric="cosine")
    assert manifest.total_rows == 2
    ((kind, points),) = store.requests
    assert kind == "upsert"
    ids = [p[0] for p in points]
    assert ids == [contract.stable_point_id("doc-1"), contract.stable_point_id("doc-2")]
    assert points[0][1] == [0.5, 1.0]
    assert points[0][2] == {"n": 1, "id": "doc-1"}, "the original id rides in the payload"
    # A second identical write lands on the same points rather than adding two more.
    bt.from_pydict(frame).write.qdrant("docs", location=":memory:")
    assert len(store.points) == 2

    got = bt.read.qdrant("docs", location=":memory:").sort("id").to_pydict()
    assert got == {"id": ["doc-1", "doc-2"], "embedding": [[0.5, 1.0], [1.5, 2.0]], "n": [1, 2]}
    assert bt.read.qdrant("docs", location=":memory:").to_arrow().schema.field(
        "embedding"
    ).type == pa.list_(pa.float32(), 2)


def test_qdrant_integer_ids_pass_through_and_scroll_pages(monkeypatch) -> None:
    store = install_qdrant(monkeypatch)
    monkeypatch.setattr("batcher.io.formats.vector.qdrant._PAGE_POINTS", 2)
    bt.from_pydict(ROWS).write.qdrant("docs", location=":memory:")
    assert sorted(store.points) == [1, 2, 3]
    ds = bt.read.qdrant("docs", location=":memory:")
    assert ds.sort("id").to_pydict()["title"] == ["a", None, "c"]
    scrolls = [r for r in store.requests if r[0] == "scroll" and r[1][2]]
    assert [s[1][1] for s in scrolls] == [None, 3], "the read follows the next-page offset"
    assert QdrantSource(collection="docs", location=":memory:").row_count() == 3


def test_qdrant_named_vectors_and_delete(monkeypatch) -> None:
    store = install_qdrant(monkeypatch, named={"text": 2, "image": 3})
    frame = {"id": [7], "t": [[1.0, 2.0]], "i": [[1.0, 2.0, 3.0]]}
    bt.from_pydict(frame).write.qdrant(
        "docs", location=":memory:", vectors={"text": "t", "image": "i"}
    )
    assert store.requests[-1][1][0][1] == {"text": [1.0, 2.0], "image": [1.0, 2.0, 3.0]}
    with pytest.raises(PlanError, match="no vector named 'text'"):
        install_qdrant(monkeypatch, named={"image": 3})
        bt.from_pydict(frame).write.qdrant(
            "docs", location=":memory:", vector_name="text", vector_column="t"
        )
    store = install_qdrant(monkeypatch)
    bt.from_pydict(ROWS).write.qdrant("docs", location=":memory:")
    bt.from_pydict({"id": [1, 3]}).write.qdrant("docs", location=":memory:", mode="delete")
    assert store.requests[-1] == ("delete", [1, 3])
    assert sorted(store.points) == [2]


def test_qdrant_refuses_a_wrong_dimension_metric_or_missing_collection(monkeypatch) -> None:
    store = install_qdrant(monkeypatch, size=3)
    with pytest.raises(PlanError, match="3-dimensional"):
        bt.from_pydict(ROWS).write.qdrant("docs", location=":memory:")
    store = install_qdrant(monkeypatch, distance="Dot")
    with pytest.raises(PlanError, match="'Dot' metric"):
        bt.from_pydict(ROWS).write.qdrant("docs", location=":memory:", metric="cosine")
    store.exists = False
    with pytest.raises(PlanError, match="does not exist"):
        bt.from_pydict(ROWS).write.qdrant("docs", location=":memory:")
    assert store.requests == []


def test_qdrant_a_batch_that_does_not_complete_is_reported(monkeypatch) -> None:
    store = install_qdrant(monkeypatch)
    store.status = "acknowledged"
    with pytest.raises(VectorWriteError) as raised:
        QdrantSink(location=":memory:", max_retries=0).write(pa.table(ROWS), "docs")
    assert [f.id for f in raised.value.failures] == [1, 2, 3]
    assert "status 'acknowledged'" in raised.value.failures[0].reason


def test_qdrant_retries_a_transient_failure(monkeypatch) -> None:
    store = install_qdrant(monkeypatch)
    store.flaky.failures = 2
    assert QdrantSink(location=":memory:").write(pa.table(ROWS), "docs").rows == 3
    assert sorted(store.points) == [1, 2, 3]


def test_qdrant_api_key_reference_is_resolved_where_the_client_opens(monkeypatch) -> None:
    store = install_qdrant(monkeypatch)
    monkeypatch.setenv("QDRANT_TEST_KEY", "s3cret")
    sink = QdrantSink(url="http://q:6333", api_key="env:QDRANT_TEST_KEY")
    assert "s3cret" not in repr(pickle.dumps(sink))
    sink.write(pa.table(ROWS), "docs")
    assert store.clients[-1] == {"api_key": "s3cret", "url": "http://q:6333"}


# --- Pinecone -------------------------------------------------------------------------


def test_pinecone_upserts_string_ids_and_metadata_without_nulls(monkeypatch) -> None:
    store = install_pinecone(monkeypatch)
    bt.from_pydict(ROWS).write.pinecone("docs", api_key="k", namespace="prod", metric="cosine")
    kind, namespace, records = store.requests[-1]
    assert (kind, namespace) == ("upsert", "prod")
    assert records[0] == {"id": "1", "values": [0.5, 1.0], "metadata": {"title": "a"}}
    assert records[1] == {"id": "2", "values": [1.5, 2.0]}, "a null is an absent field"


def test_pinecone_checks_the_index_before_writing(monkeypatch) -> None:
    store = install_pinecone(monkeypatch, dimension=3)
    with pytest.raises(PlanError, match="3-dimensional"):
        bt.from_pydict(ROWS).write.pinecone("docs", api_key="k")
    store = install_pinecone(monkeypatch, metric="dotproduct")
    with pytest.raises(PlanError, match="'dotproduct' metric, not 'cosine'"):
        bt.from_pydict(ROWS).write.pinecone("docs", api_key="k", metric="cosine")
    assert bt.from_pydict(ROWS).write.pinecone("docs", api_key="k", metric="dot").total_rows == 3
    store.exists = False
    with pytest.raises(PlanError, match="could not describe pinecone index"):
        bt.from_pydict(ROWS).write.pinecone("docs", api_key="k")


def test_pinecone_refuses_metadata_it_cannot_hold(monkeypatch) -> None:
    store = install_pinecone(monkeypatch)
    frame = pa.table({"id": [1], "embedding": [[1.0, 2.0]], "meta": [{"a": 1}]})
    with pytest.raises(PlanError, match="cannot store column 'meta'"):
        PineconeSink(api_key="k").write(frame, "docs")
    tags = pa.table({"id": [1], "embedding": [[1.0, 2.0]], "tags": [["x", "y"]]})
    assert PineconeSink(api_key="k").write(tags, "docs").rows == 1
    assert store.requests[-1][2][0]["metadata"] == {"tags": ["x", "y"]}


def test_pinecone_a_short_upsert_count_is_a_per_point_failure(monkeypatch) -> None:
    store = install_pinecone(monkeypatch)
    store.drop_one = True
    with pytest.raises(VectorWriteError, match="applied 1 of 2 records"):
        PineconeSink(api_key="k", batch_size=2, max_retries=0).write(
            pa.table({"id": [1, 2], "embedding": [[1.0, 0.0]] * 2}), "docs"
        )


def test_pinecone_reads_a_namespace_by_listing_and_fetching(monkeypatch) -> None:
    store = install_pinecone(monkeypatch)
    bt.from_pydict(ROWS).write.pinecone("docs", api_key="k", namespace="prod")
    got = bt.read.pinecone("docs", api_key="k", namespace="prod").sort("id").to_pydict()
    assert got == {
        "id": ["1", "2", "3"],
        "embedding": [[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]],
        "title": ["a", None, "c"],
    }
    fetches = [r for r in store.requests if r[0] == "fetch"]
    assert [f[2] for f in fetches][-2:] == [["1", "2"], ["3"]]
    bt.from_pydict({"id": [2]}).write.pinecone("docs", api_key="k", namespace="prod", mode="delete")
    assert store.requests[-1] == ("delete", "prod", ["2"])


# --- Milvus ---------------------------------------------------------------------------


def test_milvus_maps_id_and_vector_onto_the_collection_fields(monkeypatch) -> None:
    store = install_milvus(monkeypatch)
    bt.from_pydict(ROWS).write.milvus("docs", uri="./m.db", metric="cosine", partition="p1")
    kind, partition, rows = store.requests[-1]
    assert (kind, partition) == ("upsert", "p1")
    assert rows[0] == {"pk": 1, "vec": [0.5, 1.0], "title": "a"}


def test_milvus_refuses_a_column_the_schema_cannot_hold(monkeypatch) -> None:
    store = install_milvus(monkeypatch)
    frame = dict(ROWS, extra=[1, 2, 3])
    with pytest.raises(PlanError, match="no field 'extra'"):
        bt.from_pydict(frame).write.milvus("docs", uri="./m.db")
    assert store.requests == []
    store = install_milvus(monkeypatch, dynamic=True)
    assert bt.from_pydict(frame).write.milvus("docs", uri="./m.db").total_rows == 3


def test_milvus_checks_dimension_and_index_metric(monkeypatch) -> None:
    install_milvus(monkeypatch, dim=4)
    with pytest.raises(PlanError, match="4-dimensional"):
        bt.from_pydict(ROWS).write.milvus("docs", uri="./m.db")
    install_milvus(monkeypatch, metric="L2")
    with pytest.raises(PlanError, match="'L2' metric, not 'IP'"):
        bt.from_pydict(ROWS).write.milvus("docs", uri="./m.db", metric="dot")
    install_milvus(monkeypatch, metric=None)  # an unindexed field has no metric to disagree with
    assert bt.from_pydict(ROWS).write.milvus("docs", uri="./m.db", metric="dot").total_rows == 3
    with pytest.raises(PlanError, match="does not exist"):
        bt.from_pydict(ROWS).write.milvus("other", uri="./m.db")


def test_milvus_append_is_an_insert_and_is_never_retried(monkeypatch) -> None:
    store = install_milvus(monkeypatch)
    store.flaky.failures = 1
    with pytest.raises(VectorWriteError):
        MilvusSink(uri="./m.db", mode="append").write(pa.table(ROWS), "docs")
    assert store.requests == [], "the one attempt failed and nothing was resent"
    assert MilvusSink(uri="./m.db", mode="append").write(pa.table(ROWS), "docs").rows == 3
    assert store.requests[-1][0] == "insert"
    store.flaky.failures = 1
    assert MilvusSink(uri="./m.db").write(pa.table(ROWS), "docs").rows == 3  # upsert retries


def test_milvus_reads_one_split_per_partition(monkeypatch) -> None:
    store = install_milvus(monkeypatch)
    bt.from_pydict({"id": [1], "embedding": [[0.5, 1.0]], "title": ["a"]}).write.milvus(
        "docs", uri="./m.db"
    )
    bt.from_pydict({"id": [2], "embedding": [[1.5, 2.0]], "title": ["b"]}).write.milvus(
        "docs", uri="./m.db", partition="p1"
    )
    source = MilvusSource(collection="docs", uri="./m.db")
    splits = source.splits()
    assert len(splits) == 2
    rows = [r for s in splits for b in pickle.loads(pickle.dumps(s)).read() for r in b.to_pylist()]
    assert sorted(r["pk"] for r in rows) == [1, 2]
    assert source.schema() == pa.schema(
        [("pk", pa.int64()), ("vec", pa.list_(pa.float32(), 2)), ("title", pa.string())]
    )
    got = bt.read.milvus("docs", uri="./m.db").sort("pk").to_pydict()
    assert got == {"pk": [1, 2], "vec": [[0.5, 1.0], [1.5, 2.0]], "title": ["a", "b"]}
    bt.from_pydict({"id": [1]}).write.milvus("docs", uri="./m.db", mode="delete")
    assert store.requests[-1] == ("delete", None, [1])


def test_milvus_a_field_type_with_no_mapping_asks_for_a_schema(monkeypatch) -> None:
    install_milvus(monkeypatch, extra_fields=(("meta", "JSON"),))
    with pytest.raises(PlanError, match="'meta' is JSON"):
        MilvusSource(collection="docs", uri="./m.db").schema()


# --- Turbopuffer ----------------------------------------------------------------------


def test_turbopuffer_writes_columns_under_its_reserved_names(monkeypatch) -> None:
    store = install_turbopuffer(monkeypatch)
    bt.from_pydict(ROWS).write.turbopuffer("docs", region="gcp-us-central1", api_key="k")
    kind, namespace, request = store.requests[-1]
    assert (kind, namespace) == ("write", "docs")
    assert request["upsert_columns"] == {
        "id": [1, 2, 3],
        "vector": [[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]],
        "title": ["a", None, "c"],
    }
    assert request["distance_metric"] == "cosine_distance"
    assert store.clients[-1] == {"api_key": "k", "region": "gcp-us-central1"}


def test_turbopuffer_checks_an_existing_namespace_and_refuses_collisions(monkeypatch) -> None:
    store = install_turbopuffer(monkeypatch)
    bt.from_pydict(ROWS).write.turbopuffer("docs", region="r", api_key="k", metric="euclidean")
    assert store.requests[-1][2]["distance_metric"] == "euclidean_squared"
    wide = {"id": [9], "embedding": [[1.0, 2.0, 3.0]]}
    with pytest.raises(PlanError, match="2-dimensional vectors"):
        bt.from_pydict(wide).write.turbopuffer("docs", region="r", api_key="k")
    clash = {"id": [9], "embedding": [[1.0, 2.0]], "vector": ["x"]}
    with pytest.raises(PlanError, match="reserved 'vector'"):
        bt.from_pydict(clash).write.turbopuffer("docs", region="r", api_key="k")
    with pytest.raises(PlanError, match="exactly one of region= or base_url="):
        TurbopufferSink(region="r", base_url="https://x", api_key="k")
    with pytest.raises(PlanError, match="no distance metric 'dot'"):
        TurbopufferSink(region="r", api_key="k", metric="dot")


def test_turbopuffer_reads_pages_in_id_order_and_deletes(monkeypatch) -> None:
    store = install_turbopuffer(monkeypatch)
    monkeypatch.setattr("batcher.io.formats.vector.turbopuffer._PAGE_ROWS", 2)
    bt.from_pydict(ROWS).write.turbopuffer("docs", region="r", api_key="k")
    got = bt.read.turbopuffer("docs", region="r", api_key="k").sort("id").to_pydict()
    assert got == {
        "id": [1, 2, 3],
        "embedding": [[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]],
        "title": ["a", None, "c"],
    }
    queries = [r[2] for r in store.requests if r[0] == "query"]
    assert [q["filters"] for q in queries] == [None, ("id", "Gt", 2)]
    assert "vector" in queries[0]["include_attributes"]
    bt.from_pydict({"id": [2]}).write.turbopuffer("docs", region="r", api_key="k", mode="delete")
    assert store.requests[-1][2]["deletes"] == [2]
    assert sorted(store.docs["docs"]) == [1, 3]
