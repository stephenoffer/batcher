"""`Dataset.to_dask` and `Dataset.to_huggingface` against fake ``dask`` / ``datasets`` modules.

Neither library is a test dependency. The fakes record what the export hands them
(``dd.from_map``'s function, inputs and meta; ``datasets.Dataset``'s table and features;
``IterableDataset.from_generator``'s generator), which is where the behaviour worth pinning
lives: that no whole-table collect happens where the policy promises none, that every row
lands in exactly one partition, and how label and image columns are translated. The live
tests under ``tests/integration/live/`` run the real libraries when installed.
"""

from __future__ import annotations

import sys
import types

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.dataset.frame import Dataset


# --- fake dask ----------------------------------------------------------------
@pytest.fixture
def fake_dask(monkeypatch):
    state = types.SimpleNamespace(calls=[], read_parquet=[])

    def from_map(func, parts, meta=None, label=None):
        state.calls.append(types.SimpleNamespace(func=func, parts=parts, meta=meta, label=label))
        return ("dask-frame", len(parts))

    def read_parquet(path):
        state.read_parquet.append(path)
        return ("dask-parquet", path)

    dd = types.ModuleType("dask.dataframe")
    dd.from_map = from_map
    dd.read_parquet = read_parquet
    dask = types.ModuleType("dask")
    dask.dataframe = dd
    monkeypatch.setitem(sys.modules, "dask", dask)
    monkeypatch.setitem(sys.modules, "dask.dataframe", dd)
    return state


@pytest.fixture
def no_whole_collect(monkeypatch):
    """Fail any whole-result collect on a Dataset while the export is being built."""
    calls: list[str] = []

    def forbid(name):
        def _raise(self, *args, **kwargs):
            calls.append(name)
            raise AssertionError(f"to_dask called Dataset.{name}() on the whole result")

        return _raise

    for name in ("to_arrow", "to_pandas", "collect", "to_pydict"):
        monkeypatch.setattr(Dataset, name, forbid(name))
    return calls


def _rows(n: int = 50) -> Dataset:
    return bt.from_pydict({"i": list(range(n)), "s": [f"r{i % 7}" for i in range(n)]})


def test_arrow_policy_streams_batches_into_partitions_without_a_whole_collect(
    fake_dask, no_whole_collect
):
    frame = _rows(5000).to_dask(partition_bytes=1)
    (call,) = fake_dask.calls
    assert frame == ("dask-frame", len(call.parts))
    assert len(call.parts) >= 1
    assert all(isinstance(p, pa.Table) for p in call.parts)
    assert sum(p.num_rows for p in call.parts) == 5000
    assert list(call.meta.columns) == ["i", "s"]
    assert len(call.meta) == 0
    assert no_whole_collect == []


def test_arrow_partitions_convert_to_pandas_only_when_dask_computes_them(fake_dask):
    _rows(10).to_dask()
    (call,) = fake_dask.calls
    frame = call.func(call.parts[0])
    assert sorted(frame["i"].tolist()) == list(range(10))


def test_an_empty_result_still_carries_its_schema(fake_dask):
    _rows(5).filter(bt.col("i") < 0).to_dask()
    (call,) = fake_dask.calls
    assert [p.num_rows for p in call.parts] == [0]
    assert list(call.meta.columns) == ["i", "s"]


def test_deferred_policy_runs_nothing_until_a_partition_is_computed(fake_dask, monkeypatch):
    ds = _rows(200)
    ran: list[int] = []
    real = Dataset.to_arrow

    def counting(self):
        ran.append(1)
        return real(self)

    monkeypatch.setattr(Dataset, "to_arrow", counting)
    ds.to_dask(materialize="deferred", npartitions=4)
    (call,) = fake_dask.calls
    assert list(call.parts) == [0, 1, 2, 3]
    assert ran == []
    frames = [call.func(p) for p in call.parts]
    assert len(ran) == 4
    seen = sorted(i for f in frames for i in f["i"].tolist())
    assert seen == list(range(200)), "every row must land in exactly one partition"


def test_deferred_bucketing_is_deterministic_across_runs(fake_dask):
    _rows(100).to_dask(materialize="deferred", npartitions=3)
    (call,) = fake_dask.calls
    first = [sorted(call.func(p)["i"].tolist()) for p in call.parts]
    second = [sorted(call.func(p)["i"].tolist()) for p in call.parts]
    assert first == second


def test_deferred_refuses_a_schema_with_nothing_to_bucket_by(fake_dask):
    ds = bt.from_arrow(pa.table({"l": [[1], [2]]}))
    with pytest.raises(PlanError, match="every column is nested"):
        ds.to_dask(materialize="deferred", npartitions=2)


def test_parquet_policy_stages_and_hands_dask_the_directory(fake_dask, tmp_path):
    frame = _rows(10).to_dask(materialize="parquet", staging_path=str(tmp_path))
    (path,) = fake_dask.read_parquet
    assert frame == ("dask-parquet", path)
    assert path.startswith(str(tmp_path))
    assert bt.read.parquet(path).count() == 10


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"materialize": "pandas"}, "expected one of"),
        ({"materialize": "deferred", "npartitions": 0}, "at least 1"),
    ],
)
def test_bad_policies_are_refused_by_name(fake_dask, kwargs, match):
    with pytest.raises(PlanError, match=match):
        _rows(3).to_dask(**kwargs)


def test_without_dask_the_error_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "dask.dataframe", None)
    with pytest.raises(ImportError, match="dask"):
        _rows(3).to_dask()


# --- fake datasets -----------------------------------------------------------
class _Features(dict):
    @classmethod
    def from_arrow_schema(cls, schema):
        return cls({f.name: ("Value", str(f.type)) for f in schema})

    @property
    def arrow_schema(self):
        fields = []
        for name, feature in self.items():
            if isinstance(feature, _ClassLabel):
                fields.append(pa.field(name, pa.int64()))
            elif isinstance(feature, _Image):
                fields.append(
                    pa.field(name, pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
                )
            else:
                fields.append(pa.field(name, _SCHEMA_TYPES[name]))
        return pa.schema(fields)


_SCHEMA_TYPES: dict[str, pa.DataType] = {}


class _ClassLabel:
    def __init__(self, names):
        self.names = names


class _Image:
    pass


@pytest.fixture
def fake_datasets(monkeypatch):
    state = types.SimpleNamespace(datasets=[], generators=[])

    class InMemoryTable:
        def __init__(self, table):
            self.table = table

    class Dataset_:
        def __init__(self, table, info=None):
            self.data = table
            self.features = info.features
            state.datasets.append(self)

    class IterableDataset:
        @staticmethod
        def from_generator(generator, features=None):
            state.generators.append((generator, features))
            return ("iterable", features)

    module = types.ModuleType("datasets")
    module.Features = _Features
    module.ClassLabel = _ClassLabel
    module.Image = _Image
    module.Dataset = Dataset_
    module.IterableDataset = IterableDataset
    module.DatasetInfo = lambda features: types.SimpleNamespace(features=features)
    table_mod = types.ModuleType("datasets.table")
    table_mod.InMemoryTable = InMemoryTable
    monkeypatch.setitem(sys.modules, "datasets", module)
    monkeypatch.setitem(sys.modules, "datasets.table", table_mod)
    return state


def _labelled() -> Dataset:
    table = pa.table(
        {
            "text": ["good", "bad", "fine"],
            "label": ["pos", "neg", None],
            "img": pa.array([b"\x89PNG", None, b"\xff\xd8"], pa.binary()),
            "tags": [["a"], [], ["b", "c"]],
            "meta": [{"k": 1}, {"k": 2}, {"k": 3}],
        }
    )
    _SCHEMA_TYPES.update({f.name: f.type for f in table.schema})
    return bt.from_arrow(table)


def test_materialized_translates_labels_images_and_keeps_nested_columns(fake_datasets):
    _labelled().to_huggingface(class_labels="label", images="img")
    (hf,) = fake_datasets.datasets
    assert hf.features["label"].names == ["neg", "pos"]
    assert isinstance(hf.features["img"], _Image)
    assert hf.features["tags"] == ("Value", "list<item: string>")
    table = hf.data.table.sort_by("text")
    assert table.column("label").to_pylist() == [0, None, 1]
    assert table.column("img").to_pylist() == [
        None,
        {"bytes": b"\xff\xd8", "path": None},
        {"bytes": b"\x89PNG", "path": None},
    ]
    assert table.column("meta").to_pylist()[0] == {"k": 2}


def test_explicit_label_names_fix_the_code_order(fake_datasets):
    _labelled().to_huggingface(class_labels={"label": ["pos", "neg"]})
    table = fake_datasets.datasets[0].data.table.sort_by("text")
    assert table.column("label").to_pylist() == [1, None, 0]


def test_a_value_outside_the_label_names_is_refused(fake_datasets):
    with pytest.raises(PlanError, match="'neg', which is not one of its label names"):
        _labelled().to_huggingface(class_labels={"label": ["pos"]})


def test_integer_codes_need_explicit_names(fake_datasets):
    ds = bt.from_pydict({"y": [0, 1]})
    with pytest.raises(PlanError, match="integer codes"):
        ds.to_huggingface(class_labels="y")
    with pytest.raises(PlanError, match=r"code outside 0\.\.0"):
        ds.to_huggingface(class_labels={"y": ["only"]})


def test_an_image_column_of_the_wrong_type_is_refused(fake_datasets):
    with pytest.raises(PlanError, match="image column 'n' must be binary"):
        bt.from_pydict({"n": [1]}).to_huggingface(images=["n"])


def test_iterable_runs_the_query_only_as_it_is_iterated(fake_datasets, monkeypatch):
    ds = _labelled()
    pulls: list[int] = []
    real = Dataset.iter_batches

    def counting(self, *args, **kwargs):
        pulls.append(1)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Dataset, "iter_batches", counting)
    monkeypatch.setattr(Dataset, "to_arrow", lambda self: pytest.fail("whole collect"))
    result = ds.to_huggingface("iterable", class_labels={"label": ["neg", "pos"]}, images="img")
    assert result[0] == "iterable"
    assert pulls == []
    generator, features = fake_datasets.generators[0]
    rows = sorted(generator(), key=lambda r: r["text"])
    assert pulls == [1]
    assert [r["label"] for r in rows] == [0, None, 1]
    assert rows[0]["img"] is None
    assert rows[1]["img"] == {"bytes": b"\xff\xd8", "path": None}
    assert features["label"].names == ["neg", "pos"]


def test_an_unknown_mode_is_refused(fake_datasets):
    with pytest.raises(PlanError, match="expected one of"):
        _labelled().to_huggingface("streaming")
