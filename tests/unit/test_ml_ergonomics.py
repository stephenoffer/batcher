"""Ergonomics of the ML surface: naming parity, zero-config defaults, actionable errors.

CPU-only safe: anything needing torch/transformers/a GPU is guarded with importorskip or
tests only the error path when the optional dependency is absent.
"""

from __future__ import annotations

import inspect

import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, MissingDependencyError, PlanError
from batcher.ml import devices
from batcher.ml.preprocessors import (
    Chain,
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_pydict({"x": [1, 2, 3], "text": ["a", "b", "c"], "y": [1.0, 2.0, 3.0]})


@pytest.fixture
def num() -> bt.Dataset:
    return bt.from_pydict({"a": [1.0, 2.0, 3.0, 4.0], "c": ["x", "y", "x", "y"]})


# --------------------------------------------------------------- ds.ml accessor
def test_ml_repr_is_informative(ds: bt.Dataset) -> None:
    text = repr(ds.ml)
    assert "ds.ml" in text
    assert "infer" in text and "embed" in text
    assert "object at" not in text


def test_ml_dir_lists_operations(ds: bt.Dataset) -> None:
    listed = dir(ds.ml)
    assert {"infer", "embed", "map_batches", "to_torch"} <= set(listed)


def test_ml_unknown_attribute_suggests(ds: bt.Dataset) -> None:
    with pytest.raises(AttributeError, match="infer"):
        _ = ds.ml.inferr
    with pytest.raises(AttributeError, match="embed"):
        _ = ds.ml.embedd


def test_ml_unknown_attribute_names_accessor(ds: bt.Dataset) -> None:
    with pytest.raises(AttributeError, match=r"ds\.ml"):
        _ = ds.ml.definitely_not_here


# ---------------------------------------------------------------- naming parity
@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("map_batches", {"batch_size", "num_workers", "num_gpus", "concurrency", "batch_format"}),
        ("map_batches", {"fn_kwargs", "fn_constructor_kwargs"}),
        ("infer", {"device", "dtype", "num_workers", "model_kwargs", "batch_size"}),
        ("embed", {"device", "num_workers", "normalize", "batch_size"}),
        ("iter_torch_batches", {"device", "dtypes", "batch_size"}),
    ],
)
def test_signatures_carry_standard_kwargs(ds: bt.Dataset, method: str, expected: set[str]) -> None:
    params = set(inspect.signature(getattr(ds.ml, method)).parameters)
    assert expected <= params


def test_stream_loader_batch_size_has_default(ds: bt.Dataset) -> None:
    default = inspect.signature(ds.ml.stream_loader).parameters["batch_size"].default
    assert default is not inspect.Parameter.empty


def test_loader_aliases_exist(ds: bt.Dataset) -> None:
    for name in ("to_torch", "to_torch_dataloader", "to_tf", "to_numpy_batches"):
        assert hasattr(ds.ml, name)


def test_to_numpy_batches_streams(ds: bt.Dataset) -> None:
    first = next(ds.ml.to_numpy_batches(batch_size=2))
    assert first["x"].tolist() == [1, 2]


# --------------------------------------------------------- device / dtype config
def test_resolve_device_auto_and_cpu() -> None:
    assert devices.resolve_device("cpu") == "cpu"
    assert devices.resolve_device("auto") in {"cuda", "mps", "cpu", "xpu", "xla"}
    assert devices.get_device("cpu") == "cpu"


def test_resolve_device_typo_suggests() -> None:
    with pytest.raises(PlanError, match="cuda"):
        devices.resolve_device("cudaa")


def test_available_devices_includes_cpu() -> None:
    assert "cpu" in devices.available_devices()


def test_gpu_available_is_bool() -> None:
    assert isinstance(devices.gpu_available(), bool)


@pytest.mark.parametrize(
    ("name", "want"),
    [("fp16", "float16"), ("half", "float16"), ("bf16", "bfloat16"), ("float32", "float32")],
)
def test_resolve_dtype_aliases(name: str, want: str) -> None:
    assert devices.resolve_dtype(name) == want


def test_resolve_dtype_typo_suggests() -> None:
    with pytest.raises(PlanError, match="float"):
        devices.resolve_dtype("float17")


def test_default_dtype_and_batch_size_cpu() -> None:
    assert devices.default_dtype("cpu") == "float32"
    assert devices.default_batch_size(device="cpu") == 256


def test_validate_batch_size_and_num_gpus() -> None:
    devices.validate_batch_size(None)
    devices.validate_batch_size(64)
    with pytest.raises(PlanError, match="batch_size"):
        devices.validate_batch_size(0)
    with pytest.raises(PlanError, match="num_gpus"):
        devices.validate_num_gpus(-1)


# ---------------------------------------------------------- actionable ML errors
def test_map_batches_validates_batch_size(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="batch_size"):
        ds.ml.map_batches(lambda b: b, batch_size=0)


def test_map_batches_validates_num_gpus(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="num_gpus"):
        ds.ml.map_batches(lambda b: b, num_gpus=-1)


def test_bad_batch_format_suggests(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="numpy"):
        ds.ml.map_batches(lambda b: b, batch_format="numpyy")


def test_fn_constructor_kwargs_requires_class(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="fn_constructor_kwargs"):
        ds.ml.map_batches(lambda b: b, fn_constructor_kwargs={"a": 1})


def test_infer_requires_column(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="column"):
        ds.ml.infer("some-model")


def test_infer_bad_column_suggests(ds: bt.Dataset) -> None:
    with pytest.raises(ColumnNotFoundError, match="text"):
        ds.ml.infer("some-model", column="txet")


def test_embed_bad_column_suggests(ds: bt.Dataset) -> None:
    with pytest.raises(ColumnNotFoundError, match="text"):
        ds.ml.embed("some-model", column="txet")


# ------------------------------------------------------ missing-dependency paths
def test_missing_tritonclient_names_install() -> None:
    pytest.importorskip("batcher")
    try:
        import tritonclient  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("tritonclient is installed; the missing-dep path can't be exercised")
    from batcher.ml import triton_client

    with pytest.raises(MissingDependencyError) as exc:
        triton_client("http://x", "m", input_columns=["x"], output_columns=["y"])()
    assert "pip install" in exc.value.install
    assert "triton" in exc.value.install


# ------------------------------------------------------------------ preprocessors
def test_preprocessor_sklearn_vocab(num: bt.Dataset) -> None:
    scaler = StandardScaler(["a"])
    assert scaler.is_fitted is False
    fitted = scaler.fit(num)
    assert fitted.is_fitted is True
    assert isinstance(scaler.get_params(), dict)
    assert set(scaler.get_params()) == {"columns", "with_mean", "with_std"}


def test_preprocessor_repr_reflects_fit(num: bt.Dataset) -> None:
    scaler = StandardScaler(["a"])
    assert "unfitted" in repr(scaler)
    assert "fitted" in repr(scaler.fit(num))


def test_preprocessor_bare_str_column(num: bt.Dataset) -> None:
    # A single column name works, and is NOT split into characters.
    out = StandardScaler("a").fit_transform(num)
    assert "a" in out.columns


def test_scaler_missing_column_suggests(num: bt.Dataset) -> None:
    with pytest.raises(Exception, match="'a'"):
        StandardScaler(["aa"]).fit(num)


def test_transform_before_fit_errors(num: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="fit"):
        StandardScaler(["a"]).transform(num)


def test_chain_accepts_a_list(num: bt.Dataset) -> None:
    chained = Chain([SimpleImputer(["a"]), StandardScaler(["a"])])
    assert len(chained) == 2
    assert chained.fit_transform(num) is not None


@pytest.mark.parametrize(
    "make",
    [
        lambda: MinMaxScaler("a"),
        lambda: OneHotEncoder("c"),
        lambda: SimpleImputer("a"),
    ],
)
def test_preprocessors_accept_bare_str(num: bt.Dataset, make) -> None:
    assert make().fit_transform(num) is not None
