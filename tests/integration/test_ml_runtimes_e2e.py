"""Local model runtimes end to end — the exported model must compute what the model does.

The unit suite covers provider selection and port renaming without a model. These run a real
graph through a real `Dataset`, and hold every result against calling the model directly:
running somewhere else must change *where* a plan runs, never *what* it computes.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def scripted(tmp_path_factory):
    """A TorchScript archive of a trivial affine model, plus the values it should produce."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    path = tmp_path_factory.mktemp("runtimes") / "affine.pt"
    module = torch.jit.script(torch.nn.Linear(3, 2).eval())
    torch.jit.save(module, str(path))
    return str(path), module


@pytest.fixture(scope="module")
def graph(tmp_path_factory, scripted):
    """The same model exported to ONNX with a dynamic batch axis."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    pytest.importorskip("onnx", reason="onnx not installed (needed to export)")
    pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
    _, module = scripted
    path = tmp_path_factory.mktemp("runtimes-onnx") / "affine.onnx"
    torch.onnx.export(
        module,
        (torch.zeros(2, 3),),
        str(path),
        input_names=["features"],
        output_names=["scores"],
        dynamic_axes={"features": {0: "batch"}, "scores": {0: "batch"}},
        dynamo=False,
    )
    return str(path)


def _rows():
    return {"features": [[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0], [0.0, 0.0, 0.0]]}


def _flat(rows):
    """One flat float list, because `pytest.approx` will not compare nested sequences."""
    return [float(value) for row in rows for value in row]


def _expected(module):
    import torch

    with torch.inference_mode():
        return module(torch.tensor(_rows()["features"])).tolist()


def test_torch_predictor_matches_calling_the_module_directly(scripted):
    path, module = scripted
    udf = bt.ml.torch_predictor(path, input_columns=["features"], output_columns=["scores"])
    out = bt.from_pydict(_rows()).ml.map_batches(udf).to_pydict()
    assert _flat(out["scores"]) == pytest.approx(_flat(_expected(module)), abs=1e-5)


def test_torch_predictor_keeps_the_input_columns_alongside_the_output(scripted):
    path, _ = scripted
    udf = bt.ml.torch_predictor(path, input_columns=["features"], output_columns=["scores"])
    out = bt.from_pydict(_rows()).ml.map_batches(udf).to_pydict()
    assert out["features"] == _rows()["features"]


def test_torch_predictor_splitting_a_batch_changes_nothing(scripted):
    # `max_batch_size` is a scheduling choice, so it must not move a single value.
    path, module = scripted
    whole = bt.ml.torch_predictor(path, input_columns=["features"], output_columns=["scores"])
    split = bt.ml.torch_predictor(
        path, input_columns=["features"], output_columns=["scores"], max_batch_size=1
    )
    ds = bt.from_pydict(_rows())
    assert _flat(ds.ml.map_batches(split).to_pydict()["scores"]) == pytest.approx(
        _flat(ds.ml.map_batches(whole).to_pydict()["scores"]), abs=1e-6
    )
    assert _flat(ds.ml.map_batches(whole).to_pydict()["scores"]) == pytest.approx(
        _flat(_expected(module)), abs=1e-5
    )


def test_torch_predictor_pipelining_preserves_row_order(scripted):
    path, module = scripted
    udf = bt.ml.torch_predictor(
        path,
        input_columns=["features"],
        output_columns=["scores"],
        max_batch_size=1,
        pipeline_depth=3,
    )
    out = bt.from_pydict(_rows()).ml.map_batches(udf).to_pydict()
    assert _flat(out["scores"]) == pytest.approx(_flat(_expected(module)), abs=1e-5)


def test_torch_predictor_rejects_a_state_dict_with_an_actionable_message(tmp_path):
    torch = pytest.importorskip("torch", reason="torch not installed")
    from batcher._internal.errors import BackendError

    path = tmp_path / "weights.pt"
    torch.save(torch.nn.Linear(3, 2).state_dict(), str(path))
    udf = bt.ml.torch_predictor(str(path), input_columns=["features"], output_columns=["scores"])
    with pytest.raises(BackendError, match="state_dict"):
        bt.from_pydict(_rows()).ml.map_batches(udf).collect()


def test_onnx_predictor_matches_the_torch_model_it_was_exported_from(graph, scripted):
    _, module = scripted
    udf = bt.ml.onnx_predictor(graph, input_columns=["features"], output_columns=["scores"])
    out = bt.from_pydict(_rows()).ml.map_batches(udf).to_pydict()
    assert _flat(out["scores"]) == pytest.approx(_flat(_expected(module)), abs=1e-4)


def test_onnx_predictor_maps_a_differently_named_column_onto_the_graph_input(graph, scripted):
    _, module = scripted
    udf = bt.ml.onnx_predictor(
        graph,
        input_columns=["x"],
        input_names=["features"],
        output_columns=["y"],
    )
    ds = bt.from_pydict({"x": _rows()["features"]})
    out = ds.ml.map_batches(udf).to_pydict()
    assert _flat(out["y"]) == pytest.approx(_flat(_expected(module)), abs=1e-4)


def test_onnx_predictor_declines_an_input_the_graph_does_not_have(graph):
    from batcher._internal.errors import BackendError

    udf = bt.ml.onnx_predictor(
        graph, input_columns=["features"], input_names=["nope"], output_columns=["scores"]
    )
    with pytest.raises(BackendError, match="no input named"):
        bt.from_pydict(_rows()).ml.map_batches(udf).collect()


def test_onnx_predictor_reports_the_providers_it_actually_bound(graph):
    from batcher.ml.runtimes import OnnxSession

    session = OnnxSession(graph)
    assert "CPUExecutionProvider" in session.providers
    assert session.input_names == ["features"]
    assert session.output_names == ["scores"]
    session.close()


def test_onnx_predictor_casts_the_feed_to_the_dtype_the_graph_declared(graph, scripted):
    # The column is float64 here and the graph declares float32. A graph does not coerce.
    _, module = scripted
    udf = bt.ml.onnx_predictor(graph, input_columns=["features"], output_columns=["scores"])
    rows = {"features": [[1.0, 2.0, 3.0]]}
    out = bt.from_pydict(rows).ml.map_batches(udf).to_pydict()
    assert len(out["scores"][0]) == 2
    assert out["scores"][0] == pytest.approx(list(_expected(module)[0]), abs=1e-4)
