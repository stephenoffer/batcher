"""The public inference factories in ``batcher.ml``, which no test called.

Ten names are exported for running a model over a table -- four predictor/client UDF
factories, three LLM engine factories, and the vector-index trio -- and the coverage sweep
found none of them exercised. That is the surface a user reaches for after their pipeline
works, so a defect here surfaces at the least convenient moment.

Two of them can be tested for real on this machine, and are: ``torch_predictor`` and
``onnx_predictor`` run a genuine two-line model end to end through ``ds.ml.map_batches``,
and their results are compared **against each other** as well as against the arithmetic.
That is a real differential -- the same model exported two ways, executed by two runtimes,
through two adapters -- and it is the one shape in this family where a wrong answer rather
than a crash is the plausible failure.

The rest are covered at the contract they actually promise on a machine with no backend and
no credentials: **the factory builds without importing its backend**. That deferred import
is load-bearing. It is what lets a control plane construct a plan naming a model it cannot
itself run, which is the normal case for a GPU or a hosted model, and a factory that
imported eagerly would break plan building everywhere while every inference test still
passed on the box that had the library.

One thing this module deliberately does *not* assert: that a missing model file or absent
credential raises a Batcher error. Several of these paths surface the backend's own
exception (``onnxruntime``'s ``NoSuchFile``, ``FileNotFoundError``,
``huggingface_hub``'s ``RepositoryNotFoundError``, lance's ``ValueError``) rather than a
typed one. That is inconsistent with the project's error contract and it is recorded here
as a fact about today rather than pinned as desirable.
"""

from __future__ import annotations

import pytest

import batcher as bt
import batcher.ml as ml

pytestmark = pytest.mark.integration

INPUT = [1.0, 2.0, 3.0, 4.0, -0.5]
EXPECTED = [x * 2.0 + 1.0 for x in INPUT]


@pytest.fixture(scope="module")
def torch_module():
    """A scripted TorchScript model computing ``x * 2 + 1``, saved to a temp path."""
    torch = pytest.importorskip("torch")
    if not hasattr(ml, "torch_predictor"):
        pytest.skip("torch_predictor is not exported on this build")
    import tempfile
    from pathlib import Path

    import torch.nn as nn

    class Affine(nn.Module):
        def forward(self, x):
            return x * 2.0 + 1.0

    directory = Path(tempfile.mkdtemp())
    path = directory / "affine.pt"
    torch.jit.save(torch.jit.script(Affine()), str(path))
    return Affine, str(path)


@pytest.fixture(scope="module")
def onnx_model(torch_module):
    """The same model exported to ONNX with a dynamic batch axis."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    # The exporter needs onnxscript; without it there is no ONNX file to compare against,
    # and skipping is honest where inventing one would not be.
    pytest.importorskip("onnxscript", reason="torch.onnx.export needs onnxscript")
    from pathlib import Path

    module_cls, torch_path = torch_module
    path = Path(torch_path).with_suffix(".onnx")
    torch.onnx.export(
        module_cls(),
        (torch.tensor([1.0]),),
        str(path),
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "n"}, "y": {0: "n"}},
    )
    return str(path)


@pytest.fixture
def ds():
    return bt.from_pydict({"x": INPUT})


def test_torch_predictor_runs_a_real_model_over_a_dataset(ds, torch_module):
    """End to end: a TorchScript module scoring every row, appended as a new column."""
    _, path = torch_module
    udf = ml.torch_predictor(path, input_columns=["x"], output_columns=["y"])
    got = ds.ml.map_batches(udf).to_pydict()
    assert got["x"] == INPUT, "the input column must survive the projection"
    assert got["y"] == pytest.approx(EXPECTED), f"{got['y']}"


def test_onnx_predictor_runs_the_same_model_and_agrees_with_torch(ds, torch_module, onnx_model):
    """The differential: one model, two runtimes, two adapters, one answer.

    This is what separates a real test of these factories from a smoke test. A wrong axis, a
    dropped final row, or a silent dtype narrowing shows up as a disagreement between the
    two rather than as an exception in either.
    """
    _, torch_path = torch_module
    through_torch = ds.ml.map_batches(
        ml.torch_predictor(torch_path, input_columns=["x"], output_columns=["y"])
    ).to_pydict()
    through_onnx = ds.ml.map_batches(
        ml.onnx_predictor(onnx_model, input_columns=["x"], output_columns=["y"])
    ).to_pydict()
    assert through_onnx["y"] == pytest.approx(EXPECTED)
    assert through_onnx["y"] == pytest.approx(through_torch["y"]), (
        f"onnx {through_onnx['y']} vs torch {through_torch['y']}"
    )
    assert through_onnx["x"] == through_torch["x"] == INPUT


def test_a_predictor_scores_every_row_across_several_batches(torch_module):
    """More rows than one batch, because a per-batch adapter can lose the ragged last one."""
    _, path = torch_module
    rows = [float(i) for i in range(5000)]
    ds = bt.from_pydict({"x": rows})
    udf = ml.torch_predictor(path, input_columns=["x"], output_columns=["y"])
    got = ds.ml.map_batches(udf).to_pydict()
    assert len(got["y"]) == len(rows), "a batch went missing"
    assert got["y"] == pytest.approx([x * 2.0 + 1.0 for x in rows])


def test_a_predictor_loads_its_model_at_execution_and_not_at_build(ds):
    """Building the pipeline must not load the model; that is what "once per worker" means.

    Checked by pointing the factory at a path that does not exist. Constructing the UDF and
    calling ``map_batches`` both succeed, and the load happens later, inside
    ``core/udf/lifecycle.py::build_udf_callable`` -- so a control plane can name a model it
    cannot itself open, which is the normal case for a GPU or a multi-gigabyte checkpoint.

    Also recorded: **asking for the schema does load it.** ``Dataset.schema`` over a plan
    carrying a batch UDF is answered by executing ``Limit(plan, 0)``, and a zero-row
    execution still instantiates the UDF. So `ds.schema` on an inference pipeline pays a
    full model load on the driver. That is a real cost on a large model and it is asserted
    here as today's behaviour rather than as desirable.
    """
    if not hasattr(ml, "torch_predictor"):
        pytest.skip("torch_predictor is not exported on this build")
    udf = ml.torch_predictor("/nonexistent/model.pt", input_columns=["x"], output_columns=["y"])
    pipeline = ds.ml.map_batches(udf)
    assert pipeline is not None, "building the pipeline must not need the model"

    with pytest.raises((FileNotFoundError, ValueError)):
        pipeline.to_pydict()

    with pytest.raises((FileNotFoundError, ValueError)):
        _ = pipeline.schema.names


#: Every factory that builds a UDF class or an engine factory, with the smallest valid
#: argument set. The assertion is the same for all of them and it is the contract that
#: matters: constructing it needs no backend, no model file and no credential.
FACTORIES = [
    ("onnx_predictor", lambda: ml.onnx_predictor("m.onnx", input_columns=["f"])),
    ("openvino_predictor", lambda: ml.openvino_predictor("m.xml", input_columns=["f"])),
    ("torch_predictor", lambda: ml.torch_predictor("m.pt", input_columns=["f"])),
    (
        "torchserve_client",
        lambda: ml.torchserve_client(
            "http://localhost:8080", "m", input_columns=["f"], output_columns=["y"]
        ),
    ),
    (
        "cross_encoder_rerank_udf",
        lambda: ml.cross_encoder_rerank_udf(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", query_column="q", document_column="d"
        ),
    ),
    ("bedrock_engine", lambda: ml.bedrock_engine("anthropic.claude-3-sonnet")),
    ("gemini_engine", lambda: ml.gemini_engine("gemini-1.5-flash")),
    ("sglang_engine", lambda: ml.sglang_engine("meta-llama/Llama-3-8B")),
]


#: Names the ML package exports on this build. The inference factories are added over
#: time and land on a branch before they land here, so a module that hard-codes the list
#: fails for naming one that does not exist yet -- and a module that only iterates what
#: exists stops noticing when a new one arrives untested. Both halves are needed.
FACTORIES = [(name, build) for name, build in FACTORIES if hasattr(ml, name)]

#: Factory-shaped exports this module deliberately leaves to the suites that own them --
#: the LLM engine and UDF family (`tests/unit/test_ml_hosted_engines.py` and neighbours)
#: and the serving clients. Listing them is what lets the completeness check below tell
#: "covered elsewhere" from "nobody has looked at this", so a *new* factory still fails.
_COVERED_ELSEWHERE = frozenset(
    {
        "anthropic_engine",
        "http_client",
        "http_engine",
        "llm_pairwise_udf",
        "llm_score_udf",
        "llm_udf",
        "llm_verify_udf",
        "mmr_rerank_udf",
        "serving_udf",
        "triton_client",
        "vllm_engine",
    }
)


def test_the_factory_list_is_not_empty():
    """The filter above must never reduce the list to nothing."""
    assert FACTORIES, "no inference factory survived the build filter"


def test_every_exported_factory_is_covered_here():
    """A factory that appears in `batcher.ml` and not in this module must fail loudly.

    Without this, the filter above would quietly shrink as names were added, and the module
    would keep passing while covering less and less.
    """
    covered = {name for name, _ in FACTORIES} | _COVERED_ELSEWHERE
    exported = {n for n in getattr(ml, "__all__", ()) if not n.startswith("_")}
    factory_like = {
        n
        for n in exported
        if n.endswith(("_predictor", "_engine", "_client", "_udf")) and n not in covered
    }
    assert not factory_like, (
        f"batcher.ml exports {sorted(factory_like)}, which this module does not exercise; "
        "add each to FACTORIES with its smallest constructor call"
    )


@pytest.mark.parametrize(("name", "build"), FACTORIES)
def test_a_factory_builds_without_its_backend_a_model_or_a_credential(name, build):
    """The deferred-import contract, which is what makes a plan portable to the worker."""
    made = build()
    assert made is not None, f"{name} returned nothing"
    assert isinstance(made, type) or callable(made), (
        f"{name} returned {made!r}, which is neither a UDF class nor an engine factory"
    )


@pytest.mark.parametrize(("name", "build"), FACTORIES)
def test_a_factory_is_reachable_from_the_public_ml_package(name, build):
    """Each is exported, so each is a commitment; the export and the callable must agree."""
    assert name in ml.__all__, f"{name} is not in batcher.ml.__all__"
    assert getattr(ml, name) is not None


def test_openvino_and_sglang_name_the_extra_that_is_missing():
    """When a backend really is absent, the error must be typed and say how to fix it.

    These two are the ones this environment does not have, so they are the ones where the
    missing-dependency path is reachable. The message has to carry the install command:
    "openvino is not installed" without the extra name is a dead end for a user who does
    not know the package is behind ``batcher-engine[openvino]``.
    """
    from batcher._internal.errors import MissingDependencyError

    checked = 0
    for name, build, package in [
        (
            "openvino_predictor",
            lambda: ml.openvino_predictor("m.xml", input_columns=["f"]),
            "openvino",
        ),
        ("sglang_engine", lambda: ml.sglang_engine("model"), "sglang"),
    ]:
        if not hasattr(ml, name):
            continue
        with pytest.raises(MissingDependencyError) as missing:
            build()()
        assert package in str(missing.value)
        assert "pip install" in str(missing.value), (
            f"{name} named the missing package without saying how to install it"
        )
        checked += 1
    if not checked:
        pytest.skip("neither backend-gated factory is exported on this build")


def test_the_vector_index_helpers_refuse_a_path_that_is_not_a_dataset(tmp_path):
    """``vector_search`` and ``build_vector_index`` on a path holding no table.

    Both raise -- which is right -- but they surface lance's own ``ValueError`` rather than
    a Batcher error. Asserted as ``Exception`` on purpose: pinning the third-party type
    would make a lance upgrade fail this test for no reason, and pinning a Batcher type
    would assert something the code does not do. What is checked is that the failure names
    the path, so a user can see which one was wrong.
    """
    pytest.importorskip("lance")
    missing = str(tmp_path / "not-a-dataset.lance")
    with pytest.raises(Exception) as search:
        ml.vector_search(missing, [0.1, 0.2], k=1)
    assert "not-a-dataset" in str(search.value)

    with pytest.raises(Exception) as build:
        ml.build_vector_index(missing)
    assert "not-a-dataset" in str(build.value)


def test_vector_search_round_trips_a_written_index(tmp_path):
    """The one vector case that can run here: write embeddings, index them, search them.

    The embedding column is written as a ``FixedSizeList`` through ``bt.from_numpy`` rather
    than as a plain list of lists: lance refuses a variable-length list as a vector column,
    and that is the right refusal, since a vector index needs a fixed width.
    """
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("lance")
    uri = str(tmp_path / "vectors.lance")
    rows = 512
    embeddings = numpy.array(
        [[float(i % 8), float((i // 8) % 8), 1.0, 0.0] for i in range(rows)], dtype="float32"
    )
    bt.from_numpy(embeddings, column="embedding").write.lance(uri)

    found = ml.vector_search(uri, [0.0, 0.0, 1.0, 0.0], column="embedding", k=5).to_pydict()
    assert len(found["embedding"]) == 5, f"expected five neighbours, got {len(found)}"
    assert all(len(v) == 4 for v in found["embedding"]), "every neighbour keeps its width"
    # The query is the embedding of row 0 exactly, so row 0 must be among its own five
    # nearest neighbours -- the check that separates a real search from five arbitrary rows.
    assert [0.0, 0.0, 1.0, 0.0] in found["embedding"]
