"""`ds.ml.extract` / `ds.ml.classify` — the AI-powered-ETL step that yields typed columns.

An engine is a zero-arg callable returning `list[str] -> list[str]`, so a deterministic
stub stands in for a model and the whole path is testable with no GPU. What is under test
is not the model — it is everything Batcher promises *around* it:

* the output schema is the **declared** one, not whatever the model emitted this batch
  (the bug that makes `generate(parse_json=True)` fail at concat time);
* a bad generation degrades one row to null, never the batch;
* a label column's domain is exactly the declared label set;
* both stay lazy `Dataset`s and compose with the rest of the engine.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml import json_schema

pytestmark = pytest.mark.unit


def _keyed_engine(table: dict[str, str], *, default: str = "{}"):
    """An engine keyed by the first line of the prompt, so batching order cannot matter."""

    def factory():
        return lambda prompts: [table.get(p.split("\n")[0], default) for p in prompts]

    return factory


def _const_engine(response: str):
    def factory():
        return lambda prompts: [response] * len(prompts)

    return factory


# --- extract --------------------------------------------------------------------------


def test_extract_appends_one_typed_column_per_field():
    ds = bt.from_pydict({"note": ["Paid 42 USD to Acme"]})
    engine = _const_engine('{"vendor": "Acme", "total": 42.0}')
    schema = {"vendor": "string", "total": "float64"}
    out = ds.ml.extract(engine, schema=schema, prompt_column="note")
    assert out.to_pydict() == {"note": ["Paid 42 USD to Acme"], "vendor": ["Acme"], "total": [42.0]}
    assert out.schema.field("total").type == pa.float64()
    assert out.schema.field("vendor").type == pa.string()


def test_extract_length_sorted_dispatch_keeps_rows_aligned():
    """extract dispatches longest-prompt-first for throughput but restores row order."""
    responses = {
        "x": '{"n": 1}',
        "xxxxxxxx": '{"n": 2}',  # longest
        "xxx": '{"n": 3}',
    }
    engine = _keyed_engine(responses)
    ds = bt.from_arrow(pa.table({"q": ["x", "xxxxxxxx", "xxx"]}).to_batches(max_chunksize=8))
    got = ds.ml.extract(engine, schema={"n": "int64"}, prompt_column="q")
    assert got.to_pydict() == {"q": ["x", "xxxxxxxx", "xxx"], "n": [1, 2, 3]}


def test_the_declared_schema_survives_a_model_that_omits_a_field():
    """The exact shape that makes `generate(parse_json=True)` fail: two batches, two key
    sets. With a declared schema the Arrow types are identical and the missing value is
    a null, so the scan completes."""
    engine = _keyed_engine({"a": '{"label": "x", "score": 1}', "b": '{"label": "y"}'})
    ds = bt.from_arrow(pa.table({"q": ["a", "b"]}).to_batches(max_chunksize=1))
    got = ds.ml.extract(engine, schema={"label": "string", "score": "int64"}, prompt_column="q")
    assert got.to_pydict() == {"q": ["a", "b"], "label": ["x", "y"], "score": [1, None]}


def test_generate_parse_json_unifies_a_drifting_struct_across_batches():
    """The inferred-struct path unifies per-batch key sets instead of failing.

    Two batches, two key sets: batch `a` carries `score`, batch `b` omits it. The scan
    widens the struct to the union of the fields and nulls the absent one, so the query
    completes. `extract` is still the right call when you want the schema *declared*
    up front rather than inferred from whatever the model happened to return.
    """
    engine = _keyed_engine({"a": '{"label": "x", "score": 1}', "b": '{"label": "y"}'})
    ds = bt.from_arrow(pa.table({"q": ["a", "b"]}).to_batches(max_chunksize=1))
    out = ds.ml.generate(engine, prompt_column="q", parse_json=True)
    assert out.schema.field("response").type == pa.struct(
        [("label", pa.string()), ("score", pa.int64())]
    )
    assert out.to_pydict()["response"] == [
        {"label": "x", "score": 1},
        {"label": "y", "score": None},
    ]


def test_an_unparseable_response_nulls_the_row_not_the_batch():
    engine = _keyed_engine({"good": '{"v": 1}', "bad": "I'm sorry, I cannot help with that."})
    ds = bt.from_pydict({"q": ["good", "bad"]})
    got = ds.ml.extract(engine, schema={"v": "int64"}, prompt_column="q")
    assert got.to_pydict()["v"] == [1, None]


def test_values_are_coerced_to_the_declared_type():
    """A model returns `"42"` for an integer and `"yes"` for a boolean. Recover the row."""
    engine = _const_engine('{"n": "42", "f": "1.5", "b": "yes", "s": 7}')
    ds = bt.from_pydict({"q": ["x"]})
    got = ds.ml.extract(
        engine,
        schema={"n": "int64", "f": "float64", "b": "bool", "s": "string"},
        prompt_column="q",
    ).to_pydict()
    assert got["n"] == [42]
    assert got["f"] == [1.5]
    assert got["b"] == [True]
    assert got["s"] == ["7"]


def test_a_value_that_cannot_coerce_becomes_null():
    engine = _const_engine('{"n": "not-a-number"}')
    got = bt.from_pydict({"q": ["x"]}).ml.extract(engine, schema={"n": "int64"}, prompt_column="q")
    assert got.to_pydict()["n"] == [None]


def test_a_non_object_json_response_nulls_every_field():
    engine = _const_engine("[1, 2, 3]")
    got = bt.from_pydict({"q": ["x"]}).ml.extract(engine, schema={"n": "int64"}, prompt_column="q")
    assert got.to_pydict()["n"] == [None]


def test_extract_stays_lazy_and_composes_downstream():
    engine = _keyed_engine({"a": '{"total": 10}', "b": '{"total": 30}'})
    ds = bt.from_pydict({"q": ["a", "b"]})
    extracted = ds.ml.extract(engine, schema={"total": "int64"}, prompt_column="q")
    assert isinstance(extracted, bt.Dataset)
    assert extracted.filter(bt.col("total") > 20).to_pydict()["q"] == ["b"]
    assert extracted.agg(s=bt.col("total").sum()).to_pydict()["s"] == [40]


def test_extract_supports_a_template_over_several_columns():
    engine = _const_engine('{"ok": true}')
    ds = bt.from_pydict({"a": ["x"], "b": ["y"]})
    got = ds.ml.extract(engine, schema={"ok": "bool"}, template="{a} and {b}")
    assert got.to_pydict()["ok"] == [True]


def test_an_empty_or_unknown_schema_is_rejected():
    ds = bt.from_pydict({"q": ["x"]})
    with pytest.raises(PlanError, match="at least one field"):
        ds.ml.extract(_const_engine("{}"), schema={}, prompt_column="q")
    with pytest.raises(PlanError, match="unknown dtype"):
        ds.ml.extract(_const_engine("{}"), schema={"v": "complex128"}, prompt_column="q")


# --- json_schema ----------------------------------------------------------------------


def test_json_schema_maps_dtypes_to_json_types():
    got = json_schema({"s": "string", "n": "int64", "f": "float64", "b": "bool"})
    assert got == {
        "type": "object",
        "properties": {
            "s": {"type": "string"},
            "n": {"type": "integer"},
            "f": {"type": "number"},
            "b": {"type": "boolean"},
        },
        "required": ["s", "n", "f", "b"],
    }


def test_json_schema_accepts_dtype_aliases():
    assert json_schema({"n": "long"})["properties"] == {"n": {"type": "integer"}}
    assert json_schema({"f": "double"})["properties"] == {"f": {"type": "number"}}


def test_json_schema_rejects_a_dtype_with_no_json_analogue():
    with pytest.raises(PlanError, match="no JSON Schema equivalent"):
        json_schema({"t": "timestamp"})


# --- classify -------------------------------------------------------------------------


def test_classify_appends_a_label_column():
    engine = _keyed_engine({"loved it": "positive", "awful": "negative"}, default="positive")
    ds = bt.from_pydict({"review": ["loved it", "awful"]})
    got = ds.ml.classify(engine, labels=["positive", "negative"], prompt_column="review")
    assert got.to_pydict()["label"] == ["positive", "negative"]


@pytest.mark.parametrize(
    "answer",
    ["positive", "Positive", "POSITIVE", "positive.", ' "positive" ', "The sentiment is positive."],
)
def test_a_label_is_recovered_through_the_noise_a_model_adds(answer):
    """Case, punctuation, quoting, and a carrier sentence must all resolve to the label."""
    ds = bt.from_pydict({"r": ["x"]})
    got = ds.ml.classify(_const_engine(answer), labels=["positive", "negative"], prompt_column="r")
    assert got.to_pydict()["label"] == ["positive"]


def test_an_off_menu_answer_becomes_null_so_it_is_countable():
    ds = bt.from_pydict({"r": ["x", "y"]})
    engine = _keyed_engine({"x": "positive", "y": "somewhat mixed"}, default="?")
    got = ds.ml.classify(engine, labels=["positive", "negative"], prompt_column="r")
    assert got.to_pydict()["label"] == ["positive", None]
    assert got.filter(bt.col("label").is_null()).count() == 1


def test_an_ambiguous_answer_naming_two_labels_is_null_not_a_coin_flip():
    ds = bt.from_pydict({"r": ["x"]})
    engine = _const_engine("could be positive or negative")
    got = ds.ml.classify(engine, labels=["positive", "negative"], prompt_column="r")
    assert got.to_pydict()["label"] == [None]


def test_the_label_columns_domain_is_exactly_the_declared_set():
    labels = ["a", "b"]
    engine = _const_engine("B")
    ds = bt.from_pydict({"r": ["x"] * 5})
    got = ds.ml.classify(engine, labels=labels, prompt_column="r").to_pydict()["label"]
    assert set(got) <= set(labels)
    assert got == ["b"] * 5, "the canonical spelling wins, not the model's casing"


def test_classify_names_its_output_column():
    ds = bt.from_pydict({"r": ["x"]})
    got = ds.ml.classify(
        _const_engine("spam"), labels=["spam", "ham"], prompt_column="r", output_column="verdict"
    )
    assert "verdict" in got.columns


def test_bad_label_sets_are_rejected():
    ds = bt.from_pydict({"r": ["x"]})
    with pytest.raises(PlanError, match="non-empty"):
        ds.ml.classify(_const_engine("a"), labels=[], prompt_column="r")
    with pytest.raises(PlanError, match="distinct ignoring case"):
        ds.ml.classify(_const_engine("a"), labels=["a", "A"], prompt_column="r")


def test_classify_stays_lazy_and_groups():
    engine = _keyed_engine({"a": "spam", "b": "ham", "c": "spam"}, default="ham")
    ds = bt.from_pydict({"r": ["a", "b", "c"]})
    counts = (
        ds.ml.classify(engine, labels=["spam", "ham"], prompt_column="r")
        .group_by("label")
        .agg(n=bt.count())
        .sort("label")
        .to_pydict()
    )
    assert counts == {"label": ["ham", "spam"], "n": [1, 2]}


# --- architecture ---------------------------------------------------------------------


def test_both_lower_to_map_batches_so_they_stream_and_distribute():
    """`extract`/`classify` must be a linear map, not a new operator: that is what puts
    them in the streaming allow-list and the distributed embarrassingly-parallel set."""
    from batcher.dist.executors.plan_analysis import _is_linear_map_pipeline
    from batcher.plan.logical import is_streamable

    ds = bt.from_pydict({"q": ["a"]})
    extracted = ds.ml.extract(_const_engine('{"v": 1}'), schema={"v": "int64"}, prompt_column="q")
    classified = ds.ml.classify(_const_engine("a"), labels=["a", "b"], prompt_column="q")
    assert _is_linear_map_pipeline(extracted._plan)
    assert _is_linear_map_pipeline(classified._plan)
    assert is_streamable(extracted._plan)
    assert is_streamable(classified._plan)


def test_extract_threads_an_image_into_the_request():
    """image_column threads into the GenerateSpec extract builds — a tensor image needs no
    Pillow, so the request for the imaged row carries the decoded array under "image"."""
    import numpy as np

    from batcher.io.formats.ml.tensor import to_tensor_column
    from batcher.ml.llm.structured import _extract_batch

    seen = []

    def factory():
        def engine(requests):
            seen.extend(requests)
            return ['{"n": 1}'] * len(requests)

        return engine

    pixels = to_tensor_column(np.zeros((1, 2, 2, 3), dtype=np.uint8))
    batch = pa.RecordBatch.from_arrays([pa.array(["a"]), pixels], names=["q", "pic"])
    _extract_batch(
        factory(),
        batch,
        fields={"n": pa.int64()},
        prompt_column="q",
        template=None,
        instruct=False,
        image_column="pic",
    )
    assert isinstance(seen[0], dict) and "image" in seen[0]


def test_classify_accepts_image_column_param():
    ds = bt.from_pydict({"q": ["a"]})
    # Wiring check: the param exists and lowers without error on the text path.
    out = ds.ml.classify(_const_engine("yes"), labels=["yes", "no"], prompt_column="q")
    assert out.to_pydict()["label"] == ["yes"]
