"""Nested `extract` schemas, enums, and the raw-text / diagnostics columns (AP-383, AP-384).

A stub engine stands in for a model, so what is under test is the declared-schema contract:
nested structs and lists keep stable Arrow types across batches, a missing required field
or an off-menu enum value is null *and* explained, and the model's own text survives a
failed parse when the caller asks for it.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml import json_schema

pytestmark = pytest.mark.unit


def _keyed_engine(table: dict[str, str]):
    """An engine keyed by the first line of the prompt, so batching order cannot matter."""

    def factory():
        return lambda prompts: [table[p.split("\n")[0]] for p in prompts]

    return factory


_NESTED = {
    "customer": {"name": "string", "age": "int64"},
    "tags": ["string"],
    "level": {"low", "high"},
}


def test_a_nested_schema_types_structs_lists_and_enums_identically_across_batches():
    replies = {
        "a": '{"customer": {"name": "Ann", "age": 41}, "tags": ["vip"], "level": "High"}',
        "b": '{"customer": {"name": "Bo", "age": "7"}, "tags": [], "level": "low"}',
    }
    ds = bt.from_arrow(pa.table({"q": ["a", "b"]}).to_batches(max_chunksize=1))
    out = ds.ml.extract(_keyed_engine(replies), schema=_NESTED, prompt_column="q")
    assert out.schema.field("customer").type == pa.struct(
        [("name", pa.string()), ("age", pa.int64())]
    )
    assert out.schema.field("tags").type == pa.list_(pa.string())
    assert out.schema.field("level").type == pa.string()
    got = out.to_pydict()
    assert got["customer"] == [{"name": "Ann", "age": 41}, {"name": "Bo", "age": 7}]
    assert got["tags"] == [["vip"], []]
    # The enum maps the model's spelling onto the declared one.
    assert got["level"] == ["high", "low"]


def test_a_missing_required_nested_field_is_null_and_named_in_the_diagnostics():
    replies = {"a": '{"customer": {"name": "Ann"}, "tags": ["x"], "level": "low"}'}
    out = bt.from_pydict({"q": ["a"]}).ml.extract(
        _keyed_engine(replies), schema=_NESTED, prompt_column="q", diagnostics_column="why"
    )
    got = out.to_pydict()
    assert got["customer"] == [{"name": "Ann", "age": None}]
    assert got["why"] == [["customer.age: missing required field"]]


def test_an_off_menu_enum_value_and_a_bad_coercion_are_both_reported():
    replies = {"a": '{"customer": {"name": "Ann", "age": "old"}, "tags": ["x"], "level": "urgent"}'}
    out = bt.from_pydict({"q": ["a"]}).ml.extract(
        _keyed_engine(replies), schema=_NESTED, prompt_column="q", diagnostics_column="why"
    )
    got = out.to_pydict()
    assert got["level"] == [None]
    assert got["customer"] == [{"name": "Ann", "age": None}]
    assert got["why"] == [
        [
            'customer.age: expected int64, got "old"',
            'level: "urgent" is not one of ["high", "low"]',
        ]
    ]


def test_an_explicit_null_is_an_answer_not_a_failure():
    """The instruction tells the model to answer null when it cannot tell; that row fit."""
    replies = {"a": '{"customer": null, "tags": null, "level": null}'}
    out = bt.from_pydict({"q": ["a"]}).ml.extract(
        _keyed_engine(replies), schema=_NESTED, prompt_column="q", diagnostics_column="why"
    )
    assert out.to_pydict()["why"] == [None]


def test_raw_column_keeps_the_text_a_failed_parse_would_lose():
    replies = {"good": '{"v": 1}', "bad": "I cannot answer that."}
    out = bt.from_pydict({"q": ["good", "bad"]}).ml.extract(
        _keyed_engine(replies),
        schema={"v": "int64"},
        prompt_column="q",
        raw_column="raw",
        diagnostics_column="why",
    )
    assert out.columns == ["q", "v", "raw", "why"]
    got = out.to_pydict()
    assert got["v"] == [1, None]
    assert got["raw"] == ['{"v": 1}', "I cannot answer that."]
    assert got["why"] == [None, ["response is not a JSON object"]]
    # A validation failure is distinguishable from a model that answered null.
    failed = out.filter(bt.col("v").is_null() & bt.col("why").is_not_null())
    assert failed.to_pydict()["q"] == ["bad"]


def test_list_elements_are_coerced_one_by_one_with_an_indexed_diagnostic():
    replies = {"a": '{"n": [1, "2", "x"]}'}
    out = bt.from_pydict({"q": ["a"]}).ml.extract(
        _keyed_engine(replies), schema={"n": ["int64"]}, prompt_column="q", diagnostics_column="d"
    )
    got = out.to_pydict()
    assert got["n"] == [[1, 2, None]]
    assert got["d"] == [['n[2]: expected int64, got "x"']]


def test_a_pyarrow_type_is_accepted_as_a_declaration():
    replies = {"a": '{"pts": [{"x": 1.5, "y": 2}]}'}
    point = pa.struct([("x", pa.float64()), ("y", pa.float64())])
    out = bt.from_pydict({"q": ["a"]}).ml.extract(
        _keyed_engine(replies), schema={"pts": pa.list_(point)}, prompt_column="q"
    )
    assert out.schema.field("pts").type == pa.list_(point)
    assert out.to_pydict()["pts"] == [[{"x": 1.5, "y": 2.0}]]


@pytest.mark.parametrize(
    ("schema", "match"),
    [
        ({"v": {}}, "declares no fields"),
        ({"v": ["string", "int64"]}, "exactly one element type"),
        ({"v": set()}, "non-empty set of strings"),
        ({"v": {"A", "a"}}, "differ ignoring case"),
        ({"v": {"inner": "duration"}}, "cannot be extracted"),
        ({"v": 3}, "is declared as 3"),
    ],
)
def test_an_unsupported_declaration_is_refused_at_plan_time(schema, match):
    with pytest.raises(PlanError, match=match):
        bt.from_pydict({"q": ["x"]}).ml.extract(_keyed_engine({}), schema=schema, prompt_column="q")


def test_raw_or_diagnostics_column_may_not_repeat_an_output_name():
    ds = bt.from_pydict({"q": ["x"]})
    with pytest.raises(PlanError, match="already an output column"):
        ds.ml.extract(_keyed_engine({}), schema={"v": "int64"}, prompt_column="q", raw_column="v")
    with pytest.raises(PlanError, match="already an output column"):
        ds.ml.extract(
            _keyed_engine({}),
            schema={"v": "int64"},
            prompt_column="q",
            raw_column="r",
            diagnostics_column="r",
        )


def test_a_nested_schema_spells_out_its_shape_in_the_instruction():
    seen: list[str] = []

    def factory():
        def engine(prompts):
            seen.extend(prompts)
            return ["{}"] * len(prompts)

        return engine

    bt.from_pydict({"q": ["x"]}).ml.extract(
        factory, schema={"c": {"name": "string"}, "n": "int64"}, prompt_column="q"
    ).collect()
    assert '"c": {"name": string}' in seen[0]


def test_a_flat_schema_keeps_the_original_instruction_unchanged():
    seen: list[str] = []

    def factory():
        def engine(prompts):
            seen.extend(prompts)
            return ["{}"] * len(prompts)

        return engine

    bt.from_pydict({"q": ["x"]}).ml.extract(
        factory, schema={"n": "int64"}, prompt_column="q"
    ).collect()
    assert seen[0].endswith("Use null for any value you cannot determine.")


def test_json_schema_nests_objects_arrays_and_enums_with_required_at_every_level():
    assert json_schema(_NESTED) == {
        "type": "object",
        "properties": {
            "customer": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            },
            "tags": {"type": "array", "items": {"type": "string"}},
            "level": {"type": "string", "enum": ["high", "low"]},
        },
        "required": ["customer", "tags", "level"],
    }


# --- generate(parse_json=True, raw_column=...) ---------------------------------------


def test_generate_raw_column_keeps_the_text_beside_the_parsed_struct():
    replies = {"a": '{"x": 1}', "b": "not json"}
    out = bt.from_pydict({"q": ["a", "b"]}).ml.generate(
        _keyed_engine(replies), prompt_column="q", parse_json=True, raw_column="raw"
    )
    assert out.columns == ["q", "response", "raw"]
    got = out.to_pydict()
    assert got["response"] == [{"x": 1}, None]
    assert got["raw"] == ['{"x": 1}', "not json"]


def test_generate_raw_column_needs_parse_json_and_its_own_name():
    ds = bt.from_pydict({"q": ["a"]})
    with pytest.raises(PlanError, match="parse_json=True"):
        ds.ml.generate(_keyed_engine({}), prompt_column="q", raw_column="raw")
    with pytest.raises(PlanError, match="name them apart"):
        ds.ml.generate(_keyed_engine({}), prompt_column="q", parse_json=True, raw_column="response")
