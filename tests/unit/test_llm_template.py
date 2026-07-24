"""Prompt-template rendering: null cells and placeholder validation.

Two silent failures fixed: a null column rendered the literal string ``None`` into the
prompt, and a template naming a missing column failed the whole batch with a bare KeyError.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.llm.requests import _render


def _echo():
    return lambda prompts: list(prompts)


def test_null_prompt_cell_renders_empty_not_the_word_none():
    batch = pa.RecordBatch.from_pydict({"q": ["hi", None]})
    assert _render(None, "q", batch) == ["hi", ""]


def test_null_in_a_template_column_renders_empty():
    batch = pa.RecordBatch.from_pydict({"a": ["x", None], "b": ["y", "z"]})
    assert _render("{a}-{b}", "", batch) == ["x-y", "-z"]


def test_a_template_naming_a_missing_column_raises_plan_error():
    batch = pa.RecordBatch.from_pydict({"a": ["x"]})
    with pytest.raises(PlanError, match="nonexistent"):
        _render("{a} {nonexistent}", "", batch)


def test_generate_surfaces_the_bad_template_at_the_api_edge():
    ds = bt.from_pydict({"a": ["x", "y"]})
    with pytest.raises(PlanError):
        ds.ml.generate(_echo, prompt_column="a", template="{a} {missing}").to_pydict()


def test_generate_renders_a_null_column_as_empty_end_to_end():
    ds = bt.from_pydict({"a": ["x", None]})
    out = ds.ml.generate(_echo, prompt_column="a").to_pydict()
    assert out["response"] == ["x", ""]


def test_few_shot_prepends_demonstration_pairs():
    import pyarrow as pa

    from batcher.ml.llm.requests import _render

    batch = pa.RecordBatch.from_pydict({"q": ["9+10?"]})
    out = _render(None, "q", batch, few_shot=(("2+2?", "4"), ("3+3?", "6")))
    assert out == ["Input: 2+2?\nOutput: 4\n\nInput: 3+3?\nOutput: 6\n\n9+10?"]


def test_few_shot_reaches_generate_end_to_end():
    ds = bt.from_pydict({"q": ["x"]})
    echo = lambda: (lambda ps: list(ps))  # noqa: E731
    out = ds.ml.generate(echo, prompt_column="q", few_shot=[("a", "1")]).to_pydict()
    assert out["response"] == ["Input: a\nOutput: 1\n\nx"]
