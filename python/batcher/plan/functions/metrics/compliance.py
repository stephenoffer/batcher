"""Structured-output compliance metrics — did the model return the shape you asked for.

A pipeline that asks for JSON, or for an answer inside a tag, only works if the model actually
produces that shape, and the failure is silent: one malformed row becomes a null downstream. These
measure the compliance rate directly, as a corpus number, so a drop in JSON-mode reliability or a
prompt change that breaks the format shows up before it reaches the parser. Each is a single
mergeable aggregate over the string primitives and composes inside `group_by`.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import lit
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if
from batcher.plan.functions.string import (
    extract_between,
    extract_choice,
    extract_first_number,
    extract_json,
    extract_tag,
)

__all__ = [
    "boxed_answer_rate",
    "choice_answer_rate",
    "json_present_rate",
    "numeric_answer_rate",
    "tagged_answer_rate",
    "valid_json_rate",
]


def valid_json_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that are valid JSON on their own — the JSON-mode compliance rate.

    The strict structured-output number: the whole output parses as a JSON object or array, with no
    surrounding prose. It is the metric that says whether a JSON-mode or guided-decoding run is
    actually returning JSON, and a drop between runs flags a format regression before the downstream
    parser starts nulling rows.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The valid-JSON rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ['{"a": 1}', "sorry, no", "[1, 2, 3]"]})
            >>> round(ds.agg(v=bt.valid_json_rate("o")).to_pydict()["v"][0], 4)
            0.6667
    """
    return count_if(_as_column(text).str.looks_like_json()) / count_if(lit(True))


def json_present_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that contain an extractable JSON object — the lenient JSON rate.

    The permissive companion of `valid_json_rate`: it counts an output where a ``{...}`` object
    can be pulled out of surrounding prose ("Sure, here is the JSON: {...}"), which a tolerant
    parser recovers. Read the gap between this and `valid_json_rate` as the share of outputs that
    need extraction rather than parsing cleanly.

    Args:
        text: The generated-text column.

    Returns:
        The JSON-present rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ['here: {"a": 1}', "no json", "[1, 2]"]})
            >>> round(ds.agg(p=bt.json_present_rate("o")).to_pydict()["p"][0], 4)
            0.3333
    """
    return count_if(extract_json(_as_column(text)) != lit("")) / count_if(lit(True))


def tagged_answer_rate(text: IntoExpr, tag: str) -> Expr:
    """The fraction of generations with a non-empty ``<tag>...</tag>`` block — the tag-format rate.

    Many prompts ask the model to wrap its answer in a named tag (``<answer>...</answer>``) so it
    can be sliced out cleanly. This is the corpus rate of outputs that actually did, the compliance
    number for a tag-delimited format, so a low rate says the prompt is not reliably producing the
    wrapper the parser depends on.

    Args:
        text: The generated-text column.
        tag: The tag name to look for, without angle brackets.

    Returns:
        The tagged-answer rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"o": ["<answer>Paris</answer>", "no tag here", "<answer></answer>"]}
            ... )
            >>> round(ds.agg(t=bt.tagged_answer_rate("o", "answer")).to_pydict()["t"][0], 4)
            0.3333
    """
    return count_if(extract_tag(_as_column(text), tag) != lit("")) / count_if(lit(True))


def numeric_answer_rate(text: IntoExpr) -> Expr:
    """The fraction of generations from which a number can be parsed — the numeric-answer rate.

    A benchmark that grades a numeric answer needs a number in the output. This is the corpus rate
    of outputs where the first numeric span parses, the format-compliance number for a math or
    counting task, so a low rate says the model is answering in prose the grader cannot read rather
    than getting the arithmetic wrong.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The numeric-answer rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["the answer is 42", "no number", "about -3.5", "nope"]})
            >>> ds.agg(n=bt.numeric_answer_rate("o")).to_pydict()["n"][0]
            0.5
    """
    return count_if(extract_first_number(_as_column(text)).is_not_null()) / count_if(lit(True))


def choice_answer_rate(text: IntoExpr) -> Expr:
    """The fraction of generations with a standalone multiple-choice letter — the choice rate.

    A multiple-choice benchmark grades a single letter, and a model that answers in a full sentence
    without one is ungradeable. This is the corpus rate of outputs where a standalone ``A``-``H``
    choice can be pulled out, the format-compliance number for a multiple-choice task.

    Args:
        text: The generated-text column.

    Returns:
        The choice-answer rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["The answer is B.", "(C) is right", "no letter", "yes"]})
            >>> ds.agg(c=bt.choice_answer_rate("o")).to_pydict()["c"][0]
            0.5
    """
    return count_if(extract_choice(_as_column(text)) != lit("")) / count_if(lit(True))


def boxed_answer_rate(text: IntoExpr) -> Expr:
    r"""The fraction of generations with a ``\boxed{...}`` answer — the boxed-answer rate.

    Math benchmarks such as MATH ask the model to put its final answer in a LaTeX ``\boxed{}``, and
    the grader reads only that. This is the corpus rate of outputs that produced a non-empty box,
    the format-compliance number for that convention. It checks presence to the first closing
    brace, so a box with nested braces still counts as present.

    Args:
        text: The generated-text column.

    Returns:
        The boxed-answer rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["so \\boxed{42}", "no box here", "\\boxed{7} done"]})
            >>> round(ds.agg(b=bt.boxed_answer_rate("o")).to_pydict()["b"][0], 4)
            0.6667
    """
    boxed = extract_between(_as_column(text), "\\boxed{", "}")
    return count_if(boxed != lit("")) / count_if(lit(True))
