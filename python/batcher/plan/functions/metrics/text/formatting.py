"""Output-shape metrics — did the generation take the form it was asked to take.

An instruction-following system usually specifies a shape as well as a content: answer in JSON,
put the final answer in \boxed{}, reply with a single letter, use a markdown table, cite with a
bracketed number. These are the corpus rates for whether that shape appeared. They check form, not
correctness — a valid-JSON rate says the parser will succeed, not that the fields are right — which
is exactly what makes them the cheap first gate in front of an expensive grader.

Two families sit here: markdown structure (headings, lists, code blocks, tables, links) and answer
extraction (JSON, boxed, tagged, numeric, single-choice). Both aggregate to a fraction of rows.
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
    "bullet_list_rate",
    "choice_answer_rate",
    "code_block_present_rate",
    "heading_rate",
    "json_present_rate",
    "markdown_link_rate",
    "numbered_list_rate",
    "numeric_answer_rate",
    "table_rate",
    "tagged_answer_rate",
    "valid_json_rate",
]


def heading_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that contain a Markdown heading line — an ``ATX`` heading rate.

    Detects an ``ATX`` heading, one to six ``#`` characters then a space at the start of any line
    (``# Title``, ``## Section``). A drop between runs flags a prompt change that stopped the model
    from sectioning its answer.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The heading-present rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["# Title\\ntext", "plain text", "- item one"]})
            >>> round(ds.agg(h=bt.heading_rate("o")).to_pydict()["h"][0], 4)
            0.3333
    """
    return count_if(_as_column(text).str.regexp_matches(r"(?m)^#{1,6} ")) / count_if(lit(True))


def bullet_list_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that contain a Markdown bullet-list item — a bulleting rate.

    Detects a bullet item, one of ``-``, ``*``, or ``+`` then a space at the start of a line
    (``- point``). Read it against `numbered_list_rate` to see whether the model prefers unordered
    or ordered lists for a task that asks for either.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The bullet-list-present rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["- item one", "plain text", "line1\\n* two"]})
            >>> round(ds.agg(b=bt.bullet_list_rate("o")).to_pydict()["b"][0], 4)
            0.6667
    """
    return count_if(_as_column(text).str.regexp_matches(r"(^|\n)[-*+] ")) / count_if(lit(True))


def numbered_list_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that contain a Markdown numbered-list item — an enumeration rate.

    Detects an ordered item, one or more digits then ``.`` and a space at the start of a line
    (``1. first``). Use it when a task asks for ranked or step-by-step output to confirm the model
    actually enumerates rather than running the steps together in prose.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The numbered-list-present rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["1. first", "plain text", "line\\n2. second"]})
            >>> round(ds.agg(n=bt.numbered_list_rate("o")).to_pydict()["n"][0], 4)
            0.6667
    """
    return count_if(_as_column(text).str.regexp_matches(r"(^|\n)\d+\. ")) / count_if(lit(True))


def markdown_link_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that contain a Markdown inline link — a citation-markup rate.

    Detects the ``[text](url)`` inline-link shape anywhere in the output. A task that asks the model
    to cite sources or link to references relies on this markup, and a drop between runs flags that
    the model stopped producing clickable links.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The Markdown-link-present rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["see [docs](http://x)", "plain text", "[a](b)"]})
            >>> round(ds.agg(m=bt.markdown_link_rate("o")).to_pydict()["m"][0], 4)
            0.6667
    """
    link = _as_column(text).str.regexp_matches(r"\[[^\]]+\]\([^)]+\)")
    return count_if(link) / count_if(lit(True))


def table_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that contain a Markdown table row — a tabulation rate.

    Detects a table row, a line that starts with ``|`` and carries a second ``|`` (``| a | b |``).
    Use it when a task asks for tabular output to confirm the model laid the answer out as a table
    rather than as sentences.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The table-row-present rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["| a | b |\\n|---|---|", "plain text", "no table"]})
            >>> round(ds.agg(t=bt.table_rate("o")).to_pydict()["t"][0], 4)
            0.3333
    """
    return count_if(_as_column(text).str.regexp_matches(r"(^|\n)\|.*\|")) / count_if(lit(True))


def code_block_present_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that contain a fenced code block — a code-formatting rate.

    Counts an output with at least one triple-backtick fenced block, the layout a task asks for when
    it wants runnable or copyable code kept out of prose. A drop between runs flags that the model
    stopped fencing its code, which breaks any downstream extractor keyed on the fence.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The fenced-code-present rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["```py\\ncode\\n```", "plain text", "no fence"]})
            >>> round(ds.agg(c=bt.code_block_present_rate("o")).to_pydict()["c"][0], 4)
            0.3333
    """
    return count_if(_as_column(text).str.code_fence_count() > lit(0)) / count_if(lit(True))


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
