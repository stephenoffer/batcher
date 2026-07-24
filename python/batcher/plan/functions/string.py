"""String-building free functions (`concat`, `concat_ws`, `format_string`).

All three lower to existing IR — the `concat` binary op (SQL ``||``), `array` +
`list.join`, and casts — so they add public surface without touching the engine.
Null handling matches DuckDB: `concat`/`concat_ws` treat NULL as absent (the
differential oracle), not null-propagating like the raw ``||`` operator.
"""

from __future__ import annotations

import re

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import Binary, Coalesce, Expr, IntoExpr, Lit, _wrap
from batcher.plan.expr_ir.nodes import Array, ListJoin

__all__ = [
    "concat",
    "concat_ws",
    "extract_after",
    "extract_between",
    "extract_choice",
    "extract_code_block",
    "extract_first_number",
    "extract_json",
    "extract_json_array",
    "extract_reasoning",
    "extract_tag",
    "format_string",
    "is_refusal",
    "strip_reasoning",
]


def concat(*exprs: IntoExpr) -> Expr:
    """Concatenate values into one string (DuckDB/Spark ``concat``).

    Each argument is cast to text; NULLs are treated as the empty string (DuckDB
    semantics), so ``concat("a", lit(None), "b")`` is ``"ab"`` — unlike the raw
    ``a || b`` operator, which propagates NULL. Requires at least one argument.

    Args:
        exprs: The values to concatenate, cast to text (nulls treated as empty).

    Returns:
        A string expression joining every argument.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": ["x", "y"], "b": ["1", "2"]})
            >>> ds.select(c=bt.concat(bt.col("a"), bt.col("b"))).to_pydict()
            {'c': ['x1', 'y2']}
    """
    if not exprs:
        raise PlanError("concat() requires at least one argument")
    # NULL → '' so a null contributes nothing (DuckDB concat, not `||`).
    parts = [Coalesce([_wrap(e).cast("string"), Lit("")]) for e in exprs]
    result = parts[0]
    for part in parts[1:]:
        result = Binary("concat", result, part)
    return result


def concat_ws(separator: str, *exprs: IntoExpr) -> Expr:
    """Concatenate values with `separator` between them (DuckDB/Spark ``concat_ws``).

    NULL arguments are skipped entirely — no doubled separator — matching DuckDB:
    ``concat_ws(",", "a", lit(None), "b")`` is ``"a,b"``. Each argument is cast to
    text. Requires at least one value argument.

    Args:
        separator: The text inserted between adjacent non-null values.
        exprs: The values to concatenate, cast to text (nulls skipped).

    Returns:
        A string expression joining the arguments with ``separator``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": ["x", "y"], "b": ["1", "2"]})
            >>> ds.select(c=bt.concat_ws("-", bt.col("a"), bt.col("b"))).to_pydict()
            {'c': ['x-1', 'y-2']}
    """
    if not exprs:
        raise PlanError("concat_ws() requires at least one value argument")
    # array(...).list.join skips nulls, which is exactly concat_ws's contract.
    # `list.join` of an all-null (non-empty) list is NULL, but DuckDB `concat_ws`
    # returns the empty string when every value argument is NULL — coalesce to "".
    elements = [_wrap(e).cast("string") for e in exprs]
    return Coalesce([ListJoin(Array(elements), separator), Lit("")])


def format_string(format: str, *exprs: IntoExpr) -> Expr:
    """Interpolate values into a template with ``{}`` placeholders (Polars ``format``).

    ``format_string("{} = {}", col("k"), col("v"))`` yields ``"k = v"`` per row. The
    number of ``{}`` placeholders must equal the number of arguments. Values are cast
    to text with the same NULL-as-empty rule as :func:`concat`. The placeholder is the
    literal two-character ``{}`` (no printf width/precision — keep formatting in SQL).

    Args:
        format: The template string with one ``{}`` per value argument.
        exprs: The values to interpolate, cast to text (nulls treated as empty).

    Returns:
        A string expression with each ``{}`` replaced by its argument.

    Raises:
        PlanError: If the number of ``{}`` placeholders differs from the argument count.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": ["x", "y"], "b": ["1", "2"]})
            >>> ds.select(c=bt.format_string("{}={}", bt.col("a"), bt.col("b"))).to_pydict()
            {'c': ['x=1', 'y=2']}
    """
    chunks = format.split("{}")
    if len(chunks) - 1 != len(exprs):
        raise PlanError(
            f"format_string: {len(exprs)} argument(s) but {len(chunks) - 1} '{{}}' placeholder(s)"
        )
    parts: list[IntoExpr] = []
    for i, chunk in enumerate(chunks):
        if chunk:
            parts.append(Lit(chunk))
        if i < len(exprs):
            parts.append(exprs[i])
    if not parts:
        return Lit("")
    return concat(*parts)


def _text(value: str | Expr) -> Expr:
    """A column reference from a name, else the expression as-is, for the `.str` accessor."""
    from batcher.plan.expr_ir.constructors import col

    return col(value) if isinstance(value, str) else value


def extract_json(text: str | Expr) -> Expr:
    """The first JSON object substring in a text column — the LLM-output-to-JSON extractor.

    A language model asked for JSON wraps it in prose ("Sure, here is the JSON: {...}"), which
    breaks a bare ``json.loads``. This pulls out the first ``{...}`` span so the result parses. It
    matches to the last closing brace, which is correct for one top-level object; use
    `extract_json_array` for a top-level list.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        A string expression of the first JSON object, or an empty string where none is present.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ['answer: {"a": 1, "b": [2, 3]}']})
            >>> ds.select(j=bt.extract_json("o")).to_pydict()["j"][0]
            '{"a": 1, "b": [2, 3]}'
    """
    return _text(text).str.regexp_extract(r"\{[\s\S]*\}", 0)


def extract_json_array(text: str | Expr) -> Expr:
    """The first JSON array substring in a text column.

    The list counterpart of `extract_json`: pulls out the first ``[...]`` span from a model's
    prose-wrapped output so it parses as a JSON array.

    Args:
        text: The generated-text column.

    Returns:
        A string expression of the first JSON array, or an empty string where none is present.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["the items are [1, 2, 3] total"]})
            >>> ds.select(a=bt.extract_json_array("o")).to_pydict()["a"][0]
            '[1, 2, 3]'
    """
    return _text(text).str.regexp_extract(r"\[[\s\S]*\]", 0)


def extract_code_block(text: str | Expr) -> Expr:
    """The contents of the first fenced code block in a text column, without the fences.

    A model that returns code wraps it in triple-backtick fences, optionally with a language tag.
    This extracts just the code inside the first fence, dropping the backtick markers and the
    language, which is what you want before writing the code to a file or running it.

    Args:
        text: The generated-text column.

    Returns:
        A string expression of the first code block's contents, or an empty string where none is
        present.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["run this: ```print(1)```"]})
            >>> ds.select(c=bt.extract_code_block("o")).to_pydict()["c"][0]
            'print(1)'
    """
    return _text(text).str.regexp_extract(r"```(?:[a-zA-Z0-9_+-]*\n)?([\s\S]*?)```", 1)


def extract_first_number(text: str | Expr) -> Expr:
    """The first number in a text column, parsed to a float — the score/rating extractor.

    The model's answer to "rate this 1-10" or "how many" is a number buried in a sentence. This
    pulls the first integer or decimal (with an optional leading minus) and parses it to a
    ``Float64``, so a rating or count is a numeric column ready to aggregate. A row with no number
    becomes null rather than erroring.

    Args:
        text: The generated-text column.

    Returns:
        A Float64 expression of the first number, or null where none is present.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["I rate it 8 out of 10", "no number"]})
            >>> ds.select(n=bt.extract_first_number("o")).to_pydict()["n"]
            [8.0, None]
    """
    return _text(text).str.regexp_extract(r"-?\d+\.?\d*", 0).try_cast("float64")


def extract_tag(text: str | Expr, tag: str) -> Expr:
    """The contents of the first ``<tag>...</tag>`` block in a text column.

    Structured-output and reasoning models wrap parts of their answer in XML-like tags — an
    ``<answer>``, a ``<thinking>``, a ``<tool_call>``. This extracts what is inside the first
    matching pair, which is how you split a tagged section out of the surrounding text.

    Args:
        text: The generated-text column.
        tag: The tag name, without angle brackets.

    Returns:
        A string expression of the tag's contents, or an empty string where the tag is absent.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["<answer>Paris</answer> is the capital"]})
            >>> ds.select(a=bt.extract_tag("o", "answer")).to_pydict()["a"][0]
            'Paris'
    """
    name = re.escape(tag)
    return _text(text).str.regexp_extract(rf"<{name}>([\s\S]*?)</{name}>", 1)


def extract_reasoning(text: str | Expr) -> Expr:
    """The contents of a reasoning model's thinking block (``<think>`` or ``<thinking>``).

    Reasoning models emit their chain of thought inside a ``<think>`` (or ``<thinking>``) block
    before the final answer. This extracts that hidden reasoning — useful for auditing how a model
    reached an answer, while `strip_reasoning` removes it to keep only the answer.

    Args:
        text: The generated-text column.

    Returns:
        A string expression of the reasoning contents, or an empty string where absent.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["<think>2+2 is 4</think>The answer is 4"]})
            >>> ds.select(r=bt.extract_reasoning("o")).to_pydict()["r"][0]
            '2+2 is 4'
    """
    return _text(text).str.regexp_extract(r"<think(?:ing)?>([\s\S]*?)</think(?:ing)?>", 1)


def strip_reasoning(text: str | Expr) -> Expr:
    """Remove a reasoning model's ``<think>``/``<thinking>`` block, leaving only the answer.

    The complement of `extract_reasoning`: it strips the hidden chain of thought so the column holds
    just the user-facing answer, which is what you score, display, or store. Text outside the block
    is left untouched.

    Args:
        text: The generated-text column.

    Returns:
        A string expression with the reasoning block removed and surrounding whitespace trimmed.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["<think>2+2 is 4</think>The answer is 4"]})
            >>> ds.select(a=bt.strip_reasoning("o")).to_pydict()["a"][0]
            'The answer is 4'
    """
    return (
        _text(text).str.regexp_replace(r"<think(?:ing)?>[\s\S]*?</think(?:ing)?>", "").str.strip()
    )


def extract_after(text: str | Expr, marker: str) -> Expr:
    """The text following a literal marker — the "Answer:" / "Final answer:" extractor.

    Instruction-tuned models prefix their answer with a label ("Answer:", "Final answer:"). This
    returns everything on the same run after the first occurrence of `marker`, so the label and the
    preamble before it drop away and the bare answer remains. The marker is matched literally, not
    as a regex.

    Args:
        text: The generated-text column.
        marker: The literal marker to cut after.

    Returns:
        A string expression of the text after the marker, or an empty string where it is absent.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["Reasoning... Answer: Paris"]})
            >>> ds.select(a=bt.extract_after("o", "Answer:")).to_pydict()["a"][0]
            'Paris'
    """
    return _text(text).str.regexp_extract(re.escape(marker) + r"\s*(.*)", 1).str.strip()


def extract_between(text: str | Expr, start: str, end: str) -> Expr:
    """The text between two literal markers.

    Pulls out the substring bracketed by `start` and `end` — the value between two delimiters a
    template or a model reliably emits. Both markers are matched literally, and the shortest span is
    returned, so nested or repeated markers do not over-capture.

    Args:
        text: The generated-text column.
        start: The literal opening marker.
        end: The literal closing marker.

    Returns:
        A string expression of the text between the markers, or an empty string where the pair is
        absent.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["result=[[42]] done"]})
            >>> ds.select(v=bt.extract_between("o", "[[", "]]")).to_pydict()["v"][0]
            '42'
    """
    return _text(text).str.regexp_extract(re.escape(start) + r"([\s\S]*?)" + re.escape(end), 1)


def is_refusal(text: str | Expr) -> Expr:
    """True where a generation is a refusal — an "I can't help with that" style non-answer.

    A safety-tuned model declines some requests, and a refusal is not a wrong answer so much as a
    non-answer that should be counted separately in an eval. This flags the common refusal
    phrasings ("I can't", "I'm sorry", "as an AI", "unable to"), case-insensitively, so a refusal
    rate is one aggregate over the column.

    Args:
        text: The generated-text column.

    Returns:
        A Boolean expression, true where the text reads as a refusal.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"o": ["I'm sorry, I can't help with that.", "The answer is 4."]}
            ... )
            >>> ds.select(r=bt.is_refusal("o")).to_pydict()["r"]
            [True, False]
    """
    pattern = (
        r"(?i)\b(i can'?t|i cannot|i'?m sorry|i am sorry|as an ai|i'?m unable|unable to|i won'?t)\b"
    )
    return _text(text).str.regexp_matches(pattern)


def extract_choice(text: str | Expr) -> Expr:
    """The first standalone multiple-choice letter (A-H) in a text column.

    A model answering a multiple-choice question replies "The answer is B" or just "B". This pulls
    the first letter that stands on its own as a choice, so a benchmark answer becomes a clean label
    column to compare against the gold choice. The class stops at H so the standalone pronoun "I" is
    not mistaken for a choice. Widen the pattern yourself for a question with more than eight
    options.

    Args:
        text: The generated-text column.

    Returns:
        A string expression of the choice letter, or an empty string where none is present.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["The correct answer is B.", "I think (C) is right"]})
            >>> ds.select(c=bt.extract_choice("o")).to_pydict()["c"]
            ['B', 'C']
    """
    return _text(text).str.regexp_extract(r"\b([A-H])\b", 1)
