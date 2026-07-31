"""Pulling structure back out of a model's prose-wrapped output.

A generation is a string, and a string is not a column anyone can filter, join, or grade. Each
function here recovers one fragment a model wraps its answer in — a JSON object, a fenced code
block, an ``<answer>`` tag, a multiple-choice letter, a LaTeX box, a citation marker — and each
degrades to an empty value rather than an error when the fragment is absent, so one malformed
generation cannot abort a scan over millions of rows.

They are thin wrappers over the `.str` regex primitives, which means the contract worth knowing
is the *pattern*: what it matches, and what it quietly does not. Each says so in its own
documentation. The engine's regex engine has no lookahead or backreferences, so a pattern that
would need one is composed from list operations instead.
"""

from __future__ import annotations

import re

from batcher.plan.expr_ir.core import Expr

__all__ = [
    "extract_after",
    "extract_between",
    "extract_boxed",
    "extract_choice",
    "extract_citations",
    "extract_code_block",
    "extract_first_number",
    "extract_json",
    "extract_json_array",
    "extract_last_number",
    "extract_reasoning",
    "extract_tag",
    "is_refusal",
    "strip_reasoning",
]


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


def extract_boxed(text: str | Expr) -> Expr:
    r"""The contents of the first LaTeX ``\boxed{...}`` in a text column.

    Math benchmarks such as MATH ask the model to put its final answer in a ``\boxed{}`` and the
    grader reads only that. `boxed_answer_rate` reports how often the model complied; this is
    the answer itself, as a column to compare against the gold one.

    Extraction stops at the first closing brace, so a box containing nested braces (a fraction,
    a matrix) comes back truncated. That is a real limit of a non-recursive match rather than a
    choice — check for a trailing unbalanced brace before trusting a structured answer.

    Args:
        text: The generated-text column.

    Returns:
        A string expression of the boxed answer, or an empty string where there is none.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["so the answer is \\boxed{42}.", "no box here"]})
            >>> ds.select(a=bt.extract_boxed("o")).to_pydict()["a"]
            ['42', '']
    """
    return extract_between(text, "\\boxed{", "}")


def extract_last_number(text: str | Expr) -> Expr:
    """The last number in a text column, parsed to a float — the reasoning-chain answer.

    The companion to `extract_first_number`, and usually the one a math or arithmetic
    evaluation wants. A model that reasons before answering emits its intermediate quantities
    first, so the first number is a step and the last is the conclusion: "12 apples, minus 4,
    leaves 8" grades on the 8, and `extract_first_number` would grade it on the 12.

    Thousands separators are not understood — ``1,234`` reads as ``234`` — so strip them first
    on a corpus that has them. A row with no number becomes null rather than erroring.

    Args:
        text: The generated-text column.

    Returns:
        A Float64 expression of the last number, or null where none is present.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["12 apples minus 4 leaves 8", "no digits"]})
            >>> ds.select(n=bt.extract_last_number("o")).to_pydict()["n"]
            [8.0, None]
    """
    # Every match, then the last one: the engine's regex has no lookahead, and a "last
    # occurrence" pattern needs one. `regexp_extract_all` walks the string once either way.
    numbers = _text(text).str.regexp_extract_all(r"-?\d+\.?\d*", 0)
    return numbers.list.last().try_cast("float64")


def extract_citations(text: str | Expr) -> Expr:
    """Every bracketed citation marker in a text column, as a list of numbers.

    A grounded answer cites its sources as ``[1]``, ``[2]``. `citation_rate` reports how often
    the model cited anything; this is *which* sources it cited, which is what you need to check
    them: join the list against the retrieved passages to find a citation pointing at a passage
    that was never retrieved, the most common form of a fabricated reference.

    Markers repeat naturally when a source is cited more than once, and they are returned as
    they appear — deduplicate with `list.unique` when counting distinct sources.

    Args:
        text: The generated-text column.

    Returns:
        A List<Utf8> expression of the citation numbers, empty where there are none.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["Paris [1] is the capital [2][1]."]})
            >>> ds.select(c=bt.extract_citations("o")).to_pydict()["c"]
            [['1', '2', '1']]
    """
    return _text(text).str.regexp_extract_all(r"\[(\d+)\]", 1)
