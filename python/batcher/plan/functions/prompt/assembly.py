"""Assembling a prompt from row columns — templates, tags, chat formats, retrieved context.

Building a prompt per row is columnar work, not a Python loop: interpolate columns into a template,
wrap a field in tags, put a system and a user turn into the shape a chat model expects, or fold a
list of retrieved chunks into one context block. These lower to the same `concat` and string
primitives everything else uses, so a prompt over a hundred million rows is built in the data plane
with no per-row Python. They return a row-wise string expression for `select` / `with_columns`,
unlike the aggregate metrics elsewhere in this package.

The chat formats here are the two that are *text* formats rather than an API's message list —
ChatML and the Alpaca instruction layout. A model served over a chat completions API takes
structured messages instead, which `batcher.ml.llm` builds; these are for the completions endpoint
and for local engines that want a rendered string.
"""

from __future__ import annotations

import re

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import Expr, IntoExpr, Lit
from batcher.plan.functions.aggregate import _as_column
from batcher.plan.functions.collection import element
from batcher.plan.functions.string import concat

__all__ = [
    "chatml_prompt",
    "instruction_prompt",
    "join_context",
    "render_template",
    "tagged_fields",
    "wrap_tag",
]

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def render_template(template: str, **fields: IntoExpr) -> Expr:
    """Interpolate columns into a template with named ``{placeholder}`` slots — the prompt builder.

    ``render_template("Summarize {text} in {n} words", text=col("body"), n=lit(20))`` builds one
    prompt per row by replacing each ``{name}`` with its column or value, cast to text. It is the
    named-placeholder companion of :func:`format_string` (which is positional), the natural way to
    assemble a request from several columns without a per-row Python loop. Every placeholder must
    have a matching keyword and every keyword must appear in the template.

    Args:
        template: The prompt template with ``{name}`` placeholders.
        fields: One keyword per placeholder, giving the column or value to substitute.

    Returns:
        A string expression with each placeholder replaced by its field, nulls treated as empty.

    Raises:
        PlanError: If a placeholder has no matching field, or a field is never used.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"topic": ["cats"], "n": ["3"]})
            >>> expr = bt.render_template("Write {n} facts about {topic}.", n=bt.col("n"),
            ...                           topic=bt.col("topic"))
            >>> ds.select(prompt=expr).to_pydict()["prompt"][0]
            'Write 3 facts about cats.'
    """
    parts: list[IntoExpr] = []
    used: set[str] = set()
    last = 0
    for match in _PLACEHOLDER.finditer(template):
        literal = template[last : match.start()]
        if literal:
            parts.append(Lit(literal))
        name = match.group(1)
        if name not in fields:
            raise PlanError(f"render_template: no field for placeholder '{{{name}}}'")
        parts.append(fields[name])
        used.add(name)
        last = match.end()
    tail = template[last:]
    if tail:
        parts.append(Lit(tail))
    unused = set(fields) - used
    if unused:
        raise PlanError(f"render_template: field(s) not in the template: {sorted(unused)}")
    if not parts:
        return Lit("")
    return concat(*parts)


def wrap_tag(content: IntoExpr, tag: str) -> Expr:
    """Wrap a column in an XML-style ``<tag>...</tag>`` block — for building a structured prompt.

    The inverse of `extract_tag`: it surrounds each row's value with an opening and closing tag, the
    convention a prompt uses to delimit a field ("put the document between ``<doc>`` tags") so the
    model or a later parse step can find it. Compose several with `concat` to build a tagged prompt.

    Args:
        content: The column or value to wrap.
        tag: The tag name, without angle brackets.

    Returns:
        A string expression of ``<tag>content</tag>``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"q": ["What is 2+2?"]})
            >>> ds.select(w=bt.wrap_tag(bt.col("q"), "question")).to_pydict()["w"][0]
            '<question>What is 2+2?</question>'
    """
    return concat(Lit(f"<{tag}>"), content, Lit(f"</{tag}>"))


def tagged_fields(**fields: IntoExpr) -> Expr:
    """Wrap each named field in its own ``<name>...</name>`` block and concatenate them.

    The multi-field form of `wrap_tag`, and the shape a structured prompt wants: one delimited
    block per column, in the order given, separated by newlines so the model sees them as
    distinct sections. Tag delimiters survive a value containing punctuation or newlines,
    which is why they beat a bare ``Field: value`` layout for anything a user typed.

    Args:
        fields: One keyword per block, giving the tag name and the column or value to wrap.

    Returns:
        A string expression of the concatenated blocks, newline-separated.

    Raises:
        PlanError: If no fields are given.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"q": ["Why?"], "doc": ["Because."]})
            >>> blocks = bt.tagged_fields(question=bt.col("q"), context=bt.col("doc"))
            >>> print(ds.select(p=blocks).to_pydict()["p"][0])
            <question>Why?</question>
            <context>Because.</context>
    """
    if not fields:
        raise PlanError("tagged_fields: at least one field is required")
    parts: list[IntoExpr] = []
    for i, (name, value) in enumerate(fields.items()):
        if i:
            parts.append(Lit("\n"))
        parts.append(wrap_tag(value, name))
    return concat(*parts)


def chatml_prompt(user: IntoExpr, system: IntoExpr | None = None) -> Expr:
    """Render a row's turns in the ChatML text format, ready for a completions endpoint.

    ChatML delimits each turn with ``<|im_start|>role`` and ``<|im_end|>``, and ends with an
    open assistant turn so the model continues from there. It is the rendered-string form a
    local engine or a raw completions call takes, as opposed to the structured message list a
    chat completions API wants — build that with :mod:`batcher.ml.llm` instead.

    The exact control tokens differ between model families. Check what your model was trained
    on before assuming this is the one it expects; a mismatched template degrades quality
    quietly rather than erroring.

    Args:
        user: The user turn's text column or value.
        system: The system turn, omitted from the output when not given.

    Returns:
        A string expression holding the rendered conversation with an open assistant turn.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"q": ["Hi"]})
            >>> print(ds.select(p=bt.chatml_prompt(bt.col("q"))).to_pydict()["p"][0])
            <|im_start|>user
            Hi<|im_end|>
            <|im_start|>assistant
            <BLANKLINE>
    """
    parts: list[IntoExpr] = []
    if system is not None:
        parts += [Lit("<|im_start|>system\n"), system, Lit("<|im_end|>\n")]
    parts += [
        Lit("<|im_start|>user\n"),
        user,
        Lit("<|im_end|>\n<|im_start|>assistant\n"),
    ]
    return concat(*parts)


def instruction_prompt(
    instruction: IntoExpr,
    context: IntoExpr | None = None,
    *,
    response_prefix: str = "### Response:\n",
) -> Expr:
    """Render a row in the Alpaca-style instruction layout, ending at the response header.

    The plain-text instruction format most supervised fine-tuning sets are written in: an
    instruction section, an optional input section, then the response header the model
    completes from. Use it to build training rows in the same shape you will serve in, which
    is the mismatch that most often costs a fine-tune its accuracy.

    Args:
        instruction: The instruction column or value.
        context: The optional input the instruction operates on, omitted when not given.
        response_prefix: The header the generation continues from.

    Returns:
        A string expression holding the rendered instruction prompt.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"i": ["Summarize."], "c": ["A long text."]})
            >>> built = ds.select(p=bt.instruction_prompt(bt.col("i"), bt.col("c")))
            >>> print(built.to_pydict()["p"][0])
            ### Instruction:
            Summarize.
            <BLANKLINE>
            ### Input:
            A long text.
            <BLANKLINE>
            ### Response:
            <BLANKLINE>
    """
    parts: list[IntoExpr] = [Lit("### Instruction:\n"), instruction, Lit("\n\n")]
    if context is not None:
        parts += [Lit("### Input:\n"), context, Lit("\n\n")]
    parts.append(Lit(response_prefix))
    return concat(*parts)


def join_context(chunks: IntoExpr, separator: str = "\n\n") -> Expr:
    """Fold a list column of retrieved chunks into one context block.

    The step between retrieval and generation: a vector search leaves one list of passages per
    query, and the prompt needs them as a single string. Empty and null passages are dropped
    first, so a retriever that returned fewer than `k` hits does not leave a run of blank
    separators in the middle of the context.

    Args:
        chunks: A list-of-string column holding one query's retrieved passages.
        separator: The text placed between passages.

    Returns:
        A string expression holding the joined context.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"hits": [["first passage", "", "second passage"]]})
            >>> ds.select(c=bt.join_context(bt.col("hits"), separator=" | ")).to_pydict()
            {'c': ['first passage | second passage']}
    """
    column = _as_column(chunks)
    kept = column.list.drop_nulls().list.filter(element() != Lit(""))
    return kept.list.join(separator)
