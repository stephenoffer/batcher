"""Prompt-construction functions — assemble an LLM prompt from row columns, in the engine.

Building a prompt per row is columnar work, not a Python loop: interpolate columns into a template,
wrap a field in tags for a structured prompt, or trim a column to fit a context budget. These lower
to the same `concat` and string primitives everything else uses, so a prompt over a hundred million
rows is built in the data plane with no per-row Python. They return a row-wise string expression for
`select` / `with_columns`, unlike the aggregate metrics elsewhere in this package.
"""

from __future__ import annotations

import re

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import Expr, IntoExpr, Lit
from batcher.plan.functions.aggregate import _as_column
from batcher.plan.functions.string import concat

__all__ = [
    "render_template",
    "truncate_to_token_budget",
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


def truncate_to_token_budget(text: str | Expr, budget: int, chars_per_token: float = 4.0) -> Expr:
    """Trim a text column to fit an estimated token budget — keep a prompt inside the window.

    Cuts each value to ``budget * chars_per_token`` characters, the tokenizer-free way to keep an
    assembled prompt within a model's context window before generation truncates it silently. It is
    a character cut on an estimate, so leave headroom rather than targeting the exact window size.

    Args:
        text: The text column (name or expression) to trim.
        budget: The estimated token budget to fit within.
        chars_per_token: The average characters per token used for the estimate.

    Returns:
        A string expression trimmed to the estimated budget.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"t": ["hello world"]})
            >>> ds.select(r=bt.truncate_to_token_budget("t", budget=1)).to_pydict()
            {'r': ['hell']}
    """
    char_budget = int(budget * chars_per_token)
    return _as_column(text).str.truncate_chars(char_budget)
