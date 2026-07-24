"""Markdown-formatting metrics — did the model return the structured layout the task asked for.

A task that asks for a heading, a bulleted list, a table, or fenced code only succeeds if the
model actually emits that Markdown element, and the failure is quiet: a prompt tweak that drops the
formatting produces prose that still reads fine but no longer parses or renders as intended. These
measure the presence rate of each element directly, as a corpus number, so a regression in
structured formatting shows up before a reader or a renderer notices. Each is a single mergeable
aggregate over the string primitives and composes inside `group_by`.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import lit
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if

__all__ = [
    "bullet_list_rate",
    "code_block_present_rate",
    "heading_rate",
    "markdown_link_rate",
    "numbered_list_rate",
    "table_rate",
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
