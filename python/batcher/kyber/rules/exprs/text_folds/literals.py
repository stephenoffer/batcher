"""The shared machinery every string-literal fold is built from.

A fold recognizes `StrFunc(fn, <literal>)`, computes the answer in Python, and replaces
the call with the resulting literal. `_fold` is the lifter; `literal_text` and
`ascii_text` are the two operand gates.

The ASCII gate is the load-bearing one. Python and Rust agree on fewer string operations
than they appear to: the digests and the byte-length functions agree exactly, while
anything touching case, whitespace, or per-character indexing may diverge outside ASCII.
A fold whose operation is locale- or Unicode-sensitive takes `ascii_text` and declines on
anything else rather than picking one engine's answer.
"""

from __future__ import annotations

from collections.abc import Callable

from batcher.plan.expr_ir import Expr, Lit
from batcher.plan.expr_ir.func_nodes import StrFunc

__all__ = ["MAX_FOLDED_CHARS", "ascii_text", "fold", "literal_text", "plain_text_fold"]

#: The largest string a fold will materialize. `repeat` and the pads are the functions
#: that can grow their input without bound, and a plan literal is copied into every
#: batch -- so a folded megabyte would be a pessimization, not an optimization.
MAX_FOLDED_CHARS = 4096


def literal_text(expr: Expr) -> str | None:
    """The Python string a literal holds, or ``None`` if it is not a string literal."""
    if isinstance(expr, Lit) and isinstance(expr.value, str):
        return expr.value
    return None


def ascii_text(expr: Expr) -> str | None:
    """As `literal_text`, but only for pure-ASCII text.

    The gate for every case-, whitespace-, or index-sensitive fold. Outside ASCII,
    Python's and Rust's answers are allowed to differ and the fold declines.
    """
    text = literal_text(expr)
    return text if text is not None and text.isascii() else None


def fold(fn: str, compute: Callable[[StrFunc], object]):
    """Build a leaf rewrite folding `StrFunc(fn, <literal>)` via `compute`.

    `compute` returns the folded Python value, or ``None`` to decline -- which is how a
    fold refuses a non-default argument, a non-ASCII operand, or an out-of-range index.
    """

    def leaf(expr: Expr) -> Expr:
        if isinstance(expr, StrFunc) and expr.fn == fn:
            out = compute(expr)
            if out is not None:
                return Lit(out)
        return expr

    return leaf


def plain_text_fold(fn: str, compute, *, ascii_only: bool = False):
    """`fold` for a function taking no arguments beyond its input."""

    def compute_one(expr: StrFunc):
        text = ascii_text(expr.input) if ascii_only else literal_text(expr.input)
        return None if text is None else compute(text)

    return fold(fn, compute_one)
