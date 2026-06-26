"""The `.image` expression namespace — lazy, batch-level image decode.

`ImageFunc` lowers to ``{"e": "image", "fn": ...}`` IR consumed by Rust
`Expr::Image`. Decoding is library-backed, so the interpreter is the oracle and
the JIT falls back; there is one implementation, so the tiers cannot diverge.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.expr_ir.node_base import IRNode, child, expr_node, scalar
from batcher.plan.ir_tags import ExprTag

__all__ = ["ImageFunc", "_ImageNamespace"]


@expr_node
class ImageFunc(IRNode):
    """An image decode op over a binary (image-bytes) sub-expression (via `.image`).

    `decode` reads each image's dimensions; `to_tensor` decodes, resizes to
    ``(width, height)``, and flattens to a fixed-size RGB8 pixel list.
    """

    tag = ExprTag.IMAGE
    fn: str = scalar()
    input: Expr = child()
    width: int | None = scalar(omit_none=True, default=None)
    height: int | None = scalar(omit_none=True, default=None)


class _ImageNamespace:
    """Lazy image decode: ``col("bytes").image.decode()`` / ``.image.to_tensor(224, 224)``."""

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        self._e = e

    def decode(self) -> ImageFunc:
        """Decode image bytes → struct ``{width, height}`` (Int32 dimensions)."""
        return ImageFunc("decode", self._e)

    def to_tensor(self, width: int, height: int) -> ImageFunc:
        """Decode + resize to ``(width, height)`` → ``FixedSizeList<u8>`` (H·W·3, RGB8)."""
        return ImageFunc("to_tensor", self._e, width=width, height=height)

    def resize(self, width: int, height: int) -> ImageFunc:
        """Resize image bytes to ``(width, height)``, re-encoded as PNG bytes (Daft
        ``image.resize``). Null/undecodable input → null. → Binary."""
        return ImageFunc("resize", self._e, width=width, height=height)
