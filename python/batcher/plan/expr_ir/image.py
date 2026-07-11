"""The `.image` expression namespace — lazy, batch-level image decode.

`ImageFunc` lowers to ``{"e": "image", "fn": ...}`` IR consumed by Rust
`Expr::Image`. Decoding is library-backed, so the interpreter is the oracle and
the JIT falls back; there is one implementation, so the tiers cannot diverge.
"""

from __future__ import annotations

import base64

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


# A 1x1 red PNG. Exported so the doctests here (and the docs) have a real image to
# decode without reaching for a fixture file.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class _ImageNamespace:
    """Lazy image decode: ``col("bytes").image.decode()`` / ``.image.to_tensor(224, 224)``.

    Decoding runs in the Rust data plane over a binary column, so an image pipeline
    never materializes pixels in Python. Null or undecodable input yields null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.plan.expr_ir.image import _PNG_1X1
            >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
            >>> ds.select(dims=bt.col("img").image.decode()).to_pydict()
            {'dims': [{'width': 1, 'height': 1}]}
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        """Wrap the parent :class:`Expr` so its `.image` methods can build on it."""
        self._e = e

    def __repr__(self) -> str:
        """Show the accessor and its parent, e.g. ``<.image accessor of col('c')>``."""
        return f"<.image accessor of {self._e!r}>"

    def decode(self) -> ImageFunc:
        """Read each image's dimensions without materializing its pixels.

        Only the image header is parsed, so this succeeds — and is cheap — even for a
        file whose pixel data is truncated or corrupt. Use `to_tensor` when the pixels
        themselves must be valid.

        Returns:
            An expression evaluating to a struct ``{width, height}`` of Int32
            dimensions; null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> ds.select(d=bt.col("img").image.decode()).to_pydict()
                {'d': [{'width': 1, 'height': 1}]}
        """
        return ImageFunc("decode", self._e)

    def to_tensor(self, width: int, height: int) -> ImageFunc:
        """Decode and resize to ``(width, height)``, flattened to RGB8 pixels.

        The training-ingest path: it produces a fixed-size numeric column that feeds
        a model directly, with no per-row Python.

        Args:
            width: Target width in pixels.
            height: Target height in pixels.

        Returns:
            An expression evaluating to a ``FixedSizeList<u8>`` of ``height * width * 3``
            RGB8 samples; null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> len(ds.select(t=bt.col("img").image.to_tensor(2, 2)).to_pydict()["t"][0])
                12
        """
        return ImageFunc("to_tensor", self._e, width=width, height=height)

    def resize(self, width: int, height: int) -> ImageFunc:
        """Resize the image and re-encode it as PNG bytes.

        Args:
            width: Target width in pixels.
            height: Target height in pixels.

        Returns:
            An expression evaluating to Binary PNG bytes; null for null or
            undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> small = bt.col("img").image.resize(2, 2)
                >>> ds.select(d=small.image.decode()).to_pydict()
                {'d': [{'width': 2, 'height': 2}]}
        """
        return ImageFunc("resize", self._e, width=width, height=height)
