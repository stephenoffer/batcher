"""The `.video` expression namespace — lazy, batch-level video decode.

`VideoFunc` lowers to ``{"e": "video", "fn": ...}`` IR consumed by Rust
`Expr::Video` (FFmpeg-backed). Decode requires building the engine with the
``video`` cargo feature (system FFmpeg); without it, evaluating a `.video` op
raises a clear error. Like image/audio, the interpreter is the oracle and the JIT
falls back, so the tiers cannot diverge.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.expr_ir.node_base import IRNode, child, expr_node, scalar
from batcher.plan.ir_tags import ExprTag

__all__ = ["VideoFunc", "_VideoNamespace"]


@expr_node
class VideoFunc(IRNode):
    """A video decode op over a binary (video-bytes) sub-expression (via `.video`).

    `decode` reads each clip's metadata. Requires the engine's ``video`` feature.
    """

    tag = ExprTag.VIDEO
    fn: str = scalar()
    input: Expr = child()


class _VideoNamespace:
    """Lazy video decode: ``col("bytes").video.decode()``.

    Decoding runs in the Rust data plane over a binary column (FFmpeg-backed), so a
    video pipeline never materializes frames in Python. Evaluating a `.video` op needs
    the engine built with the ``video`` cargo feature; building the expression does not.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.col("clip").video.decode().to_ir()["fn"]
            'decode'
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        """Wrap the parent :class:`Expr` so its `.video` methods can build on it."""
        self._e = e

    def __repr__(self) -> str:
        """Show the accessor and its parent, e.g. ``<.video accessor of col('c')>``."""
        return f"<.video accessor of {self._e!r}>"

    def decode(self) -> VideoFunc:
        """Read each clip's metadata without materializing its frames.

        Requires the engine built with the ``video`` cargo feature (system FFmpeg);
        without it, evaluating this op raises a clear error rather than returning null.

        Returns:
            An expression evaluating to a struct ``{width, height, num_frames,
            duration_secs, fps}``; null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.video("s3://bucket/clips/")  # doctest: +SKIP
                >>> ds.select(m=bt.col("bytes").video.decode()).to_pydict()  # doctest: +SKIP
                {'m': [{'width': 1920, 'height': 1080, ...}]}
        """
        return VideoFunc("decode", self._e)
