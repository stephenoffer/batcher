"""The `.video` expression namespace — lazy, batch-level video decode.

`VideoFunc` lowers to ``{"e": "video", "fn": ...}`` IR consumed by Rust
`Expr::Video` (FFmpeg-backed). Decode requires building the engine with the
``video`` cargo feature (system FFmpeg); without it, evaluating a `.video` op
raises a clear error. Like image/audio, the interpreter is the oracle and the JIT
falls back, so the tiers cannot diverge.
"""

from __future__ import annotations

from typing import Any

from batcher.plan.expr_ir.core import Expr
from batcher.plan.ir_tags import ExprTag

__all__ = ["VideoFunc", "_VideoNamespace"]


class VideoFunc(Expr):
    """A video decode op over a binary (video-bytes) sub-expression (via `.video`).

    `decode` reads each clip's metadata. Requires the engine's ``video`` feature.
    """

    __slots__ = ("fn", "input")

    def __init__(self, fn: str, input: Expr) -> None:
        self.fn = fn
        self.input = input

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.VIDEO, "fn": self.fn, "input": self.input.to_ir()}


class _VideoNamespace:
    """Lazy video decode: ``col("bytes").video.decode()``."""

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        self._e = e

    def decode(self) -> VideoFunc:
        """Decode video bytes → struct ``{width, height, num_frames, duration_secs,
        fps}`` (requires the ``video`` engine feature; null/undecodable → null)."""
        return VideoFunc("decode", self._e)
