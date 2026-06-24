"""The `.audio` expression namespace — lazy, batch-level audio decode.

`AudioFunc` lowers to ``{"e": "audio", "fn": ...}`` IR consumed by Rust
`Expr::Audio` (symphonia-backed). Like image decode, the interpreter is the oracle
and the JIT falls back; one implementation, so the tiers cannot diverge. This moves
audio decode off the per-row Python ``map_batches`` path into the native data plane.
"""

from __future__ import annotations

from typing import Any

from batcher.plan.expr_ir.core import Expr
from batcher.plan.ir_tags import ExprTag

__all__ = ["AudioFunc", "_AudioNamespace"]


class AudioFunc(Expr):
    """An audio decode op over a binary (audio-bytes) sub-expression (via `.audio`).

    `decode` reads each clip's metadata; `to_waveform` decodes to a mono signal.
    """

    __slots__ = ("fn", "input")

    def __init__(self, fn: str, input: Expr) -> None:
        self.fn = fn
        self.input = input

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.AUDIO, "fn": self.fn, "input": self.input.to_ir()}


class _AudioNamespace:
    """Lazy audio decode: ``col("bytes").audio.decode()`` / ``.audio.to_waveform()``."""

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        self._e = e

    def decode(self) -> AudioFunc:
        """Decode audio bytes → struct ``{sample_rate, channels, num_frames,
        duration_secs}`` (WAV/FLAC; null/undecodable → null)."""
        return AudioFunc("decode", self._e)

    def to_waveform(self) -> AudioFunc:
        """Decode to a mono PCM signal → ``List<Float32>`` (channel-averaged samples;
        null/undecodable → null). The training-ingest path, native (no per-row
        Python)."""
        return AudioFunc("to_waveform", self._e)
