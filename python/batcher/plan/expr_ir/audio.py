"""The `.audio` expression namespace — lazy, batch-level audio decode.

`AudioFunc` lowers to ``{"e": "audio", "fn": ...}`` IR consumed by Rust
`Expr::Audio` (symphonia-backed). Like image decode, the interpreter is the oracle
and the JIT falls back; one implementation, so the tiers cannot diverge. This moves
audio decode off the per-row Python ``map_batches`` path into the native data plane.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.expr_ir.node_base import IRNode, child, expr_node, scalar
from batcher.plan.ir_tags import ExprTag

__all__ = ["AudioFunc", "_AudioNamespace"]


@expr_node
class AudioFunc(IRNode):
    """An audio decode op over a binary (audio-bytes) sub-expression (via `.audio`).

    `decode` reads each clip's metadata; `to_waveform` decodes to a mono signal;
    `resample` decodes then band-limited-resamples that signal to `rate` Hz.
    """

    tag = ExprTag.AUDIO
    fn: str = scalar()
    input: Expr = child()
    rate: int | None = scalar(omit_none=True, default=None)


class _AudioNamespace:
    """Lazy audio decode: ``col("bytes").audio.decode()`` / ``.audio.to_waveform()``.

    Decoding runs in the Rust data plane over a binary column (symphonia-backed), so an
    audio pipeline never materializes samples in Python. Null or undecodable input —
    including bytes that are not audio at all — yields null rather than raising.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"clip": [b"not audio"]})
            >>> ds.select(meta=bt.col("clip").audio.decode()).to_pydict()
            {'meta': [None]}
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        """Wrap the parent :class:`Expr` so its `.audio` methods can build on it."""
        self._e = e

    def __repr__(self) -> str:
        """Show the accessor and its parent, e.g. ``<.audio accessor of col('c')>``."""
        return f"<.audio accessor of {self._e!r}>"

    def decode(self) -> AudioFunc:
        """Read each clip's metadata without materializing its samples.

        Returns:
            An expression evaluating to a struct ``{sample_rate, channels, num_frames,
            duration_secs}`` (WAV/FLAC); null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.audio("s3://bucket/clips/")  # doctest: +SKIP
                >>> ds.select(m=bt.col("bytes").audio.decode()).to_pydict()  # doctest: +SKIP
                {'m': [{'sample_rate': 44100, 'channels': 2, ...}]}
        """
        return AudioFunc("decode", self._e)

    def to_waveform(self) -> AudioFunc:
        """Decode to a mono PCM signal by averaging channels.

        The training-ingest path: it produces a numeric column that feeds a model
        directly, with no per-row Python.

        Returns:
            An expression evaluating to a ``List<Float32>`` of channel-averaged
            samples; null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.audio("s3://bucket/clips/")  # doctest: +SKIP
                >>> ds.select(w=bt.col("bytes").audio.to_waveform())  # doctest: +SKIP
        """
        return AudioFunc("to_waveform", self._e)

    def resample(self, rate: int) -> AudioFunc:
        """Decode to mono and band-limited-resample to ``rate`` Hz.

        The audio-ML preprocessing step (models expect a fixed rate — 16 kHz for
        Whisper/wav2vec, 22 kHz for many audio models). Sinc resampling runs natively in
        the data plane over the whole batch, replacing a per-file Python ``librosa`` call.
        The output length is ``ceil(n * rate / source_rate)`` — the length ``librosa``
        produces — so a resampled frame count is reproducible and engine-independent.

        Args:
            rate: The target sample rate in Hz (must be positive).

        Returns:
            An expression evaluating to a ``List<Float32>`` of resampled mono samples;
            null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.audio("s3://bucket/clips/")  # doctest: +SKIP
                >>> ds.select(w=bt.col("bytes").audio.resample(16000))  # doctest: +SKIP
        """
        return AudioFunc("resample", self._e, rate=rate)
