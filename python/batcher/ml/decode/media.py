"""Image and audio decode — the two media kinds the data plane handles natively.

Both stages prefer a pure `with_columns` over a `map_batches`, because that is what
keeps the decode on the fully-parallel native path (`col.image.to_tensor`,
`col.audio.to_waveform`) instead of the slower opaque-UDF path. Only multi-channel
audio, which has no native kernel, falls back to a Python decoder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.decode.stage import _require_size

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["audio_dataset", "image_tensor_dataset"]


def image_tensor_dataset(
    ds: Dataset,
    *,
    size: tuple[int, int] | None,
    source_column: str = "bytes",
    output_column: str = "image",
) -> Dataset:
    """Decode an image-bytes column into a ``(H, W, 3)`` uint8 tensor column.

    The decode/resize runs natively (``col(source).image.to_tensor``), and the engine
    tags the output column with the canonical ``arrow.fixed_shape_tensor`` extension
    metadata, so it crosses the FFI already shaped as an ``(N, H, W, 3)`` training tensor
    — no per-batch re-type pass. Staying a pure ``with_columns`` (no ``map_batches``) is
    what keeps the decode on the fully-parallel native path instead of the slower
    opaque-UDF path, the difference that made image ingest the pipeline bottleneck.
    """
    from batcher.plan.expr_ir import col

    height, width = _require_size(size, "read.images(decode=True)")
    return ds.with_columns(**{output_column: col(source_column).image.to_tensor(width, height)})


def audio_dataset(
    ds: Dataset,
    *,
    source_column: str = "bytes",
    output_column: str = "waveform",
    sample_rate: int | None = None,
    mono: bool = True,
) -> Dataset:
    """Decode an audio-bytes column into a ``list<float32>`` waveform column.

    Every mono case decodes in the native data plane — ``col(...).audio.to_waveform()``
    at the source rate, or ``col(...).audio.resample(sample_rate)`` when a target rate is
    given (both Rust ``symphonia`` + sinc, no per-row Python). Only ``mono=False``
    (multi-channel output) falls back to the `soundfile`/`librosa` Python path
    (``batcher-engine[audio]``). Waveforms are variable length, so the output is a
    ``list<float32>`` column (one per row).
    """
    # Native path: mono decode (and, with a target rate, sinc resample) is exactly what
    # the Rust kernels produce, so the bytes never cross into Python per-row.
    if mono:
        from batcher.plan.expr_ir import col

        source = col(source_column)
        expr = source.audio.resample(sample_rate) if sample_rate else source.audio.to_waveform()
        return ds.with_columns(**{output_column: expr})

    # Fallback: multi-channel output needs the Python decoder.
    def _decode(batch: Any) -> Any:
        import numpy as np
        import pyarrow as pa

        raw = batch.column(source_column).to_pylist()
        waves = [_decode_audio_bytes(b, sample_rate, mono) for b in raw]
        col = pa.array(
            [None if w is None else np.asarray(w, dtype=np.float32) for w in waves],
            type=pa.list_(pa.float32()),
        )
        return batch.append_column(output_column, col)

    return ds.map_batches(_decode, output_columns=[*list(ds.columns), output_column])


def _decode_audio_bytes(data: bytes | None, sample_rate: int | None, mono: bool) -> Any:
    if data is None:
        return None
    import io

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - optional extra
        raise PlanError("audio needs soundfile: pip install 'batcher-engine[audio]'") from exc
    wave, native_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if mono:
        wave = wave.mean(axis=1)
    if sample_rate is not None and sample_rate != native_sr:
        wave = _resample(wave, native_sr, sample_rate)
    return wave


def _resample(wave: Any, src_sr: int, dst_sr: int) -> Any:
    try:
        import librosa
    except ImportError:  # pragma: no cover - resampling is best-effort
        return wave
    return librosa.resample(wave, orig_sr=src_sr, target_sr=dst_sr)
