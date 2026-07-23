"""Video decode — sampling a fixed number of resized frames per clip, via PyAV.

The only decoder here that has no native kernel behind it. A clip is an independent
container with its own codec state, so the work is a bounded-concurrency loop over rows
rather than a vectorized pass; see `video_dataset` for why that is inherent to the
codec API rather than a gap waiting to be closed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.decode.stage import _bounded_map, _require_size

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["video_dataset"]


def video_dataset(
    ds: Dataset,
    *,
    size: tuple[int, int] | None,
    num_frames: int = 8,
    source_column: str = "bytes",
    output_column: str = "frames",
    seek: bool = False,
    decode_concurrency: int = 4,
) -> Dataset:
    """Decode a video-bytes column into a ``(num_frames, H, W, 3)`` uint8 tensor column.

    Samples `num_frames` evenly-spaced frames and resizes each to `size` via `PyAV`
    (``batcher-engine[video]``). Fixed frame count and size make the result a fixed-shape
    uint8 tensor, ready for a video model; undecodable rows become all-zero frames.

    **A clip is decoded in Python, one clip at a time.** That is a loop in the control
    plane, and it is unavoidable rather than fixed: PyAV exposes no batch API, and each
    row is an independent container with its own codec state, so there is nothing to
    vectorize across. What is avoidable is doing it serially — `decode_concurrency` clips
    decode at once and PyAV releases the GIL inside the codec, so the work genuinely
    overlaps. The cost is memory: peak residency is `decode_concurrency` clips rather
    than one, so lower it to ``1`` for GB-sized clips.

    `seek` changes how frames are found. By default every frame is decoded in order and
    the wanted ones kept, which is exact but linear in clip length — sampling 8 frames
    out of a 100k-frame video decodes ~100k to keep 8. With ``seek=True`` the decoder
    jumps to the keyframe before each target timestamp, far faster for sparse sampling
    but landing on *approximately* the requested frames. Use it when the sample is a
    summary of the clip, not when exact frame indices matter.
    """
    from batcher.io.formats.ml.tensor import as_tensor_column

    height, width = _require_size(size, "read.video(decode=True)")
    shape = (num_frames, height, width, 3)
    per_row = num_frames * height * width * 3

    def _one(data: bytes | None) -> Any:
        if data is None:
            return None
        return _decode_video_bytes(data, num_frames, height, width, seek)

    def _decode(batch: Any) -> Any:
        import numpy as np
        import pyarrow as pa

        src = batch.column(source_column)
        flat = np.zeros((batch.num_rows, per_row), dtype=np.uint8)
        # Materialize clip bytes lazily per row rather than via `.to_pylist()`, so at most
        # `decode_concurrency` clips are resident instead of the whole batch.
        clips = (s.as_py() if s.is_valid else None for s in src)
        for i, frames in enumerate(_bounded_map(_one, clips, max(decode_concurrency, 1))):
            if frames is not None:
                flat[i] = frames.reshape(-1)
        storage = pa.FixedSizeListArray.from_arrays(pa.array(flat.reshape(-1)), per_row)
        return batch.append_column(output_column, as_tensor_column(storage, shape))

    return ds.map_batches(_decode, output_columns=[*list(ds.columns), output_column])


def _decode_video_bytes(
    data: bytes, num_frames: int, height: int, width: int, seek: bool = False
) -> Any:
    """Sample `num_frames` resized frames from one clip, using a single container open.

    Retaining every decoded frame to sample a handful costs the whole clip in RAM (a
    1-minute 1080p clip at 30fps is ~11 GB for 8 frames of output), so only the wanted
    frames are kept and peak memory is the output plus one frame.

    The container is opened **once**. The header's advertised frame count is free when
    present but only a hint — a stream copy or a truncated file leaves it wrong — so a
    sampling pass that finds it wrong rewinds this same container and re-samples against
    a counted total instead of reopening the file. That is what collapses the three opens
    this used to cost (header probe, sample, recount-and-resample) into one.
    """
    import io

    try:
        import av
    except ImportError as exc:  # pragma: no cover - optional extra
        raise PlanError("video decode needs PyAV: pip install 'batcher-engine[video]'") from exc
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        if seek:
            return _seek_frames(container, stream, num_frames, height, width)
        out = _sample_frames(container, int(stream.frames), num_frames, height, width)
        if out is not None:
            return out
        container.seek(0, stream=stream)
        counted = sum(1 for _ in container.decode(video=0))
        container.seek(0, stream=stream)
        return _sample_frames(container, counted, num_frames, height, width)


def _sample_frames(container: Any, total: int, num_frames: int, height: int, width: int) -> Any:
    """Keep only the `num_frames` frames sampled from a clip of `total`, decoding in order.

    Returns None when `total` does not match what the container actually decodes, which is
    the caller's signal to recount rather than emit a differently-sampled result.
    """
    import numpy as np

    if total <= 0:
        return None
    idx = np.linspace(0, total - 1, num=num_frames).astype(int)
    wanted = set(idx.tolist())
    kept: dict[int, Any] = {}
    seen = 0
    for pos, frame in enumerate(container.decode(video=0)):
        seen = pos + 1
        if pos in wanted:
            kept[pos] = _resize(frame, height, width)
            if len(kept) == len(wanted):
                # Every frame we need is in hand; the tail of the clip is never decoded.
                return _stack(kept, idx, num_frames, height, width)
    if seen != total:
        return None
    return _stack(kept, idx, num_frames, height, width) if kept else None


def _seek_frames(container: Any, stream: Any, num_frames: int, height: int, width: int) -> Any:
    """Sample by seeking to each target timestamp instead of decoding the clip in order.

    Costs one keyframe-to-target decode per sampled frame rather than a decode of the
    whole clip — reading 8 frames instead of 100,000 to keep 8. The frames are
    approximate: `seek` lands on the keyframe at or before the target.
    """
    import numpy as np

    span = _stream_span(container, stream)
    if not span:
        # Nothing to seek against (no duration in the header); fall back to an in-order
        # pass, which still yields the right frames, just without the speedup.
        return _sample_frames(container, int(stream.frames), num_frames, height, width)
    # Drop the final target: seeking to the very last timestamp often lands past the last
    # decodable frame, which would leave the last sample black.
    targets = np.linspace(0, span, num=num_frames + 1)[:num_frames].astype(np.int64)
    out = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    for j, target in enumerate(targets):
        container.seek(int(target), stream=stream)
        for frame in container.decode(video=0):
            out[j] = _resize(frame, height, width)
            break
    return out


def _stream_span(container: Any, stream: Any) -> int:
    """The clip's duration in the stream's own timestamp units (what `seek` takes), or 0."""
    if stream.duration:
        return int(stream.duration)
    if container.duration and stream.time_base:
        return int(container.duration / 1_000_000 / float(stream.time_base))
    return 0


def _resize(frame: Any, height: int, width: int) -> Any:
    import numpy as np
    from PIL import Image

    rgb = frame.to_ndarray(format="rgb24")
    return np.asarray(Image.fromarray(rgb).resize((width, height)))


def _stack(kept: dict, idx: Any, num_frames: int, height: int, width: int) -> Any:
    import numpy as np

    out = np.empty((num_frames, height, width, 3), dtype=np.uint8)
    for j, k in enumerate(idx):
        out[j] = kept[int(k)]
    return out
