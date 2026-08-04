"""Video decode — sampling a fixed number of resized frames per clip.

Two decoders, and which one runs is a property of the *engine build* rather than of the
pipeline. An engine built with the ``video`` cargo feature carries an FFmpeg-backed
`.video.frames` kernel, and this stage lowers to it: the decode then happens in the data
plane, row-parallel, with no Python in the loop at all. An engine built without it has no
video codec to reach, so the stage falls back to the PyAV loop below.

The fallback is a per-row Python loop, which the architecture otherwise forbids. It earns
its exception the way `core/gpu_plan` does — there is nothing else to call — and it is
kept honest by being *only* a fallback: `engine_features()` decides, so a build that can
do this natively never takes it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher._internal.native import engine_features
from batcher.ml.decode.stage import (
    _bounded_map,
    _require_frames,
    _require_size,
    _require_source_column,
)

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
    error_column: str | None = None,
) -> Dataset:
    """Decode a video-bytes column into a ``(num_frames, H, W, 3)`` uint8 tensor column.

    Samples `num_frames` evenly-spaced frames and resizes each to `size`. Fixed frame
    count and size make the result a fixed-shape uint8 tensor, ready for a video model.

    **Where the decode runs depends on the engine build.** When the engine carries the
    ``video`` feature this lowers to the native ``col(...).video.frames(...)`` kernel and
    stays a pure `with_columns`, so the clips are decoded row-parallel in the data plane
    with no Python in the loop — the same shape the image path has. There, an undecodable
    clip is *null*, which is what `error_column` exists to reconstruct on the fallback.

    Otherwise the engine has no video codec to reach and a clip is decoded in Python via
    PyAV (``batcher-engine[video]``), one clip at a time. What is avoidable there is doing
    it serially — `decode_concurrency` clips decode at once and PyAV releases the GIL
    inside the codec, so the work genuinely overlaps. The cost is memory: peak residency is
    `decode_concurrency` clips rather than one, so lower it to ``1`` for GB-sized clips.

    **Both paths give the same answer**, which is the point of choosing between them: an
    undecodable clip is *null* either way, never all-zero frames. Zeros would be
    indistinguishable from a legitimately black clip, so they put blank samples into a
    training set with nothing to detect them by. Set `error_column` to append a boolean
    column that is ``True`` exactly for the rows whose bytes were present but would not
    decode, which separates a decode failure from an input that was null to begin with.

    `seek` and `decode_concurrency` apply to the Python fallback only; the native kernel
    always samples exact frame indices and parallelizes across rows on its own. By default
    the fallback decodes every frame in order and keeps the wanted ones, which is exact but
    linear in clip length — sampling 8 frames out of a 100k-frame video decodes ~100k to
    keep 8. With ``seek=True`` it jumps to the keyframe before each target timestamp, far
    faster for sparse sampling but landing on *approximately* the requested frames.
    """
    from batcher.io.formats.ml.tensor import as_tensor_column

    _require_source_column(ds, source_column, who="video_dataset", param="source_column=")
    num_frames = _require_frames(num_frames, "video_dataset")
    height, width = _require_size(size, "read.video(decode=True)")
    shape = (num_frames, height, width, 3)
    per_row = num_frames * height * width * 3

    if "video" in engine_features():
        return _native_frames(
            ds,
            num_frames=num_frames,
            width=width,
            height=height,
            source_column=source_column,
            output_column=output_column,
            error_column=error_column,
        )

    def _one(data: bytes | None) -> Any:
        if data is None:
            return None
        return _decode_video_bytes(data, num_frames, height, width, seek)

    def _decode(batch: Any) -> Any:
        import numpy as np
        import pyarrow as pa

        src = batch.column(source_column)
        flat = np.zeros((batch.num_rows, per_row), dtype=np.uint8)
        # A row failed only if its bytes were present (`is_valid`) yet decoded to nothing —
        # a null-input row is not a failure. The validity bits are cheap (no bytes copied).
        valid = src.is_valid().to_numpy(zero_copy_only=False) if error_column else None
        failed = np.zeros(batch.num_rows, dtype=bool) if error_column else None
        # A row that produced no frames is *null*, not zeros. Zeros are indistinguishable
        # from a legitimately black clip, so they put blank samples into a training set
        # with nothing to detect them by — and they would make this path disagree with the
        # native kernel, which nulls, about what the same call means.
        missing = np.zeros(batch.num_rows, dtype=bool)
        # Materialize clip bytes lazily per row rather than via `.to_pylist()`, so at most
        # `decode_concurrency` clips are resident instead of the whole batch.
        clips = (s.as_py() if s.is_valid else None for s in src)
        for i, frames in enumerate(_bounded_map(_one, clips, max(decode_concurrency, 1))):
            if frames is not None:
                flat[i] = frames.reshape(-1)
            else:
                missing[i] = True
                if error_column and valid[i]:
                    failed[i] = True
        storage = pa.FixedSizeListArray.from_arrays(
            pa.array(flat.reshape(-1)), per_row, mask=pa.array(missing)
        )
        out = batch.append_column(output_column, as_tensor_column(storage, shape))
        if error_column:
            out = out.append_column(error_column, pa.array(failed))
        return out

    appended = [output_column, *([error_column] if error_column else [])]
    return ds.map_batches(_decode, output_columns=[*list(ds.columns), *appended])


def _native_frames(
    ds: Dataset,
    *,
    num_frames: int,
    width: int,
    height: int,
    source_column: str,
    output_column: str,
    error_column: str | None,
) -> Dataset:
    """Lower the stage onto the engine's `.video.frames` kernel.

    Staying a pure `with_columns` — no `map_batches` — is what keeps the decode on the
    fully-parallel native path rather than the opaque-UDF one, the same distinction that
    made image ingest fast. It is also why `error_column` is derived from the *result*
    here rather than recorded during a loop: a row failed exactly when its bytes were
    present and its frames came back null, which is a two-column predicate the engine can
    evaluate itself.
    """
    from batcher.plan.expr_ir import col

    frames = col(source_column).video.frames(num_frames, width, height)
    out = ds.with_columns(**{output_column: frames})
    if error_column:
        out = out.with_columns(
            **{error_column: col(source_column).is_not_null() & col(output_column).is_null()}
        )
    return out


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
