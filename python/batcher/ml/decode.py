"""Decode media columns into tensors — native image decode, Python audio/video.

Multimodal sources read references + header metadata only (no pixels/samples at read
time). These helpers turn the raw ``bytes`` column into model-ready tensors at the
point a pipeline asks for it:

* **images** decode in the **Rust data plane** — the existing ``col.image.to_tensor``
  kernel resizes and flattens to RGB8 in the engine; here we only re-type the result
  (zero-copy) into a fixed-shape ``(H, W, 3)`` tensor column.
* **audio / video** decode in **Python UDFs** (soundfile / PyAV behind optional
  extras), because their codecs live in those libraries; decoding stays whole-batch.

Each returns a new lazy `Dataset`, so decode composes with the rest of a pipeline.
A row whose bytes are null or fail to decode yields a null (image) or zero (video)
result rather than failing the batch — the multimodal convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "audio_dataset",
    "download_dataset",
    "image_tensor_dataset",
    "upload_dataset",
    "video_dataset",
]


def upload_dataset(
    ds: Dataset,
    *,
    data_column: str,
    directory: str,
    output_column: str = "path",
    name_column: str | None = None,
    extension: str = "",
    max_concurrency: int = 16,
) -> Dataset:
    """Write each row's bytes to a file under `directory`, appending the written path.

    The counterpart to `download_dataset` (cf. Daft's ``url.upload``) — write decoded
    or transformed media back to ``s3://`` / ``gs://`` / ``az://`` / local storage.
    File names come from `name_column` (plus `extension`) or a content-addressed hash
    when no name column is given (collision-free across distributed workers). Writes
    each batch's rows concurrently and parallelizes across the cluster.

    Args:
        data_column: the binary column to write.
        directory: the destination directory/prefix.
        output_column: the appended column of written paths.
        name_column: optional column of file names (else a content hash is used).
        extension: appended to the file name (e.g. ``".jpg"``).
        max_concurrency: concurrent writes per batch.
    """
    base = directory.rstrip("/")

    def _write(name: str, data: bytes | None) -> str | None:
        if data is None:
            return None
        from batcher.io.filesystem import resolve_filesystem

        path = f"{base}/{name}{extension}"
        with resolve_filesystem(path).atomic_writer(path) as handle:
            handle.write(data)
        return path

    def _udf(batch: Any) -> Any:
        import hashlib
        from concurrent.futures import ThreadPoolExecutor

        import pyarrow as pa

        data = batch.column(data_column).to_pylist()
        if name_column is not None:
            names = [str(n) for n in batch.column(name_column).to_pylist()]
        else:
            names = [hashlib.sha1(b).hexdigest() if b is not None else "" for b in data]
        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            paths = list(pool.map(_write, names, data))
        col = pa.array(paths, type=pa.large_string())
        if output_column in batch.schema.names:
            return batch.set_column(batch.schema.get_field_index(output_column), output_column, col)
        return batch.append_column(output_column, col)

    out_cols = list(ds.columns) if output_column in ds.columns else [*ds.columns, output_column]
    return ds.map_batches(_udf, output_columns=out_cols)


def download_dataset(
    ds: Dataset,
    *,
    url_column: str,
    output_column: str = "bytes",
    max_concurrency: int = 16,
    on_error: str = "raise",
) -> Dataset:
    """Fetch the bytes at each URL/path into a ``large_binary`` column.

    The entry point of a multimodal pipeline (URL table → bytes → decode → model),
    the counterpart to Daft's ``col(url).url.download()``. Reads ``s3://`` / ``gs://`` /
    ``az://`` / ``http(s)://`` / local paths through the shared filesystem resolver,
    fetching the rows of each batch **concurrently** (`max_concurrency` connections).
    Runs as a `map_batches` stage, so it parallelizes across the Ray cluster like any
    other operator. `on_error="null"` turns a failed/missing fetch into a null instead
    of raising — so one bad URL doesn't kill the job.

    Args:
        url_column: the column of URLs/paths to fetch.
        output_column: the appended (or replaced) bytes column.
        max_concurrency: concurrent fetches per batch (I/O-bound, GIL-releasing).
        on_error: ``"raise"`` (default) or ``"null"``.
    """
    if on_error not in ("raise", "null"):
        raise PlanError(f"download on_error must be 'raise' or 'null', got {on_error!r}")

    def _fetch(url: str | None) -> bytes | None:
        if url is None:
            return None
        from batcher.io.filesystem import resolve_filesystem

        try:
            with resolve_filesystem(url).open(url) as handle:
                return handle.read()
        except Exception:
            if on_error == "null":
                return None
            raise

    def _udf(batch: Any) -> Any:
        from concurrent.futures import ThreadPoolExecutor

        import pyarrow as pa

        urls = batch.column(url_column).to_pylist()
        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            data = list(pool.map(_fetch, urls))
        col = pa.array(data, type=pa.large_binary())
        if output_column in batch.schema.names:
            return batch.set_column(batch.schema.get_field_index(output_column), output_column, col)
        return batch.append_column(output_column, col)

    out_cols = list(ds.columns) if output_column in ds.columns else [*ds.columns, output_column]
    return ds.map_batches(_udf, output_columns=out_cols)


def _require_size(size: tuple[int, int] | None, who: str) -> tuple[int, int]:
    if size is None:
        raise PlanError(f"{who} requires size=(height, width), e.g. size=(224, 224)")
    height, width = size
    if height <= 0 or width <= 0:
        raise PlanError(f"{who} size must be positive, got {size}")
    return height, width


def image_tensor_dataset(
    ds: Dataset,
    *,
    size: tuple[int, int] | None,
    source_column: str = "bytes",
    output_column: str = "image",
) -> Dataset:
    """Decode an image-bytes column into a ``(H, W, 3)`` uint8 tensor column.

    The decode/resize runs natively (``col(source).image.to_tensor``); the flat
    ``FixedSizeList`` result is then re-typed to a fixed-shape-tensor column (zero
    copy) so it converts straight to an ``(N, H, W, 3)`` training tensor.
    """
    from batcher.io.formats.ml.tensor import as_tensor_column
    from batcher.plan.expr_ir import col

    height, width = _require_size(size, "read.images(decode=True)")
    shape = (height, width, 3)
    decoded = ds.with_columns(**{output_column: col(source_column).image.to_tensor(width, height)})
    out_cols = list(decoded.columns)

    def _retype(batch: Any) -> Any:
        idx = batch.schema.get_field_index(output_column)
        return batch.set_column(idx, output_column, as_tensor_column(batch.column(idx), shape))

    return decoded.map_batches(_retype, output_columns=out_cols)


def audio_dataset(
    ds: Dataset,
    *,
    source_column: str = "bytes",
    output_column: str = "waveform",
    sample_rate: int | None = None,
    mono: bool = True,
) -> Dataset:
    """Decode an audio-bytes column into a ``list<float32>`` waveform column.

    Uses `soundfile` (``batcher-engine[audio]``). Waveforms are variable length, so the
    output is a list column (one waveform per row); `sample_rate` (when given)
    resamples via `librosa` if available, else the native rate is kept.
    """

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


def video_dataset(
    ds: Dataset,
    *,
    size: tuple[int, int] | None,
    num_frames: int = 8,
    source_column: str = "bytes",
    output_column: str = "frames",
) -> Dataset:
    """Decode a video-bytes column into a ``(num_frames, H, W, 3)`` uint8 tensor column.

    Samples `num_frames` evenly-spaced frames and resizes each to `size` via `PyAV`
    (``batcher-engine[video]``). Fixed frame count and size make the result a fixed-shape
    tensor, ready for a video model; undecodable rows become all-zero frames.
    """
    from batcher.io.formats.ml.tensor import as_tensor_column

    height, width = _require_size(size, "read.video(decode=True)")
    shape = (num_frames, height, width, 3)
    per_row = num_frames * height * width * 3

    def _decode(batch: Any) -> Any:
        import numpy as np
        import pyarrow as pa

        flat = np.zeros((batch.num_rows, per_row), dtype=np.uint8)
        for i, b in enumerate(batch.column(source_column).to_pylist()):
            frames = None if b is None else _decode_video_bytes(b, num_frames, height, width)
            if frames is not None:
                flat[i] = frames.reshape(-1)
        storage = pa.FixedSizeListArray.from_arrays(pa.array(flat.reshape(-1)), per_row)
        return batch.append_column(output_column, as_tensor_column(storage, shape))

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


def _decode_video_bytes(data: bytes, num_frames: int, height: int, width: int) -> Any:
    import io

    import numpy as np
    from PIL import Image

    try:
        import av
    except ImportError as exc:  # pragma: no cover - optional extra
        raise PlanError("video decode needs PyAV: pip install 'batcher-engine[video]'") from exc
    with av.open(io.BytesIO(data)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    if not frames:
        return None
    idx = np.linspace(0, len(frames) - 1, num=num_frames).astype(int)
    out = np.empty((num_frames, height, width, 3), dtype=np.uint8)
    for j, k in enumerate(idx):
        out[j] = np.asarray(Image.fromarray(frames[k]).resize((width, height)))
    return out
