"""Output types for the multimodal expressions, where the shape is in the arguments.

Split from `infer` on a responsibility seam that file was about to cross: the scalar
inference there answers "what type does this arithmetic produce", while these answer "how
big is a decoded frame", which is a *sizing* question that happens to be phrased as a type.

Every image expression used to infer as `None`. That reads as harmless, since `None` means
"fall back" -- but the fallback for a width is a flat 64-byte prior, and these are the widest
columns the engine ever holds. A decode pipeline lives entirely in derived columns, so the
fallback applied to all of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    from batcher.plan.expr_ir.audio import AudioFunc
    from batcher.plan.expr_ir.image import ImageFunc
    from batcher.plan.expr_ir.video import VideoFunc

__all__ = ["audiofunc_type", "imagefunc_type", "videofunc_type"]


# The header facts each `decode` reads, exactly as the engine builds them
# (`bc-expr::eval::media`). The struct itself is nullable -- an undecodable row is null --
# while its fields are not, which is the shape the kernels declare.
def _header(*fields: tuple[str, pa.DataType]) -> pa.DataType:
    return pa.struct([pa.field(name, dtype, nullable=False) for name, dtype in fields])


_IMAGE_HEADER = _header(
    ("width", pa.int32()),
    ("height", pa.int32()),
    ("channels", pa.int32()),
    ("mode", pa.string()),
)
_AUDIO_HEADER = _header(
    ("sample_rate", pa.int32()),
    ("channels", pa.int32()),
    ("num_frames", pa.int64()),
    ("duration_secs", pa.float64()),
)
_VIDEO_HEADER = _header(
    ("width", pa.int32()),
    ("height", pa.int32()),
    ("num_frames", pa.int64()),
    ("duration_secs", pa.float64()),
    ("fps", pa.float64()),
)


# The image ops whose output is a decoded pixel tensor, and how many channels each
# produces. Their shape is not a property of the data -- it is the arguments -- so the
# width of a decoded image column is knowable before a single byte is read.
_IMAGE_TENSOR_CHANNELS = {
    "to_tensor": 3,
    "letterbox": 3,
    "to_tensor_f32": 3,
    "center_crop": 3,
    "to_grayscale": 1,
}

# The image ops that hand back a still-encoded image, so the payload stays compressed.
# Everything from `rotate` down is a pixel transform that re-encodes rather than resizing,
# but the output shape is the same question and the same answer: an encoded image, whose
# byte count is its content and so is genuinely not knowable from the arguments.
_IMAGE_BINARY_FNS = frozenset(
    {
        "resize", "encode", "convert", "auto_orient", "thumbnail",
        "rotate", "flip_horizontal", "flip_vertical", "pad", "invert",
        "adjust_brightness", "adjust_contrast", "adjust_saturation", "adjust_hue",
        "blur", "sharpen", "posterize", "solarize", "equalize", "autocontrast",
    }
)  # fmt: skip

# The image measures: one `Float64` score per row (`bc-expr::eval::media::image::quality`
# and `::probe`).
_IMAGE_SCORE_FNS = frozenset({"brightness", "sharpness", "entropy", "colorfulness", "aspect_ratio"})

# The perceptual hashes: 64 bits reinterpreted as `Int64`, so a Hamming distance over them
# is `bit_count(a ^ b)` and a near-duplicate join is expressible in the engine.
_IMAGE_HASH_FNS = frozenset({"dhash", "phash", "ahash"})

# The per-row predicates a curation filter is written against.
_IMAGE_FLAG_FNS = frozenset({"is_grayscale", "has_alpha"})

# `mean_color()` reports the three channel means (`bc-expr::eval::media::image::quality`).
_IMAGE_MEAN_COLOR = _header(("r", pa.float64()), ("g", pa.float64()), ("b", pa.float64()))


def imagefunc_type(expr: ImageFunc) -> pa.DataType | None:
    """The Arrow type an `.image.*` expression produces, or `None` when not certain.

    Every image expression previously inferred as `None`, which sounds harmless -- `None`
    means "fall back" -- but the fallback for a *width* is a flat 64-byte prior, and these
    are the widest columns the engine ever holds. Measured on a real decode pipeline:
    `select("id", "img")` after `.image.to_tensor(224, 224)` was costed at **16 bytes per
    row against a true 150,536**, and `select("img")` at 64 against 150,528. That is the
    ordinary shape of every image workload -- decode, then drop the compressed bytes -- and
    it was mis-sized by four orders of magnitude, in the direction that under-provisions the
    memory envelope and makes a build side look broadcastable.

    This is the same blind spot as an extension type hiding its storage from
    `plan.types.widths`, one step further down the plan: there the *source* column's type
    was unreadable, here the *derived* column's is, and a decode pipeline lives entirely in
    derived columns.

    Nothing here is inferred from a name. Each mapping was read off the engine's actual
    output: `to_tensor(w, h)` yields `fixed_shape_tensor(uint8, [h, w, 3])`, `to_tensor_f32`
    the `float32` form (`[3, h, w]` under `channels_first`), `to_grayscale` a single luma
    channel, `center_crop` the cropped RGB region, `dhash` an `int64` digest,
    `exif_orientation` an `int32` code, `format` the container name as `string`,
    `is_grayscale`/`has_alpha` a `bool`, `mean_color` the three channel means,
    `brightness`/`sharpness`/`entropy`/`colorfulness`/`aspect_ratio` a `float64` score,
    `dhash`/`phash`/`ahash` an `int64` digest, every pixel transform and
    `resize`/`crop`/`encode`/`convert`/`auto_orient` a still-encoded `binary`, and `decode`
    the header struct the kernel declares.

    `decode` used to be left `None` on the grounds that a handful of header bytes is not
    worth typing. That reasoning measured the wrong cost: a projection's
    `available_schema` is all-or-nothing, so a single untyped column discards the resolved
    type of *every column beside it*. One `.image.decode()` was enough to throw away the
    `fixed_shape_tensor` width this function exists to compute.

    Args:
        expr: The `ImageFunc` node to type.

    Returns:
        The output Arrow type, or `None` when it is not statically known.
    """
    if expr.fn == "decode":
        return _IMAGE_HEADER
    if expr.fn == "mean_color":
        return _IMAGE_MEAN_COLOR
    if expr.fn in _IMAGE_HASH_FNS:
        return pa.int64()
    if expr.fn == "exif_orientation":
        return pa.int32()
    if expr.fn == "format":
        return pa.string()
    if expr.fn in _IMAGE_FLAG_FNS:
        return pa.bool_()
    if expr.fn in _IMAGE_SCORE_FNS:
        return pa.float64()
    if expr.fn in _IMAGE_BINARY_FNS:
        return pa.binary()
    channels = _IMAGE_TENSOR_CHANNELS.get(expr.fn)
    if channels is None or expr.width is None or expr.height is None:
        return None
    value = pa.float32() if expr.fn == "to_tensor_f32" else pa.uint8()
    shape = (
        (channels, expr.height, expr.width)
        if expr.channels_first
        else (expr.height, expr.width, channels)
    )
    return pa.fixed_shape_tensor(value, shape)


# The video ops that hand back a still-encoded still, so the payload stays compressed.
_VIDEO_BINARY_FNS = frozenset({"thumbnail", "frame_at"})


def videofunc_type(expr: VideoFunc) -> pa.DataType | None:
    """The Arrow type a `.video.*` expression produces, or `None` when not certain.

    The same sizing question :func:`imagefunc_type` answers, for the columns where getting
    it wrong costs the most. A sampled clip is the widest thing the engine ever holds: a
    modest ``frames(8, 224, 224)`` is **1,204,224 bytes per row**, eight times a decoded
    image and roughly nineteen thousand times the 64-byte fallback prior. A memory envelope
    built on that prior is not merely optimistic, it is off by four orders of magnitude in
    the direction that makes a build side look broadcastable and a morsel look cheap.

    As with the image ops, nothing here is inferred from a name: `frames` yields
    ``fixed_shape_tensor(uint8, [num_frames, height, width, 3])``,
    `thumbnail`/`frame_at` still-encoded PNG ``binary``, and `decode` the header struct
    the kernel declares (see :func:`imagefunc_type` for why a "handful of bytes" column is
    still worth typing).

    Args:
        expr: The `VideoFunc` node to type.

    Returns:
        The output Arrow type, or `None` when it is not statically known.
    """
    if expr.fn == "decode":
        return _VIDEO_HEADER
    if expr.fn in _VIDEO_BINARY_FNS:
        return pa.binary()
    if expr.fn != "frames":
        return None
    if expr.num_frames is None or expr.width is None or expr.height is None:
        return None
    return pa.fixed_shape_tensor(pa.uint8(), (expr.num_frames, expr.height, expr.width, 3))


# Every `.audio` op whose result is a signal rather than a scalar. All six emit
# `List<Float32>`: the waveform conditioners hand back mono samples, and the two
# spectral front ends hand back a flattened `(n_mels, n_frames)` / `(n_mfcc, n_frames)`
# matrix. The row *length* varies with clip duration, which no argument fixes, so the
# element type is knowable statically and the length is not.
_AUDIO_SIGNAL_FNS = frozenset(
    {
        # Waveform conditioners: mono samples in, mono samples out.
        "to_waveform", "resample", "trim_silence", "peak_normalize", "rms_normalize",
        "pre_emphasis", "pad_or_trim", "slice",
        # Spectral front ends: a flattened `(bands, frames)` matrix.
        "mel_spectrogram", "mfcc", "spectrogram",
    }
)  # fmt: skip

# The `.audio` measures: one `Float64` per row, whether a level (`bc-expr::eval::media::
# level`), a spectral descriptor (`::spectral`) or the zero-crossing rate (`::speech`).
_AUDIO_SCORE_FNS = frozenset(
    {
        "zero_crossing_rate", "rms", "dbfs", "peak_dbfs", "clipping_ratio", "silence_ratio",
        "spectral_centroid", "spectral_rolloff", "spectral_bandwidth", "spectral_flatness",
    }
)  # fmt: skip


def audiofunc_type(expr: AudioFunc) -> pa.DataType | None:
    """The Arrow type an `.audio.*` expression produces, or `None` when not certain.

    The audio half of what :func:`imagefunc_type` and :func:`videofunc_type` do for pixels,
    and the half that was missing: every `.audio` expression inferred as `None`, so an audio
    pipeline had no static schema at all.

    The cost of that was not confined to the audio column. A projection resolves its
    `available_schema` all-or-nothing, so one untyped column discards the resolved type of
    every column beside it -- and an audio column beside an image one is not an exotic
    shape, it is what a video pipeline looks like the moment it reads the sound track.
    ``select(tens=col("img").image.to_tensor(224, 224), wav=col("clip").audio.to_waveform())``
    reported **both** columns as `null`, so the tensor sizing the other two functions here
    exist to provide was discarded by the column sitting next to it.

    Types are read off the kernels (`bc-expr::eval::media::audio`, `::speech`, `::level`
    and `::spectral`), not inferred from names: `decode` yields the header struct
    ``{sample_rate, channels, num_frames, duration_secs}``, the level and spectral measures
    a `float64` score, `encode_wav` a still-encoded `binary` container, and every signal op
    a ``list<float32>``.

    Note what this deliberately does **not** claim. A waveform's row *length* follows the
    clip's duration, which no argument on the node fixes, so unlike a decoded image these
    columns are typed but not sized: `list<float32>` charges the generic list prior. A
    thirty-second clip at 16 kHz is nearer 1.9 MB than the prior's 36 bytes, so an audio
    memory envelope stays a measurement rather than a static answer.

    `pad_or_trim` is the one op whose length *is* knowable — that is its whole purpose —
    but it is typed `list<float32>` alongside the rest, because the kernel emits a
    variable-length `List` and declaring a `fixed_size_list` here would make the engine's
    own schema contract disagree with the column it describes.

    Args:
        expr: The `AudioFunc` node to type.

    Returns:
        The output Arrow type, or `None` when it is not statically known.
    """
    if expr.fn == "decode":
        return _AUDIO_HEADER
    if expr.fn == "encode_wav":
        return pa.binary()
    if expr.fn in _AUDIO_SCORE_FNS:
        return pa.float64()
    if expr.fn in _AUDIO_SIGNAL_FNS:
        return pa.list_(pa.float32())
    return None
