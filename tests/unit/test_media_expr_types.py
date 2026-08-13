"""Every multimodal expression resolves to the type the engine actually emits.

`plan/types/media.py` states the output type of each `.image`/`.audio`/`.video` op. Two
things can go wrong with a table like that, and they fail in opposite directions:

* **A missing entry** reads as harmless, because `None` means "fall back". It is not.
  A projection resolves its `available_schema` all-or-nothing, so one untyped column
  discards the resolved type of *every column beside it*. Before this file existed, every
  `.audio` op was untyped, and
  ``select(tens=img.image.to_tensor(224, 224), wav=clip.audio.to_waveform())`` reported
  **both** columns as `null` -- the audio gap silently deleting the image sizing that
  `test_decoded_tensor_width.py` exists to provide.
* **A wrong entry** is worse than a missing one. It is the device tier's characteristic
  defect (`.claude/rules/device-tier.md`): correct values under a column type nothing
  agrees with. So nothing here is asserted from a name -- each type is held against what
  the engine emits for a real decode.

`test_every_media_op_is_classified` is the self-maintaining half: a new op added to a
family vocabulary is either typed or listed here with a reason, so the table cannot fall
behind the surface by accident.
"""

from __future__ import annotations

import math
import struct

import pyarrow as pa
import pytest

pytest.importorskip("batcher._native", reason="native engine not built")

import batcher as bt
from batcher import col
from batcher.plan.expr_ir.audio import AudioFunc
from batcher.plan.expr_ir.fn_names import AUDIO_FNS, IMAGE_FNS, VIDEO_FNS
from batcher.plan.expr_ir.image import _PNG_1X1, ImageFunc
from batcher.plan.expr_ir.video import VideoFunc
from batcher.plan.types.media import audiofunc_type, imagefunc_type, videofunc_type

pytestmark = pytest.mark.unit

_RATE = 8000


def _wav() -> bytes:
    """One second of a 440 Hz tone as 16-bit mono PCM WAV.

    Generated rather than committed: the suite needs a clip the decoder genuinely decodes,
    and a synthesized tone is both smaller than a fixture and unambiguously licensed.
    """
    pcm = b"".join(
        struct.pack("<h", int(16000 * math.sin(2 * math.pi * 440 * i / _RATE)))
        for i in range(_RATE)
    )
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, _RATE, _RATE * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm))
    )
    return header + pcm


@pytest.fixture(scope="module")
def media():
    """A one-row dataset carrying a real PNG and a real WAV."""
    return bt.from_pydict({"img": [_PNG_1X1], "clip": [_wav()]})


# Every `.image` op that needs no pixel-shape argument, and every `.audio` op, paired with
# the expression that builds it. Each is executed, so the declared type is checked against
# a genuine decode rather than against another statement of the same table.
_IMAGE_CASES = {
    "decode": lambda e: e.image.decode(),
    "to_tensor": lambda e: e.image.to_tensor(4, 4),
    "to_tensor_f32": lambda e: e.image.to_tensor_f32(4, 4),
    "to_grayscale": lambda e: e.image.to_grayscale(4, 4),
    "center_crop": lambda e: e.image.center_crop(4, 4),
    "letterbox": lambda e: e.image.letterbox(4, 4),
    "resize": lambda e: e.image.resize(4, 4),
    "encode": lambda e: e.image.encode("png"),
    "convert": lambda e: e.image.convert("RGB"),
    "thumbnail": lambda e: e.image.thumbnail(4),
    "auto_orient": lambda e: e.image.auto_orient(),
    "exif_orientation": lambda e: e.image.exif_orientation(),
    "dhash": lambda e: e.image.dhash(),
    "brightness": lambda e: e.image.brightness(),
    "sharpness": lambda e: e.image.sharpness(),
    # The pixel transforms: every one re-encodes, so every one is `binary`.
    "rotate": lambda e: e.image.rotate(90),
    "flip_horizontal": lambda e: e.image.flip_horizontal(),
    "flip_vertical": lambda e: e.image.flip_vertical(),
    "pad": lambda e: e.image.pad(4, 4),
    "invert": lambda e: e.image.invert(),
    "adjust_brightness": lambda e: e.image.adjust_brightness(1.1),
    "adjust_contrast": lambda e: e.image.adjust_contrast(1.1),
    "adjust_saturation": lambda e: e.image.adjust_saturation(1.1),
    "adjust_hue": lambda e: e.image.adjust_hue(0.1),
    "blur": lambda e: e.image.blur(1.0),
    "sharpen": lambda e: e.image.sharpen(1.0),
    "posterize": lambda e: e.image.posterize(4),
    "solarize": lambda e: e.image.solarize(128),
    "equalize": lambda e: e.image.equalize(),
    "autocontrast": lambda e: e.image.autocontrast(),
    # The measures and probes.
    "phash": lambda e: e.image.phash(),
    "ahash": lambda e: e.image.ahash(),
    "entropy": lambda e: e.image.entropy(),
    "colorfulness": lambda e: e.image.colorfulness(),
    "aspect_ratio": lambda e: e.image.aspect_ratio(),
    "mean_color": lambda e: e.image.mean_color(),
    "is_grayscale": lambda e: e.image.is_grayscale(),
    "has_alpha": lambda e: e.image.has_alpha(),
    "format": lambda e: e.image.format(),
}

_AUDIO_CASES = {
    "decode": lambda e: e.audio.decode(),
    "to_waveform": lambda e: e.audio.to_waveform(),
    "resample": lambda e: e.audio.resample(4000),
    "trim_silence": lambda e: e.audio.trim_silence(),
    "peak_normalize": lambda e: e.audio.peak_normalize(),
    "zero_crossing_rate": lambda e: e.audio.zero_crossing_rate(),
    "mel_spectrogram": lambda e: e.audio.mel_spectrogram(_RATE),
    "mfcc": lambda e: e.audio.mfcc(_RATE),
    # Level measures and conditioners.
    "rms": lambda e: e.audio.rms(),
    "dbfs": lambda e: e.audio.dbfs(),
    "peak_dbfs": lambda e: e.audio.peak_dbfs(),
    "clipping_ratio": lambda e: e.audio.clipping_ratio(),
    "silence_ratio": lambda e: e.audio.silence_ratio(),
    "rms_normalize": lambda e: e.audio.rms_normalize(),
    "pre_emphasis": lambda e: e.audio.pre_emphasis(),
    "pad_or_trim": lambda e: e.audio.pad_or_trim(0.5, _RATE),
    "slice": lambda e: e.audio.slice(0.1, 0.2),
    "encode_wav": lambda e: e.audio.encode_wav(),
    # Spectral descriptors.
    "spectrogram": lambda e: e.audio.spectrogram(_RATE),
    "spectral_centroid": lambda e: e.audio.spectral_centroid(_RATE),
    "spectral_rolloff": lambda e: e.audio.spectral_rolloff(_RATE),
    "spectral_bandwidth": lambda e: e.audio.spectral_bandwidth(_RATE),
    "spectral_flatness": lambda e: e.audio.spectral_flatness(_RATE),
}


# Parametrized over the *live* vocabulary rather than over the case table, so the suite
# runs exactly the ops the engine it is running against actually has. A hardcoded list
# would fail on an engine built before an op landed and, worse, would keep passing on one
# built after -- silently not executing the new op at all.
@pytest.mark.parametrize("fn", sorted(_IMAGE_CASES.keys() & IMAGE_FNS))
def test_an_image_type_is_what_the_engine_emits(media, fn):
    expr = _IMAGE_CASES[fn](col("img"))
    produced = media.select(x=expr).collect().schema.field("x").type
    assert imagefunc_type(expr) == produced


@pytest.mark.parametrize("fn", sorted(_AUDIO_CASES.keys() & AUDIO_FNS))
def test_an_audio_type_is_what_the_engine_emits(media, fn):
    expr = _AUDIO_CASES[fn](col("clip"))
    produced = media.select(x=expr).collect().schema.field("x").type
    assert audiofunc_type(expr) == produced


def test_every_op_in_the_vocabulary_is_actually_executed():
    """A declared type nothing runs is a guess. Every op must have a case above.

    `test_every_media_op_is_classified` proves each op *has* a type; this proves that type
    was checked against a real decode rather than written down. Without it, adding an op
    and typing it from its name passes both the classification test and the per-op tests,
    because the per-op tests only run the cases the table happens to list.
    """
    missing_image = sorted(IMAGE_FNS - _IMAGE_CASES.keys())
    missing_audio = sorted(AUDIO_FNS - _AUDIO_CASES.keys())
    assert not missing_image and not missing_audio, (
        "these ops are typed but never executed, so their declared type is unverified: "
        f"image={missing_image} audio={missing_audio}. Add a case to the table above."
    )


def test_an_audio_column_no_longer_blinds_the_image_column_beside_it(media):
    """The amplification this file's audio half exists to remove.

    Both columns came back `null` when only the audio one was untyped, so the image
    sizing was discarded by its neighbour rather than by anything wrong with it.
    """
    both = media.select(
        tens=col("img").image.to_tensor(224, 224),
        wav=col("clip").audio.to_waveform(),
    )
    schema = both._plan.available_schema()
    assert schema is not None
    assert schema.field("tens").type == imagefunc_type(col("img").image.to_tensor(224, 224))
    assert schema.field("wav").type == pa.list_(pa.float32())


def test_a_waveform_is_typed_but_not_sized():
    """The honest limit of the audio half, stated so it is not mistaken for a sizing fix.

    A decoded image's shape is in its arguments, so its width is exact. A waveform's
    length follows the clip's duration, which no argument fixes, so `list<float32>` gets
    the generic list prior -- far under a real clip. Audio memory stays a measurement.
    """
    from batcher.plan.types.widths import column_bytes

    assert audiofunc_type(AudioFunc("to_waveform", col("c"))) == pa.list_(pa.float32())
    one_second_at_16k = 16_000 * 4
    assert column_bytes(pa.list_(pa.float32())) < one_second_at_16k


# Ops whose type genuinely is not knowable, each with the reason. Anything else in a
# family vocabulary must resolve.
_UNTYPED: dict[str, str] = {
    # `.video` ops need the engine's `video` cargo feature to execute, but their types are
    # as statically known as the image ones and are asserted below without decoding.
}


def test_every_media_op_is_classified():
    """A new op is typed, or listed in `_UNTYPED` with a reason -- never silently `None`.

    Without this, adding an op to a family vocabulary and forgetting `plan/types/media.py`
    costs the static schema of every column beside it, and nothing turns red.
    """
    unresolved: list[str] = []
    for fn in sorted(IMAGE_FNS):
        build = _IMAGE_CASES.get(fn)
        expr = build(col("img")) if build else ImageFunc(fn, col("img"), width=4, height=4)
        if imagefunc_type(expr) is None and fn not in _UNTYPED:
            unresolved.append(f"image.{fn}")
    for fn in sorted(AUDIO_FNS):
        if audiofunc_type(AudioFunc(fn, col("clip"), rate=_RATE)) is None and fn not in _UNTYPED:
            unresolved.append(f"audio.{fn}")
    for fn in sorted(VIDEO_FNS):
        expr = VideoFunc(fn, col("clip"), num_frames=2, width=4, height=4)
        if videofunc_type(expr) is None and fn not in _UNTYPED:
            unresolved.append(f"video.{fn}")

    assert not unresolved, (
        "these multimodal ops have no static type, so every column sharing their "
        f"projection loses its own: {unresolved}. Type them in plan/types/media.py, or "
        "list them in _UNTYPED with the reason they cannot be typed."
    )


def test_the_video_types_are_stated_even_without_the_video_feature():
    """`.video` decode needs a cargo feature to *run*; its types do not need it to be known.

    Asserted separately from the image/audio cases, which are held against a real decode,
    so it is visible that these two are the ones no execution backs.
    """
    assert videofunc_type(VideoFunc("thumbnail", col("c"), width=8, height=8)) == pa.binary()
    assert videofunc_type(VideoFunc("frame_at", col("c"), width=8, height=8)) == pa.binary()
    assert videofunc_type(
        VideoFunc("frames", col("c"), num_frames=2, width=4, height=4)
    ) == pa.fixed_shape_tensor(pa.uint8(), (2, 4, 4, 3))
    assert pa.types.is_struct(videofunc_type(VideoFunc("decode", col("c"))))
