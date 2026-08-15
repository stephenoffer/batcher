"""Native video frame sampling — `.video.frames` / `.video.thumbnail` / `.video.frame_at`.

These are the kernels that keep a video pipeline in the data plane, so what they must be
held to is *which frame* they returned, not merely that they returned one. Every fixture
here encodes the frame index into the pixels (frame ``i`` is a solid gray of ``i * 8``),
which is what lets an assertion say "this is frame 15" rather than "this is a frame".

That distinction is not academic. A seek-based sampler that stops at the keyframe it
landed on returns frame zero for *every* timestamp in a long-GOP clip, and a test that
only checked the output was a decodable image of the right size would pass while the
whole column was one repeated title frame.

The default engine build does not enable the `video` cargo feature, so each test skips
cleanly when the accessor reports the feature is off (CI passes either way).
"""

from __future__ import annotations

import io

import pytest

import batcher as bt
from batcher import col
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.integration

#: Gray level of frame `i`. Small enough that 30 frames stay inside a byte, and coarse
#: enough that yuv420p's round trip cannot move a frame onto its neighbour's value.
_STEP = 8


def _av():
    return pytest.importorskip("av", reason="PyAV needed to build a video fixture")


def _clip(n_frames: int = 30, width: int = 64, height: int = 48, fps: int = 10) -> bytes:
    """An H.264 clip whose frame `i` is a solid gray of `i * _STEP`.

    One keyframe, at frame zero (`g` is the GOP size). That is deliberate: it is the
    layout under which a sampler that trusts its seek returns frame zero forever, so it
    is the layout the seek-and-decode-forward path has to be proved against.
    """
    av = _av()
    import numpy as np

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"crf": "0", "g": str(n_frames), "preset": "veryfast"}
        for i in range(n_frames):
            plane = np.full((height, width, 3), i * _STEP, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(plane, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buf.getvalue()


def _sample(ds, **exprs):
    """Collect `exprs`, or skip when the engine was built without the `video` feature."""
    try:
        return ds.select(**exprs).to_pydict()
    except RuntimeError as exc:  # pragma: no cover - depends on the engine build
        message = str(exc).lower()
        if "video" in message and "feature" in message:
            pytest.skip("engine built without the `video` cargo feature")
        raise


def _frame_index(png: bytes) -> int:
    """Recover which source frame a PNG is, from the gray level it was drawn with."""
    import numpy as np
    from PIL import Image

    mean = float(np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).mean())
    return round(mean / _STEP)


def test_frames_samples_the_indices_linspace_names():
    """The frames must be the ones the reference preprocessing would pick.

    A sampler that returns *some* eight frames trains a different model from one that
    returns the eight `numpy.linspace` names, and nothing downstream can tell.
    """
    import numpy as np

    ds = bt.from_pydict({"v": [_clip(n_frames=30)]})
    out = _sample(ds, f=col("v").video.frames(4, 8, 6))["f"]
    arr = np.asarray(out[0], dtype=np.uint8).reshape(4, 6, 8, 3)
    # The indices numpy.linspace(0, 29, 4).astype(int) names are 0, 9, 19 and 29.
    got = [round(float(arr[j].mean()) / _STEP) for j in range(4)]
    assert got == [0, 9, 19, 29]


def test_frames_is_a_fixed_shape_tensor_with_nulls_for_bad_rows():
    """Shape is fixed across rows, and an undecodable clip is null rather than black.

    A row of zeros would be indistinguishable from a legitimately black clip, which is
    how blank samples get into a training set unnoticed.
    """
    ds = bt.from_pydict({"v": [_clip(), None, b"not a video"]})
    out = _sample(ds, f=col("v").video.frames(3, 8, 6))["f"]
    assert len(out[0]) == 3 * 6 * 8 * 3
    assert out[1] is None
    assert out[2] is None


def test_frames_repeats_when_the_clip_is_shorter_than_the_sample():
    """A ragged row would not be a fixed-shape tensor, so short clips repeat frames."""
    import numpy as np

    ds = bt.from_pydict({"v": [_clip(n_frames=3)]})
    out = _sample(ds, f=col("v").video.frames(6, 8, 6))["f"]
    arr = np.asarray(out[0], dtype=np.uint8).reshape(6, 6, 8, 3)
    indices = [round(float(arr[j].mean()) / _STEP) for j in range(6)]
    assert len(indices) == 6
    assert indices[0] == 0 and indices[-1] == 2
    assert indices == sorted(indices)


def test_thumbnail_is_the_middle_frame_not_the_first():
    """The regression test for the seek that stopped at its keyframe.

    The fixture has exactly one keyframe, at frame zero. A sampler that returns the first
    frame it decodes after seeking returns frame 0 here — a plausible-looking PNG of the
    right size, and the wrong frame for every clip in the corpus.
    """
    ds = bt.from_pydict({"v": [_clip(n_frames=30)]})
    out = _sample(ds, t=col("v").video.thumbnail(16))["t"]
    assert _frame_index(out[0]) == pytest.approx(15, abs=1)


def test_frame_at_returns_the_frame_shown_at_that_second():
    """`frame_at(t)` is the frame a player displays at `t`, not the preceding keyframe."""
    ds = bt.from_pydict({"v": [_clip(n_frames=30, fps=10)]})
    out = _sample(
        ds,
        a=col("v").video.frame_at(0.0, 16),
        b=col("v").video.frame_at(1.0, 16),
        c=col("v").video.frame_at(2.5, 16),
    )
    assert _frame_index(out["a"][0]) == pytest.approx(0, abs=1)
    assert _frame_index(out["b"][0]) == pytest.approx(10, abs=1)
    assert _frame_index(out["c"][0]) == pytest.approx(25, abs=1)


def test_frame_at_past_the_end_is_null_not_the_last_frame():
    """Returning the last frame under a timestamp that does not exist invents data."""
    ds = bt.from_pydict({"v": [_clip(n_frames=30, fps=10)]})  # 3.0 seconds long
    out = _sample(ds, t=col("v").video.frame_at(99.0, 16))["t"]
    assert out[0] is None


def test_a_thumbnail_keeps_the_clip_aspect_ratio():
    """A still is looked at, so squashing a 4:3 clip onto a square is a real distortion.

    It is also the rule that makes `.image.thumbnail` and `.video.thumbnail` one operation
    rather than two methods sharing a name: encoded stills take a longest side, tensors
    take exact dimensions.
    """
    ds = bt.from_pydict({"v": [_clip(width=64, height=48)]})  # 4:3
    out = _sample(ds, d=col("v").video.thumbnail(20).image.decode())["d"]
    assert (out[0]["width"], out[0]["height"]) == (20, 15)


def test_a_thumbnail_never_upscales():
    """Enlarging a small clip's frame invents detail, as it does for an image."""
    ds = bt.from_pydict({"v": [_clip(width=64, height=48)]})
    out = _sample(ds, d=col("v").video.thumbnail(500).image.decode())["d"]
    assert (out[0]["width"], out[0]["height"]) == (64, 48)


def test_frames_survives_streaming_and_multiple_batches():
    """The kernel is per-row, so batching must not change which frames come back."""
    import numpy as np

    clip = _clip(n_frames=30)
    ds = bt.from_pydict({"v": [clip] * 4})
    try:
        whole = ds.select(f=col("v").video.frames(2, 8, 6)).to_pydict()["f"]
    except RuntimeError as exc:  # pragma: no cover - depends on the engine build
        if "video" in str(exc).lower() and "feature" in str(exc).lower():
            pytest.skip("engine built without the `video` cargo feature")
        raise
    streamed = [
        row
        for batch in ds.select(f=col("v").video.frames(2, 8, 6)).iter_batches(batch_size=2)
        for row in batch.column("f").to_pylist()
    ]
    assert len(streamed) == 4
    for a, b in zip(whole, streamed, strict=True):
        assert np.array_equal(np.asarray(a, dtype=np.uint8), np.asarray(b, dtype=np.uint8))


def test_frames_crosses_the_ffi_already_shaped():
    """The 4-D shape must travel with the data, not be reattached in Python.

    A sampled clip is the widest column the engine produces, so a per-batch re-type pass
    would put the whole decode on the opaque-UDF path — and the planner would size the
    column at the varlen fallback prior rather than its true megabyte-per-row.
    """
    import pyarrow as pa

    ds = bt.from_pydict({"v": [_clip()]})
    try:
        table = ds.select(f=col("v").video.frames(4, 8, 6)).collect()
    except RuntimeError as exc:  # pragma: no cover - depends on the engine build
        if "video" in str(exc).lower() and "feature" in str(exc).lower():
            pytest.skip("engine built without the `video` cargo feature")
        raise
    field = table.schema.field("f")
    assert isinstance(field.type, pa.FixedShapeTensorType)
    assert tuple(field.type.shape) == (4, 6, 8, 3)


def test_the_planner_knows_how_wide_a_sampled_clip_is():
    """Sizing a 1.2 MB row at the 64-byte varlen prior under-provisions by four orders."""
    import pyarrow as pa

    from batcher.plan.types.media import videofunc_type

    frames = col("v").video.frames(8, 224, 224)
    assert videofunc_type(frames) == pa.fixed_shape_tensor(pa.uint8(), (8, 224, 224, 3))
    assert videofunc_type(col("v").video.thumbnail(8)) == pa.binary()
    assert videofunc_type(col("v").video.frame_at(1.0, 8)) == pa.binary()
    # `decode` is the clip's header struct. It was left untyped on the grounds that a
    # header is a handful of bytes either way -- but `available_schema` resolves a
    # projection all-or-nothing, so one untyped column discards the resolved type of every
    # column beside it, including the 1.2 MB tensor this test is about.
    assert pa.types.is_struct(videofunc_type(col("v").video.decode()))


def test_video_dataset_uses_the_native_kernel_when_the_engine_has_it():
    """`read.video(decode=True)` must not fall into a Python loop on a capable engine."""
    import pyarrow as pa

    from batcher._internal.native import engine_features
    from batcher.ml.decode import video_dataset

    if "video" not in engine_features():
        pytest.skip("engine built without the `video` cargo feature")
    ds = bt.from_pydict({"bytes": [_clip(), None, b"not a video"]})
    out = video_dataset(ds, size=(6, 8), num_frames=4, error_column="failed").collect()
    assert isinstance(out.schema.field("frames").type, pa.FixedShapeTensorType)
    assert tuple(out.schema.field("frames").type.shape) == (4, 6, 8, 3)
    frames = out.column("frames").to_pylist()
    assert frames[0] is not None
    assert frames[1] is None and frames[2] is None
    # Present-but-undecodable is a failure; a null input is not.
    assert out.column("failed").to_pylist() == [False, False, True]


@pytest.mark.parametrize(
    "build",
    [
        lambda c: c.video.frames(0, 8, 8),
        lambda c: c.video.frames(-1, 8, 8),
        lambda c: c.video.frames(2, 0, 8),
        lambda c: c.video.frames(2, 8, -3),
        lambda c: c.video.thumbnail(0),
        lambda c: c.video.thumbnail(-4),
        lambda c: c.video.frame_at(-1.0, 8),
        lambda c: c.video.frame_at(1.0, 0),
    ],
)
def test_bad_arguments_are_rejected_at_plan_build(build):
    """A bad frame count or size is a caller bug, so it fails before any decode runs."""
    with pytest.raises(PlanError):
        build(col("v"))


def test_frame_at_takes_its_timestamp_from_a_column():
    """The usual case, not the exotic one: the row that wants a still knows when.

    A detection, a caption, a scene boundary — each already carries the moment it refers
    to, so a constant timestamp made the common case the one this could not express.
    """
    ds = bt.from_pydict({"v": [_clip(n_frames=30, fps=10)] * 4, "t": [0.0, 1.0, 2.0, 2.9]})
    out = _sample(ds, s=col("v").video.frame_at(col("t"), 16))["s"]

    got = [_frame_index(p) for p in out]
    assert got == [pytest.approx(w, abs=1) for w in (0, 10, 20, 29)]


def test_a_constant_timestamp_still_works():
    """The literal form is the same operation with a constant column, not a second one."""
    ds = bt.from_pydict({"v": [_clip(n_frames=30, fps=10)] * 3})
    out = _sample(ds, s=col("v").video.frame_at(1.0, 16))["s"]

    assert [_frame_index(p) for p in out] == [pytest.approx(10, abs=1)] * 3


def test_an_unusable_timestamp_nulls_only_its_own_row():
    """A moment the caller could not supply is a row with no answer, not a failed batch.

    Same call `crop` makes about a bad bounding box, for the same reason: the timestamps
    come from something that does not answer for every row, and failing the batch would
    lose the rows that did.
    """
    clip = _clip(n_frames=30, fps=10)  # 3.0 seconds long
    ds = bt.from_pydict({"v": [clip] * 4, "t": [0.5, None, -1.0, 99.0]})
    out = _sample(ds, s=col("v").video.frame_at(col("t"), 16))["s"]

    assert out[0] is not None, "a usable timestamp must still produce a still"
    assert out[1] is None, "null"
    assert out[2] is None, "negative"
    assert out[3] is None, "past the end of a clip whose duration is known"
