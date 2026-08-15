"""A media decode gives the same answer on one node and on many.

The media kernels are per-row projections, so their distributed equivalence is
structural rather than earned — there is no partial state to merge and nothing a
shuffle can reorder. That is exactly why it is worth an assertion: a claim resting on
"it cannot go wrong" is a claim nobody has checked, and CI never runs this path at all
(the PR gate installs no Ray, so `just lint-skips` counts every one of these as
unreachable).

What can actually go wrong here is not the arithmetic. It is the *column*: a decoded
tensor carries its shape as Arrow extension metadata, and metadata is the first thing a
shuffle drops. A distributed run that returned the right pixels under a plain
`FixedSizeList` rather than a `fixed_shape_tensor` would feed a model a flat vector and
be very hard to trace back. So these check the type as carefully as the values.
"""

from __future__ import annotations

import io

import pyarrow as pa
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray
from batcher import col

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")
pytest.importorskip("PIL", reason="Pillow needed to build an image fixture")

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


def _png(width: int, height: int, shade: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (shade, 40, 200 - shade)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def frames():
    """Mixed sizes, orientations, nulls and garbage — the shape of a real corpus.

    Enough rows to span several partitions, so the comparison is between a genuinely
    partitioned run and a single-node one rather than between two one-partition runs.
    """
    rows: list[bytes | None] = []
    for i in range(600):
        if i % 37 == 0:
            rows.append(None)
        elif i % 53 == 0:
            rows.append(b"not an image")
        else:
            rows.append(_png(8 + (i % 5), 4 + (i % 7), i % 200))
    return bt.from_arrow(pa.table({"b": pa.array(rows, type=pa.binary())}))


# One representative per *output shape*, not one per operation. The kernels number in the
# dozens and they share their assembly code, so what a distributed run can break is the
# shape a column is handed back in — a fixed-shape tensor, a still-encoded blob, a struct,
# a digest, a flag — rather than any individual op's arithmetic. Covering every op would
# multiply the cluster time by six and test the same five things.
_OPS = {
    # Fixed-shape tensors: the shape lives in Arrow extension metadata, which a shuffle
    # drops before it drops anything else.
    "to_tensor": lambda c: c.image.to_tensor(6, 6),
    "letterbox": lambda c: c.image.letterbox(6, 6),
    # Still-encoded blobs, including one that names a non-default container: the format is
    # resolved once per batch, so a worker resolving it differently is a real hazard.
    "thumbnail": lambda c: c.image.thumbnail(6),
    "auto_orient": lambda c: c.image.auto_orient(),
    "rotate_jpeg": lambda c: c.image.rotate(90, format="jpeg", quality=70),
    "adjust_brightness": lambda c: c.image.adjust_brightness(1.4),
    "equalize": lambda c: c.image.equalize(),
    # Digests. These are the ones that must be *bit*-identical across workers, because a
    # hash computed two ways is a dedup join that matches fewer rows on a cluster.
    "dhash": lambda c: c.image.dhash(),
    "phash": lambda c: c.image.phash(),
    "ahash": lambda c: c.image.ahash(),
    # Scalar measures, a struct, a flag, and a header fact — the four remaining shapes.
    "brightness": lambda c: c.image.brightness(),
    "entropy": lambda c: c.image.entropy(),
    "colorfulness": lambda c: c.image.colorfulness(),
    "mean_color": lambda c: c.image.mean_color(),
    "is_grayscale": lambda c: c.image.is_grayscale(),
    "exif_orientation": lambda c: c.image.exif_orientation(),
    "aspect_ratio": lambda c: c.image.aspect_ratio(),
    "format": lambda c: c.image.format(),
}


@pytest.mark.parametrize("label", sorted(_OPS))
def test_distributed_media_decode_equals_single_node(frames, label):
    """Same rows, same order-independent multiset, same nulls."""
    build = _OPS[label]
    local = frames.select(x=build(col("b"))).collect()
    distributed = frames.select(x=build(col("b"))).collect(distributed=True, num_workers=4)

    assert distributed.num_rows == local.num_rows
    assert distributed.schema.field("x").type == local.schema.field("x").type
    assert sorted(map(repr, distributed.column("x").to_pylist())) == sorted(
        map(repr, local.column("x").to_pylist())
    )


def test_a_tensor_column_keeps_its_shape_through_the_shuffle(frames):
    """Extension metadata is the first thing a shuffle drops, and losing it is silent.

    The values would still be right; the column would just stop being a
    `(6, 6, 3)` tensor and start being a flat 108-element list, which a model consumes
    without complaint and gets wrong.
    """
    out = frames.select(x=col("b").image.letterbox(6, 6)).collect(distributed=True, num_workers=4)
    field = out.schema.field("x")

    assert isinstance(field.type, pa.FixedShapeTensorType), field.type
    assert tuple(field.type.shape) == (6, 6, 3)


def test_undecodable_rows_stay_null_on_every_worker(frames):
    """A worker that failed the batch instead of nulling would lose its whole partition.

    Counting is the check that survives partitioning: which rows are null is a property
    of the data, so the count must not depend on how the data was split.
    """
    predicate = col("b").is_not_null() & col("b").image.decode().is_null()
    bad = frames.filter(predicate).select("b")
    local = bad.collect().num_rows
    distributed = bad.collect(distributed=True, num_workers=4).num_rows

    assert local > 0, "the fixture must contain undecodable bytes for this to mean anything"
    assert distributed == local


def test_an_aggregate_over_a_decoded_column_merges_correctly(frames):
    """The one media shape whose distributed equivalence is *not* structural.

    Everything above is a per-row projection: no partial state, nothing a shuffle can
    reorder, which is what the module docstring means by "structural". An aggregate over
    a decoded column is the opposite. The decode runs per partition, its output is
    shuffled by the group key, and the partials are then merged -- so this is the first
    case where the mergeable algebra actually has to hold over a column the media kernels
    produced, and the first where a wrong answer would be a real merge bug rather than a
    dropped extension type.

    Grouped by a decoded *header* fact rather than an arbitrary key, because that is the
    shape a corpus audit is written in: "how bright is the average image at each width".
    """
    query = frames.filter(col("b").is_not_null()).with_columns(
        w=col("b").image.decode().struct.field("width"),
        bright=col("b").image.brightness(),
    )
    grouped = query.group_by("w").agg(
        n=col("bright").count(),
        lo=col("bright").min(),
        hi=col("bright").max(),
    )

    local = grouped.collect().sort_by("w").to_pydict()
    distributed = grouped.collect(distributed=True, num_workers=4).sort_by("w").to_pydict()

    assert local["w"], "the fixture must produce several widths for this to mean anything"
    assert distributed["w"] == local["w"]
    assert distributed["n"] == local["n"]
    # `min`/`max` are exact under any merge order, so these compare exactly rather than
    # approximately -- unlike a `sum`/`mean`, which float reassociation may move.
    assert distributed["lo"] == local["lo"]
    assert distributed["hi"] == local["hi"]


def test_a_decoded_column_survives_a_join_across_workers(frames):
    """A media-derived column used as a join key, which the shuffle repartitions by.

    `dhash` is the join key a near-duplicate pass is built on, so this is the shape that
    matters rather than an arbitrary one: if the hash were computed differently on two
    workers -- a different downsample, a different rounding -- the join would silently
    match fewer rows on a cluster than on one node, and nothing else here would see it.
    """
    hashed = frames.filter(col("b").is_not_null()).select(h=col("b").image.dhash())
    other = hashed.select(h2=col("h"))
    joined = hashed.join(other, left_on="h", right_on="h2")

    local = joined.collect().num_rows
    distributed = joined.collect(distributed=True, num_workers=4).num_rows

    assert local > 0
    assert distributed == local


# ---- audio ------------------------------------------------------------------------
#
# The `.audio` kernels had no distributed coverage at all, which is a larger gap than it
# looks: their column shapes are the ones a shuffle handles *worst*. A waveform is a
# variable-length list, so unlike an image tensor its rows genuinely differ in size, and a
# spectral descriptor resolves its framing once per batch — a per-batch parameter is
# exactly what differs between one partition and eight.


def _wav(seconds: float, freq: float, rate: int = 16000) -> bytes:
    """A mono 16-bit PCM WAV of a sine, built here so the suite needs no fixture file."""
    import math
    import struct

    n = int(seconds * rate)
    pcm = b"".join(
        struct.pack("<h", round(0.4 * math.sin(2 * math.pi * freq * i / rate) * 32767))
        for i in range(n)
    )
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )


@pytest.fixture(scope="module")
def clips():
    """Clips of unequal length, plus nulls and garbage — a real corpus's shape.

    Unequal length matters here in a way it does not for images: a waveform column's rows
    are genuinely different sizes, so the partitioning cannot be assumed uniform.
    """
    rows: list[bytes | None] = []
    for i in range(240):
        if i % 31 == 0:
            rows.append(None)
        elif i % 47 == 0:
            rows.append(b"not audio")
        else:
            rows.append(_wav(0.02 + (i % 5) * 0.01, 200 + (i % 7) * 60))
    return bt.from_arrow(pa.table({"b": pa.array(rows, type=pa.binary())}))


_AUDIO_OPS = {
    # Scalar level measures: pure functions of the samples, no framing.
    "rms": lambda c: c.audio.rms(),
    "dbfs": lambda c: c.audio.dbfs(),
    "silence_ratio": lambda c: c.audio.silence_ratio(),
    # Variable-length waveforms, including the one whose whole purpose is to make every
    # row the *same* length — which is only meaningful if it holds per partition too.
    "trim_silence": lambda c: c.audio.trim_silence(),
    "pad_or_trim": lambda c: c.audio.pad_or_trim(0.05, 16000),
    "pre_emphasis": lambda c: c.audio.pre_emphasis(),
    # A per-batch-resolved framing, and a still-encoded container.
    "spectral_centroid": lambda c: c.audio.spectral_centroid(16000, n_fft=64, hop_length=32),
    "spectral_rolloff": lambda c: c.audio.spectral_rolloff(16000, n_fft=64, hop_length=32),
    "encode_wav": lambda c: c.audio.encode_wav(8000),
}


@pytest.mark.parametrize("label", sorted(_AUDIO_OPS))
def test_distributed_audio_equals_single_node(clips, label):
    """Same rows, same multiset, same nulls, same column type."""
    build = _AUDIO_OPS[label]
    local = clips.select(x=build(col("b"))).collect()
    distributed = clips.select(x=build(col("b"))).collect(distributed=True, num_workers=4)

    assert distributed.num_rows == local.num_rows
    assert distributed.schema.field("x").type == local.schema.field("x").type
    assert sorted(map(repr, distributed.column("x").to_pylist())) == sorted(
        map(repr, local.column("x").to_pylist())
    )


def test_pad_or_trim_gives_one_width_on_every_worker(clips):
    """The op exists to make a clip corpus batchable, which is a per-partition promise.

    A worker that resolved the target length from its own partition rather than from the
    query would produce rows of two widths, and the concatenated result would be
    unbatchable in a way no row-count check can see.
    """
    widths = (
        clips.filter(col("b").is_not_null())
        .select(n=col("b").audio.pad_or_trim(0.05, 16000).list.len())
        .collect(distributed=True, num_workers=4)
        .column("n")
        .to_pylist()
    )
    assert {w for w in widths if w is not None} == {800}


def test_a_cleaned_clip_round_trips_the_same_way_on_a_cluster(clips):
    """The chain a corpus-cleaning job actually runs, end to end across workers.

    Three kernels feeding each other — a waveform into a level match into an encoder —
    where the middle two read a *decoded* column rather than bytes. That is the seam the
    single-node tests cover and the cluster has never exercised.
    """
    cleaned = col("b").audio.trim_silence().audio.rms_normalize().audio.encode_wav(16000)
    query = clips.filter(col("b").is_not_null()).select(
        rate=cleaned.audio.decode().struct.field("sample_rate")
    )
    local = sorted(map(repr, query.collect().column("rate").to_pylist()))
    distributed = sorted(
        map(repr, query.collect(distributed=True, num_workers=4).column("rate").to_pylist())
    )
    assert distributed == local
    assert local.count("16000") > 0
