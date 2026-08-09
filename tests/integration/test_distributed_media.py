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


_OPS = {
    "to_tensor": lambda c: c.image.to_tensor(6, 6),
    "letterbox": lambda c: c.image.letterbox(6, 6),
    "thumbnail": lambda c: c.image.thumbnail(6),
    "auto_orient": lambda c: c.image.auto_orient(),
    "exif_orientation": lambda c: c.image.exif_orientation(),
    "dhash": lambda c: c.image.dhash(),
    "brightness": lambda c: c.image.brightness(),
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
