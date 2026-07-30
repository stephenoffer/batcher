"""A decoded image column's width is in the expression's arguments, not in the data.

Every `.image.*` expression inferred as `None`, which reads as harmless because `None` means
"fall back". But the fallback for a *width* is a flat 64-byte prior, and these are the widest
columns the engine holds. Measured on a real decode pipeline before the fix:

    select("id", "img") after .image.to_tensor(224, 224)   ->      16 B/row  (true 150,536)
    select("img")                                          ->      64 B/row  (true 150,528)

That is the ordinary shape of every image workload -- decode, then drop the compressed bytes
-- mis-sized by four orders of magnitude in the direction that under-provisions the memory
envelope and makes a build side look broadcastable when replicating it would OOM a worker.

It is the same blind spot as an extension type hiding its storage type, one step down the
plan: there the *source* column's type was unreadable, here the *derived* column's is, and a
decode pipeline lives entirely in derived columns.
"""

from __future__ import annotations

import io

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.api.source_stats import collect_source_stats, column_bounds_needed
from batcher.carbonite.memory.pressure import PressureLevel
from batcher.carbonite.policies.morsel import morsel_target
from batcher.config import active_config
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.cost import CostModel
from batcher.plan.types.media import imagefunc_type

pytestmark = pytest.mark.unit

_ROWS = 64
_W = _H = 32


@pytest.fixture(scope="module")
def jpeg() -> bytes:
    """A real encoded image, so the decode under test is a genuine decode."""
    Image = pytest.importorskip("PIL.Image")
    pixels = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype="uint8")
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def frame(jpeg):
    return bt.from_arrow(
        pa.table(
            {
                "id": pa.array(np.arange(_ROWS)),
                "b": pa.array([jpeg] * _ROWS, type=pa.binary()),
            }
        )
    )


def _row_bytes(ds, plan) -> float:
    stats = collect_source_stats(ds._sources, None, need_columns=column_bounds_needed(plan))
    return CostModel(CardinalityEstimator(ds._sources, source_stats=stats)).row_bytes(plan)


def _measured(dataset) -> float:
    table = dataset.collect()
    return table.nbytes / table.num_rows


@pytest.mark.parametrize(
    ("label", "build"),
    [
        ("to_tensor", lambda e: e.image.to_tensor(_W, _H)),
        ("to_tensor_f32", lambda e: e.image.to_tensor_f32(_W, _H)),
        ("to_tensor_f32 CHW", lambda e: e.image.to_tensor_f32(_W, _H, channels_first=True)),
        ("to_grayscale", lambda e: e.image.to_grayscale(_W, _H)),
        ("center_crop", lambda e: e.image.center_crop(_W, _H)),
        ("dhash", lambda e: e.image.dhash()),
    ],
)
def test_a_derived_media_column_is_priced_at_what_it_actually_is(frame, label, build):
    """The estimate must match the bytes the engine really holds, not a prior."""
    derived = frame.with_columns(x=build(col("b"))).select("x")
    assert _row_bytes(frame, derived._plan) == pytest.approx(_measured(derived), rel=0.01), label


def test_the_pipeline_shape_this_exists_for(frame):
    """Decode, then drop the compressed bytes -- the shape that cost 16 B/row."""
    pipeline = frame.with_columns(img=col("b").image.to_tensor(224, 224)).select("id", "img")
    estimated = _row_bytes(frame, pipeline._plan)
    assert estimated == pytest.approx(_measured(pipeline), rel=0.01)
    # And it is emphatically not the old answer.
    assert estimated > 100_000


def test_the_inferred_type_matches_what_the_engine_emits(frame):
    """Nothing here is inferred from a function name -- the engine decides."""
    for build in (
        lambda e: e.image.to_tensor(_W, _H),
        lambda e: e.image.to_tensor_f32(_W, _H),
        lambda e: e.image.to_tensor_f32(_W, _H, channels_first=True),
        lambda e: e.image.to_grayscale(_W, _H),
        lambda e: e.image.dhash(),
    ):
        expr = build(col("b"))
        produced = frame.select(x=expr).collect().schema.field("x").type
        assert imagefunc_type(expr) == produced


def test_a_still_encoded_result_is_binary_not_a_tensor(frame):
    """`resize`/`crop`/`encode` hand back an encoded image, whose size is the image's
    content and so is genuinely not knowable from the arguments."""
    for build in (
        lambda e: e.image.resize(_W, _H),
        lambda e: e.image.crop(0, 0, _W, _H),
        lambda e: e.image.encode("png"),
    ):
        assert imagefunc_type(build(col("b"))) == pa.binary()


def test_an_unrecognised_image_op_stays_unknown():
    """`None` must remain the answer for anything not verified, since a wrong type is
    worse than no type."""
    from batcher.plan.expr_ir.image import ImageFunc

    assert imagefunc_type(ImageFunc("decode", col("b"))) is None


def test_the_width_reaches_the_morsel_cap(frame):
    """A width nothing consumes is not a fix. A 16,384-row morsel of decoded 224x224x3
    images is 2.5 GB against a 1 MiB budget."""
    config, level = active_config(), PressureLevel.NORMAL
    decoded = frame.with_columns(img=col("b").image.to_tensor(224, 224)).select("img")._plan
    target = morsel_target(config, level, plan=decoded)
    assert target is not None
    rows, budget = target
    assert rows * 224 * 224 * 3 <= budget


def test_a_narrow_plan_is_unchanged(frame):
    """The safety property: nothing about a plan with no media column may move."""
    assert morsel_target(active_config(), PressureLevel.NORMAL, plan=frame.select("id")._plan) is (
        None
    )


def test_the_decoded_column_still_holds_the_right_pixels(frame):
    """A width is a sizing input, so it may never change the relation."""
    out = frame.with_columns(img=col("b").image.to_tensor(_W, _H)).select("img").collect()
    assert out.num_rows == _ROWS
    assert len(out.column("img")[0].as_py()) == _W * _H * 3
