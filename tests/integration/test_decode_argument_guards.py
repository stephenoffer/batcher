"""The multimodal decode stages validate their arguments the way their siblings do.

`image_tensor_dataset` and `audio_dataset` build their output as a projection, so a mistyped
source column is caught by the projection's own validation. `video_dataset`,
`download_dataset`, and `upload_dataset` read the column inside a `map_batches` callback, so
the same slip reached Arrow and returned ``KeyError: 'Field "nope" does not exist in
schema'`` — naming neither the function, nor the argument, nor the columns that exist. One
mistake reported two different ways depending on which modality you were decoding.
"""

from __future__ import annotations

import io

import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.decode import (
    audio_dataset,
    download_dataset,
    image_tensor_dataset,
    upload_dataset,
    video_dataset,
)


def _png(width: int = 8, height: int = 8) -> bytes:
    Image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (255, 0, 0)).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_pydict({"id": [1, 2], "bytes": [_png(), _png()]})


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("video", lambda d: video_dataset(d, size=(4, 4), source_column="nope")),
        ("download", lambda d: download_dataset(d, url_column="nope")),
        ("upload", lambda d: upload_dataset(d, data_column="nope", directory="/tmp/bt-test")),
    ],
)
def test_an_unknown_source_column_is_typed_and_lists_alternatives(
    ds: bt.Dataset, name: str, call
) -> None:
    with pytest.raises(ColumnNotFoundError) as caught:
        call(ds)
    message = str(caught.value)
    assert "nope" in message
    assert "'bytes'" in message and "'id'" in message


def test_the_message_names_the_argument(ds: bt.Dataset) -> None:
    with pytest.raises(ColumnNotFoundError, match="url_column"):
        download_dataset(ds, url_column="nope")


@pytest.mark.parametrize("frames", [0, -1])
def test_a_non_positive_frame_count_is_rejected(ds: bt.Dataset, frames: int) -> None:
    """Zero raised from Arrow and a negative from NumPy; neither named `num_frames`."""
    with pytest.raises(PlanError, match="num_frames must be >= 1"):
        video_dataset(ds, size=(4, 4), num_frames=frames)


def test_the_image_and_audio_stages_still_report_theirs(ds: bt.Dataset) -> None:
    """Pins the convention the other three now follow, rather than assuming it."""
    for call in (
        lambda: image_tensor_dataset(ds, size=(4, 4), source_column="nope").to_pydict(),
        lambda: audio_dataset(ds, source_column="nope").to_pydict(),
    ):
        with pytest.raises(Exception, match="nope"):
            call()


def test_valid_arguments_are_untouched(ds: bt.Dataset) -> None:
    decoded = image_tensor_dataset(ds, size=(4, 4))
    assert "image" in decoded.columns
    assert decoded.count() == 2
