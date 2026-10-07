"""Live smoke test: to_huggingface against the real datasets library.

Skipped unless ``BATCHER_LIVE_HUGGINGFACE`` is set. See tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt

pytestmark = pytest.mark.skipif(
    not os.environ.get("BATCHER_LIVE_HUGGINGFACE"),
    reason="set BATCHER_LIVE_HUGGINGFACE to run against a live service",
)


def test_to_huggingface_features_and_rows():
    pytest.importorskip("datasets")
    ds = bt.from_pydict(
        {
            "text": ["a", "b"],
            "label": ["pos", "neg"],
            "img": [b"\x89PNG", None],
            "tags": [["x"], []],
        }
    )
    hf = ds.to_huggingface(class_labels="label", images="img")
    assert hf.features["label"].names == ["neg", "pos"]
    assert type(hf.features["img"]).__name__ == "Image"
    assert hf.num_rows == 2
    stream = ds.to_huggingface("iterable", class_labels={"label": ["neg", "pos"]})
    assert sorted(r["label"] for r in stream) == [0, 1]
