"""Ray Data reads what Batcher's NumPy, WebDataset and TFRecord writers produce.

These writers exist so a Ray Data pipeline can move onto Batcher without changing what lands
on disk, so the check that matters is Ray reading the files back. It starts a private local Ray
instance once for the module (about twenty seconds) and leaves an instance someone else started
untouched. Ray is optional for Batcher, so the module skips cleanly without it.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt

pytestmark = pytest.mark.integration

ray = pytest.importorskip("ray")
ray_data = pytest.importorskip("ray.data")


@pytest.fixture(scope="module", autouse=True)
def _local_ray():
    started = not ray.is_initialized()
    if started:
        ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
    yield
    if started:
        ray.shutdown()


def test_ray_reads_a_numpy_file_as_the_same_rows(tmp_path):
    array = np.arange(6, dtype=np.float32).reshape(3, 2)
    out = str(tmp_path / "x.npy")
    bt.from_numpy(array).write.numpy(out)
    rows = ray_data.read_numpy(out).take_all()
    got = np.stack([row["data"] for row in rows])
    np.testing.assert_array_equal(got, array)
    assert got.dtype == np.float32


def test_ray_reads_webdataset_samples_with_missing_members(tmp_path):
    out = str(tmp_path / "shard.tar")
    bt.from_pydict({"__key__": ["a", "b"], "txt": ["hi", None], "cls": [3, 4]}).write.webdataset(
        out
    )
    rows = sorted(
        ray_data.read_webdataset(out, decoder=None).take_all(), key=lambda r: r["__key__"]
    )
    assert [(r["__key__"], r["txt"], r["cls"]) for r in rows] == [
        ("a", b"hi", b"3"),
        ("b", None, b"4"),
    ]


def test_ray_parses_tfrecord_examples_into_typed_columns(tmp_path):
    try:
        import google_crc32c  # noqa: F401
    except ImportError:
        pytest.importorskip("crc32c")
    out = str(tmp_path / "t.tfrecord")
    ds = bt.from_pydict({"label": [1, None], "text": ["x", "y"], "score": [0.5, 1.5]})
    ds.write.tfrecord(out)
    rows = ray_data.read_tfrecords(out).take_all()
    assert sorted(rows, key=lambda r: r["text"]) == [
        {"label": 1, "text": b"x", "score": 0.5},
        {"label": None, "text": b"y", "score": 1.5},
    ]
