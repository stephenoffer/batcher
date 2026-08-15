"""`ds.ml.to_tf` prepares its stream with the same code the PyTorch loader does.

It was the one loader in the package with no shuffle, no `drop_last` and no dtype control,
which meant a TensorFlow user either trained on the corpus in storage order or rebuilt the
stream by hand. These tests hold the two paths to the same options rather than to two
implementations of them.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError

tf = pytest.importorskip("tensorflow")


def _ds(n: int = 100) -> bt.Dataset:
    return bt.from_pydict({"f": [float(i) for i in range(n)], "label": [i % 2 for i in range(n)]})


def _widths(tf_ds) -> list[int]:
    return [int(batch["f"].shape[0]) for batch in tf_ds]


def test_to_tf_batches_to_the_requested_width():
    assert _widths(_ds().ml.to_tf(batch_size=16)) == [16, 16, 16, 16, 16, 16, 4]


def test_to_tf_drop_last_removes_only_the_ragged_tail():
    # A fixed-shape graph cannot take the short final batch, and a TF user had no way to
    # ask for it to go.
    assert _widths(_ds().ml.to_tf(batch_size=16, drop_last=True)) == [16] * 6


def test_to_tf_shuffles_locally_and_reproducibly():
    def rows(**kw) -> list[float]:
        return [v for b in _ds().ml.to_tf(batch_size=16, **kw) for v in b["f"].numpy().tolist()]

    shuffled = rows(local_shuffle_buffer_size=48, seed=1)
    assert sorted(shuffled) == [float(i) for i in range(100)], "no row may be lost or repeated"
    assert shuffled != [float(i) for i in range(100)], "the stream came back in storage order"
    assert shuffled == rows(local_shuffle_buffer_size=48, seed=1), "same seed, same order"
    assert shuffled != rows(local_shuffle_buffer_size=48, seed=1, epoch=1), "epoch must reshuffle"


def test_to_tf_casts_dtypes_the_same_way_the_torch_loader_names_them():
    every = next(iter(_ds().ml.to_tf(batch_size=8, dtypes="fp16")))
    assert every["f"].dtype == tf.float16
    assert every["label"].dtype == tf.float16
    per_column = next(iter(_ds().ml.to_tf(batch_size=8, dtypes={"f": "float32"})))
    assert per_column["f"].dtype == tf.float32
    assert per_column["label"].dtype == tf.int64, "an unnamed column keeps its source dtype"


def test_a_dtype_numpy_cannot_represent_numerically_is_refused():
    # `np.dtype("bfloat16")` SUCCEEDS when `ml_dtypes` is installed — TensorFlow depends on
    # it — and yields an opaque kind "V". Every numeric check then rejected the column, so
    # asking for bf16 emptied the batch and blamed the columns instead of the dtype.
    with pytest.raises(PlanError, match="no numeric NumPy equivalent"):
        next(iter(_ds().ml.to_tf(batch_size=8, dtypes="bf16")))


def test_the_first_batch_is_not_eaten_by_the_signature_probe():
    # `from_generator` needs a shape/dtype signature, which means peeking at batch 0 of a
    # one-shot stream. Peeking with a second pass silently dropped it from every TF run.
    assert sum(_widths(_ds(40).ml.to_tf(batch_size=10))) == 40
