"""`ds.ml.infer`/`embed` refuse options that their callable shape cannot honor.

Both methods have two shapes. Given a model identifier they build the encoder and honor
`column`, `output_column`, `device`, `normalize`; given a callable they forward to
`map_batches`, where none of those exist. Every one of them used to be dropped in silence,
so ``embed(MyEncoder, normalize=True)`` returned unnormalized vectors and looked fine.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError


class Encoder:
    def __call__(self, batch: pa.RecordBatch) -> dict:
        return {"pred": np.arange(batch.num_rows, dtype="float32")}


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_pydict({"x": [1, 2], "t": ["a", "b"]})


def test_callable_infer_still_works(ds: bt.Dataset) -> None:
    assert ds.ml.infer(Encoder, output_columns=["pred"]).to_pydict() == {"pred": [0.0, 1.0]}


@pytest.mark.parametrize(
    "option", [{"column": "t"}, {"output_column": "p"}, {"task": "x"}, {"device": "cuda"}]
)
def test_infer_rejects_model_id_only_options(ds: bt.Dataset, option: dict) -> None:
    with pytest.raises(PlanError, match=next(iter(option))):
        ds.ml.infer(Encoder, output_columns=["pred"], **option).to_pydict()


@pytest.mark.parametrize(
    "option", [{"normalize": True}, {"fp16": True}, {"output_column": "e"}, {"column": "t"}]
)
def test_embed_rejects_model_id_only_options(ds: bt.Dataset, option: dict) -> None:
    with pytest.raises(PlanError, match=next(iter(option))):
        ds.ml.embed(Encoder, output_columns=["pred"], **option).to_pydict()


def test_the_message_names_every_ignored_option(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError) as caught:
        ds.ml.embed(Encoder, output_columns=["pred"], normalize=True, fp16=True).to_pydict()
    assert "'fp16'" in str(caught.value) and "'normalize'" in str(caught.value)


def test_options_that_do_apply_are_untouched(ds: bt.Dataset) -> None:
    """`batch_size`/`num_workers`/`concurrency` forward to `map_batches` and must stay legal."""
    out = ds.ml.infer(Encoder, output_columns=["pred"], batch_size=1, num_workers=2, concurrency=1)
    assert out.to_pydict()["pred"] == [0.0, 0.0]
