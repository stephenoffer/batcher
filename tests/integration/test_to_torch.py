"""`Dataset.to_torch` / `to_torch_dataloader` — framework-export round trip.

Skipped when `torch` is not installed. Verifies tensor dicts, numeric-only column
selection, re-iterability (multi-epoch), and the DataLoader wrapper.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.integration

torch = pytest.importorskip("torch")


def _ds():
    return bt.from_pydict(
        {
            "x": [1, 2, 3, 4],
            "y": [1.0, 2.0, 3.0, 4.0],
            "label": ["a", "b", "c", "d"],  # non-numeric → skipped
        }
    )


def test_to_torch_tensor_dicts_numeric_only():
    td = _ds().to_torch()
    batches = list(td)
    assert batches, "expected at least one batch"
    for b in batches:
        assert set(b) == {"x", "y"}  # the string column is dropped
        assert all(isinstance(v, torch.Tensor) for v in b.values())


def test_to_torch_is_reiterable_across_epochs():
    td = _ds().to_torch()
    epoch1 = [{k: v.tolist() for k, v in b.items()} for b in td]
    epoch2 = [{k: v.tolist() for k, v in b.items()} for b in td]
    assert epoch1 == epoch2
    # The values survive the round trip.
    merged_x = [x for b in epoch1 for x in b["x"]]
    assert sorted(merged_x) == [1, 2, 3, 4]


def test_to_torch_tensors_are_writable():
    # Training mutates batches in place; the buffer must be owned/writable.
    td = _ds().to_torch()
    first = next(iter(td))
    first["x"] += 1  # must not raise on a read-only Arrow-backed buffer


def test_to_torch_dataloader_iterates():
    dl = _ds().to_torch_dataloader()
    seen = sum(1 for _ in dl)
    assert seen >= 1
