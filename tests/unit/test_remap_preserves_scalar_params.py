"""Regression guard: `remap_columns` must preserve every non-column scalar field.

`remap_columns` (used to push a predicate through a join by rewriting its column
names into one side's source names) rebuilds each node from its remapped children.
A rebuild that forgets a node's *scalar* parameters silently drops them from the
pushed-down expression — the B17-class omission bug, but for a scalar field rather
than a child column.

`AudioFunc.resample(rate)` carries the target sample rate in `rate`; a remap that
rebuilt it as ``AudioFunc(fn, input)`` reset ``rate`` to ``None``, so a resample
pushed below a join lost its rate and the engine produced the wrong (or a
default-rate) waveform.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.plan.expr_ir.audio import AudioFunc
from batcher.plan.expr_ir.walk import remap_columns


@pytest.mark.unit
def test_remap_preserves_audio_resample_rate() -> None:
    expr = bt.col("b").audio.resample(16000)
    remapped = remap_columns(expr, {"b": "c"})

    assert isinstance(remapped, AudioFunc)
    assert remapped.rate == 16000  # was reset to None before the fix
    assert remapped.to_ir() == {
        "e": "audio",
        "fn": "resample",
        "input": {"e": "col", "name": "c"},
        "rate": 16000,
    }


@pytest.mark.unit
def test_remap_audio_resample_rate_nested_in_predicate() -> None:
    # A resample buried inside a larger expression must keep its rate through remap.
    expr = bt.col("b").audio.resample(22050).list.get(0)
    remapped = remap_columns(expr, {"b": "c"})
    assert remapped.input.rate == 22050
