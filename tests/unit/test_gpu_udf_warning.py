"""GPU map_batches with a plain function warns (the model-reload-per-batch foot-gun).

Ray Data's most common inference mistake is a plain-function UDF on a GPU stage:
the model reloads on every batch. Batcher warns at plan-construction time and
points at the class/factory spelling that loads once per worker. CPU stages and
class UDFs do not warn. Pure plan construction — no engine needed.
"""

from __future__ import annotations

import warnings

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PerformanceWarning


class _Model:
    def __call__(self, batch):  # loaded once per worker
        return batch


def test_gpu_plain_function_warns():
    ds = bt.from_arrow(pa.table({"x": [1, 2, 3]}))
    with pytest.warns(PerformanceWarning, match="once per worker"):
        ds.ml.map_batches(lambda b: b, num_gpus=1, output_columns=["x"])


def test_gpu_class_does_not_warn():
    """A class is constructed once per worker, which is the thing the warning asks for."""
    ds = bt.from_arrow(pa.table({"x": [1, 2, 3]}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        out = ds.ml.map_batches(_Model, num_gpus=1, output_columns=["x"])
    assert out.columns == ["x"]


def test_cpu_plain_function_does_not_warn():
    """The warning is about per-worker model load cost, which a CPU stage does not pay."""
    ds = bt.from_arrow(pa.table({"x": [1, 2, 3]}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = ds.ml.map_batches(lambda b: b, output_columns=["x"])  # num_gpus=0 default
    assert out.columns == ["x"]


# --- projection pushdown, defeated by an undeclared UDF -------------------------------
# Pushdown is the highest-impact IO optimization in the field guides (2-10x on a wide
# table, 10-50x past 50 columns) and the one Batcher does automatically — right up to a
# `map_batches`, whose `fn` the optimizer cannot see into. `input_columns` is the only way
# back, and it cannot be inferred, so the one case where automatic pushdown stops working
# is said out loud at the call site that caused it.


def _wide(n: int) -> bt.Dataset:
    return bt.from_arrow(pa.table({f"c{i}": [1, 2, 3] for i in range(n)}))


def test_a_wide_table_without_input_columns_warns():
    with pytest.warns(PerformanceWarning, match="input_columns"):
        _wide(20).ml.map_batches(lambda b: b)


def test_declaring_input_columns_silences_it():
    """The declaration is the fix, so making it must remove the advice."""
    ds = _wide(20)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ds.ml.map_batches(lambda b: b, input_columns=["c0", "c1"])


def test_a_narrow_table_does_not_warn():
    """Below the wide-table threshold the unpruned read costs little, and advice on every
    narrow table is how a reader learns to filter these out."""
    ds = _wide(3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ds.ml.map_batches(lambda b: b)


def test_an_append_a_column_udf_is_not_told_to_prune():
    """`ds.ml.generate`/`embed`/`classify` return every input column plus new ones, so they
    genuinely read all of them. Telling them to declare `input_columns` is wrong advice, and
    wrong advice on the most common ML entry points is worse than none."""
    ds = _wide(20)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ds.ml.generate(lambda: lambda p: ["x"] * len(p), prompt_column="c0")


def test_a_narrowing_udf_over_a_wide_table_still_warns():
    """Declaring output_columns must not become a blanket exemption: a stage that returns
    fewer columns than it took is exactly the one that should have pruned its read."""
    ds = _wide(20)
    with pytest.warns(PerformanceWarning, match="input_columns"):
        ds.ml.map_batches(lambda b: b.select(["c0"]), output_columns=["c0"])
