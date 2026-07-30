"""When a GPU worker may read its own shard, and the far more important when it may not.

The device Parquet reader is a *second implementation* of Parquet. Its value is that it skips
a CPU decode and a trip across the bus; its risk is that it produces rows the host reader would
not have. Every test here is about the second thing, because the first one only pays if the
second never happens.

None of this needs a GPU: the decision is made from the descriptor before any device is
touched, which is exactly why it can be tested at all.
"""

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa
import pytest

from batcher.core.gpu_plan import DfBackend
from batcher.dist.gpu.device_read import read_descriptor_on_device
from batcher.io.splits.device import DeviceReadSpec, device_read_specs
from batcher.io.splits.file import FileSplit
from batcher.io.splits.parquet import RowGroupSplit

pytestmark = pytest.mark.unit

PLAIN = pa.schema(
    [
        pa.field("i", pa.int64()),
        pa.field("f", pa.float64()),
        pa.field("s", pa.string()),
        pa.field("b", pa.bool_()),
        pa.field("t", pa.timestamp("us")),
        pa.field("d", pa.date32()),
    ]
)


class _Split:
    """A split standing in for a locator kind, carrying only the schema the gate reads."""

    def __init__(self, schema: pa.Schema):
        self._schema = schema

    def schema(self) -> pa.Schema:
        return self._schema


def _row_group(path="a.parquet", groups=(0, 1), schema=PLAIN):
    """A `RowGroupSplit` whose schema is declared rather than read from a real file.

    Subclassed rather than patched: the split types are frozen and slotted, which is the
    point of them, and reading a schema is the one thing they do that needs a file to exist.
    """

    class _Declared(RowGroupSplit):
        def schema(self) -> pa.Schema:
            return schema

    return _Declared(path, groups)


def _file(path="a.parquet", fmt="parquet", schema=PLAIN, **kwargs):
    class _Declared(FileSplit):
        def schema(self) -> pa.Schema:
            return schema

    return _Declared(fmt, path, kwargs)


# --- which splits carry a locator -----------------------------------------------------


def test_row_group_splits_carry_their_row_groups():
    specs = device_read_specs([_row_group("a.parquet", (0, 2))], None)
    assert specs == [DeviceReadSpec("a.parquet", (0, 2))]


def test_a_whole_parquet_file_split_carries_no_row_groups():
    assert device_read_specs([_file("b.parquet")], None) == [DeviceReadSpec("b.parquet", None)]


def test_a_non_parquet_file_split_has_no_locator():
    assert device_read_specs([_file("b.csv", fmt="csv")], None) is None


def test_a_file_split_with_reader_arguments_has_no_locator():
    """Those arguments change what the host reader returns and the device reader ignores them."""
    assert device_read_specs([_file("b.parquet", sheet="Sheet2")], None) is None


def test_an_unrecognized_split_has_no_locator():
    assert device_read_specs([_Split(PLAIN)], None) is None


def test_no_splits_means_no_locator():
    assert device_read_specs([], None) is None


def test_one_unreadable_split_disqualifies_the_whole_shard():
    """Half on the device and half on the host concatenates two readers' schemas."""
    assert device_read_specs([_row_group(), _file("b.csv", fmt="csv")], None) is None


# --- which types are allowed to cross -------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [
        pa.decimal128(10, 2),
        pa.list_(pa.int64()),
        pa.struct([pa.field("x", pa.int64())]),
        pa.dictionary(pa.int32(), pa.string()),
        pa.binary(),
        pa.map_(pa.string(), pa.int64()),
    ],
)
def test_a_type_the_two_readers_may_disagree_on_is_declined(dtype):
    schema = pa.schema([pa.field("i", pa.int64()), pa.field("odd", dtype)])
    assert device_read_specs([_row_group(schema=schema)], None) is None


def test_an_unsupported_column_nobody_selected_does_not_disqualify_the_read():
    """The gate asks about the columns that will cross, not about every column in the file."""
    schema = pa.schema([pa.field("i", pa.int64()), pa.field("odd", pa.decimal128(10, 2))])
    assert device_read_specs([_row_group(schema=schema)], ["i"]) is not None


@pytest.mark.parametrize("column", ["i", "f", "s", "b", "t", "d"])
def test_the_core_types_are_allowed(column):
    assert device_read_specs([_row_group()], [column]) is not None


# --- the descriptor-level gate --------------------------------------------------------


@pytest.fixture
def host_backend():
    import pandas as pd

    return DfBackend(pd)


def test_a_host_backend_never_reads_on_the_device(host_backend):
    """The verification path must exercise the host reader the engine's own tests cover."""
    descriptor = {"splits": [_row_group()], "projection": None, "predicate": None}
    assert read_descriptor_on_device(descriptor, host_backend) is None


def test_an_in_memory_descriptor_has_nothing_to_read_from_storage(host_backend):
    assert read_descriptor_on_device({"batches": []}, host_backend) is None


def test_a_pushed_predicate_keeps_the_reader_that_can_use_it():
    """The device read cannot skip row groups, so it would move more bytes, not fewer."""
    from batcher.dist.gpu.device_read import _specs

    descriptor = {
        "splits": [_row_group()],
        "projection": None,
        "predicate": {"e": "binary", "op": "gt"},
    }
    assert _specs(descriptor) is None
    assert _specs({**descriptor, "predicate": None}) is not None


def test_a_schema_the_device_reader_reordered_is_rejected():
    """Two readers agreeing on types is not the same as agreeing on column order."""
    from batcher.dist.gpu.device_read import _schema_agrees

    class _Frame:
        columns: ClassVar[list[str]] = ["f", "i"]

    descriptor = {"splits": [_row_group()]}
    assert _schema_agrees(_Frame(), descriptor, ["i", "f"]) is False
    assert _schema_agrees(_Frame(), descriptor, ["f", "i"]) is True


def test_the_expected_names_come_from_the_split_when_nothing_was_projected():
    from batcher.dist.gpu.device_read import _expected_names

    assert _expected_names({"splits": [_row_group()]}, None) == PLAIN.names
