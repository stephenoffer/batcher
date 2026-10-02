"""Clustering an Iceberg scan declares under *transformed* and *evolved* partition specs.

The integration tests write identity-partitioned tables, because writing a `bucket` or `days`
table needs the `pyiceberg-core` extra this repository does not depend on. So the transform
half of `IcebergSource._common_clustering` was never exercised. It needs no table to test:
the decision reads only the specs and the schema, which `pyiceberg` builds in pure Python.

What must hold, from the rule the method states: a field shared by every live spec with the
same source column *and* transform is claimed, by its source column, at each spec's own
position; a field whose transform changed (`days` then `hours`) is not; and nothing common
means nothing is declared, so the scan keeps its shuffle.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import BucketTransform, DayTransform, HourTransform, VoidTransform
from pyiceberg.types import LongType, NestedField, StringType, TimestampType

from batcher.io.formats.lakehouse.iceberg.source import IcebergSource

pytestmark = pytest.mark.unit

_SCHEMA = Schema(
    NestedField(1, "ts", TimestampType(), required=False),
    NestedField(2, "region", StringType(), required=False),
    NestedField(3, "id", LongType(), required=False),
)
_DAY = PartitionField(1, 1000, DayTransform(), "ts_day")
_BUCKET = PartitionField(3, 1001, BucketTransform(16), "id_bucket")


def _source(current: PartitionSpec, specs: dict) -> IcebergSource:
    """An `IcebergSource` whose table exposes exactly these specs, and nothing else."""

    class _Table:
        def spec(self):
            return current

        def specs(self):
            return specs

        def schema(self):
            return _SCHEMA

    class _Stubbed(IcebergSource):
        __slots__ = ()

        def _table(self):
            return _Table()

    return object.__new__(_Stubbed)


def test_a_bucket_and_day_spec_clusters_on_both_source_columns():
    spec = PartitionSpec(_BUCKET, _DAY, spec_id=0)
    assert _source(spec, {0: spec})._common_clustering({0}) == (("id", "ts"), {0: (0, 1)})


def test_an_evolved_spec_keeps_the_common_transformed_field_at_each_specs_position():
    """`days(ts)` evolved to `(bucket(id), days(ts))`: `ts` is common, read at 0 and at 1."""
    old = PartitionSpec(_DAY, spec_id=0)
    new = PartitionSpec(_BUCKET, _DAY, spec_id=1)
    got = _source(new, {0: old, 1: new})._common_clustering({0, 1})
    assert got == (("ts",), {0: (0,), 1: (1,)})


def test_a_changed_transform_on_one_column_is_not_common():
    """A day-file and an hour-file with the same `ts` carry different values, so no claim."""
    old = PartitionSpec(_DAY, spec_id=0)
    new = PartitionSpec(PartitionField(1, 1002, HourTransform(), "ts_hour"), spec_id=2)
    assert _source(new, {0: old, 2: new})._common_clustering({0, 2}) == ((), {})


def test_a_bucket_count_change_is_a_different_transform():
    """`bucket[16](id)` and `bucket[32](id)` place one id in different buckets."""
    old = PartitionSpec(_BUCKET, spec_id=0)
    new = PartitionSpec(PartitionField(3, 1003, BucketTransform(32), "id_bucket32"), spec_id=1)
    assert _source(new, {0: old, 1: new})._common_clustering({0, 1}) == ((), {})


def test_a_void_field_is_never_claimed():
    """A v1 table drops a field by voiding it; the remaining day field is still claimed."""
    spec = PartitionSpec(PartitionField(3, 1001, VoidTransform(), "id_void"), _DAY, spec_id=0)
    assert _source(spec, {0: spec})._common_clustering({0}) == (("ts",), {0: (1,)})


def test_a_file_under_an_unknown_spec_declares_nothing():
    spec = PartitionSpec(_DAY, spec_id=0)
    assert _source(spec, {0: spec})._common_clustering({0, 7}) == ((), {})
