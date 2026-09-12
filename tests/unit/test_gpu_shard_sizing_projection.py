"""Sizing a shard must not charge a whole row's width for every column it cannot resolve.

`_row_bytes` narrows a relation's decoded width to the columns a plan projects. When a projected
name is absent from the source schema it charged `_FALLBACK_ROW_BYTES` (128) for that column —
deliberately generous, and correct for *one* unknown column among known ones.

It is wrong when **nothing** resolves, which is the ordinary case for a renamed scan rather than
a corner: TPC-H Parquet in the wild is named positionally (`column00`, ...), every query names
its columns, and schema-on-read renaming leaves the *source* positional while the projection
carries canonical names. Every lookup then misses and the relation is charged 128 bytes per
projected column. Measured on sf100 `lineitem` with seven projected columns: **500.7 GiB against
a real ~40 GiB**, so `plan_shard_count` asked for 24 shards where the data wanted 6.

The rule that replaces it: a projection that shares no name with the schema cannot be applied,
so size the whole row from the schema that *can* be read — a real measurement of the relation
and an upper bound on any projection of it, which keeps the conservative direction without
inventing a width per column.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.dist.gpu.shards import _FALLBACK_ROW_BYTES, _row_bytes

pytestmark = pytest.mark.unit


class _Source:
    def __init__(self, schema: pa.Schema) -> None:
        self._schema = schema

    def schema(self) -> pa.Schema:
        return self._schema


_POSITIONAL = pa.schema([pa.field(f"column{i:02d}", pa.int64()) for i in range(16)])
_NAMED = pa.schema(
    [pa.field("a", pa.int64()), pa.field("b", pa.int64()), pa.field("c", pa.float64())]
)


def test_an_unresolvable_projection_is_sized_from_the_whole_row():
    """Seven canonical names against a positional schema: the row, not 7 x 128."""
    wanted = ["l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_shipdate"]
    got = _row_bytes(_Source(_POSITIONAL), wanted)

    whole_row = 16 * 8  # sixteen int64 columns
    assert got == pytest.approx(whole_row)
    assert got < _FALLBACK_ROW_BYTES * len(wanted)


def test_a_resolvable_projection_is_still_narrowed():
    """The positive control. Without it, an assertion about the unresolvable case would pass
    against an implementation that had stopped narrowing at all — which would size every
    projected scan at its full relation width and cut shards for columns nobody reads."""
    assert _row_bytes(_Source(_NAMED), ["a", "b"]) == pytest.approx(16.0)
    assert _row_bytes(_Source(_NAMED), None) == pytest.approx(24.0)


def test_one_unknown_column_among_known_ones_still_pays_the_generous_figure():
    """A partially-resolvable projection is a different situation and keeps the old rule: the
    namespaces do match, so a missing name really is one column of unknown width."""
    got = _row_bytes(_Source(_NAMED), ["a", "mystery"])
    assert got == pytest.approx(8.0 + _FALLBACK_ROW_BYTES)


def test_a_source_that_will_not_describe_itself_keeps_the_fallback():
    """An unreadable schema is the case the constant was written for, and is unchanged."""

    class _Mute:
        def schema(self):
            raise RuntimeError("no footer")

    assert _row_bytes(_Mute(), ["a"]) == pytest.approx(float(_FALLBACK_ROW_BYTES))


def test_an_empty_schema_falls_back_rather_than_returning_zero():
    """Zero bytes per row would make `plan_shard_count` see a relation of no size and cut one
    shard for any input — the opposite failure, and worse, because it ends in a dead device."""
    assert _row_bytes(_Source(pa.schema([])), ["a"]) == pytest.approx(float(_FALLBACK_ROW_BYTES))
