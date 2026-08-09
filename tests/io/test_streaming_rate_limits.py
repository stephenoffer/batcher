"""Per-trigger admission control: how much of a backlog one micro-batch may take.

A streaming source with a backlog has to decide how much of it to hand over at once, and
the two bounds that matter are different questions: a file *count* says nothing about
memory when files range from 4 KiB of JSON to 8 GiB of Parquet. Spark spells them
`maxFilesPerTrigger` / `maxBytesPerTrigger` for files and `maxOffsetsPerTrigger` /
`maxBytesPerTrigger` for a broker; these pin that both bounds exist here, compose, and
never stall the stream on a single oversized arrival.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher._internal.errors import PlanError
from batcher.io.formats.streaming.autoloader import IncrementalFileSource
from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource


def _write(directory, name: str, rows: int) -> None:
    pq.write_table(
        pa.table({"a": pa.array(list(range(rows)), type=pa.int64())}), str(directory / name)
    )


@pytest.fixture
def landing(tmp_path):
    data = tmp_path / "landing"
    data.mkdir()
    return data, str(tmp_path / "state")


def test_max_files_per_trigger_caps_a_discovery_pass(landing):
    data, state = landing
    for i in range(5):
        _write(data, f"f{i}.parquet", 1)
    source = IncrementalFileSource(str(data), "parquet", state_dir=state, max_files_per_trigger=2)

    assert len(source.discover()) == 2
    assert len(source.discover()) == 2
    assert len(source.discover()) == 1


def test_max_bytes_per_trigger_caps_a_pass_by_size(landing):
    data, state = landing
    for i in range(4):
        _write(data, f"f{i}.parquet", 200)
    one = (data / "f0.parquet").stat().st_size
    # Room for two files but not three.
    source = IncrementalFileSource(
        str(data), "parquet", state_dir=state, max_bytes_per_trigger=one * 2 + 1
    )

    first = source.discover()
    assert 1 <= len(first) < 4, "the byte budget admitted the whole backlog"
    assert sum((data / name.rsplit("/", 1)[-1]).stat().st_size for name in first) <= one * 2 + 1


def test_a_single_file_over_the_budget_is_still_admitted(landing):
    """Refusing it would stall the stream permanently: the next pass refuses it again."""
    data, state = landing
    _write(data, "big.parquet", 5000)
    source = IncrementalFileSource(str(data), "parquet", state_dir=state, max_bytes_per_trigger=1)
    assert len(source.discover()) == 1


def test_the_two_bounds_compose_and_the_tighter_one_wins(landing):
    data, state = landing
    for i in range(6):
        _write(data, f"f{i}.parquet", 1)
    source = IncrementalFileSource(
        str(data),
        "parquet",
        state_dir=state,
        max_files_per_trigger=4,
        max_bytes_per_trigger=10**9,
    )
    assert len(source.discover()) == 4


@pytest.mark.parametrize("kwarg", ["max_files_per_trigger", "max_bytes_per_trigger"])
def test_a_nonpositive_bound_is_refused_by_name(landing, kwarg):
    data, state = landing
    with pytest.raises(PlanError, match=kwarg):
        IncrementalFileSource(str(data), "parquet", state_dir=state, **{kwarg: 0})


class _Broker(BrokerSource):
    """A minimal concrete broker, for the base class's own contracts."""

    format_name = "test_broker"

    def _discover_partitions(self):
        return [0]

    def _poll(self):
        return [BrokerMessage(value=b"v", partition=0, offset=0, timestamp=0, topic=self.topic)]


def test_max_offsets_per_trigger_is_the_spark_spelling_of_poll_size():
    assert _Broker("t", max_offsets_per_trigger=500).poll_size == 500


def test_max_bytes_per_trigger_is_the_spark_spelling_of_poll_bytes():
    assert _Broker("t", max_bytes_per_trigger=4096).poll_bytes == 4096


def test_a_spark_named_bound_never_reaches_the_client_config():
    """Forwarded as an unknown option it would land in the broker client's config, where
    it is rejected outright or — worse — ignored, leaving the stream unthrottled."""
    source = _Broker("t", max_offsets_per_trigger=10, bootstrap_servers="b:9092")
    assert "max_offsets_per_trigger" not in source._options
    assert source._options == {"bootstrap_servers": "b:9092"}


def test_an_explicit_poll_size_is_overridden_by_the_spark_spelling():
    assert _Broker("t", poll_size=99, max_offsets_per_trigger=7).poll_size == 7


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"poll_size": 0}, "poll_size"),
        ({"max_offsets_per_trigger": -1}, "max_offsets_per_trigger"),
        ({"poll_bytes": 0}, "poll_bytes"),
        ({"max_bytes_per_trigger": -5}, "max_bytes_per_trigger"),
    ],
)
def test_a_nonpositive_broker_bound_is_refused(kwargs, message):
    with pytest.raises(PlanError, match=message):
        _Broker("t", **kwargs)
