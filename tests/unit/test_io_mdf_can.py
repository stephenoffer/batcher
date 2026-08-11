"""ASAM MDF4 — vehicle CAN/sensor measurements, and fusing them with a robot log.

MDF is what automotive OEMs and test fleets log to. The shape that drives the design: one
file holds several *channel groups*, each with **its own sampling raster** — powertrain at
100 Hz, chassis at 4 Hz. There is no single wide table, so this reads long format (one row
per signal sample) and leaves resampling an explicit choice rather than something the
reader did silently.

The payoff, and the last test here, is the ADAS query: because `timestamp` is absolute, a
CAN measurement as-of joins against an MCAP log from the same drive — attaching vehicle
speed to every LiDAR sweep, across two different file formats.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("asammdf")

from asammdf import MDF, Signal

from batcher.io.formats.robotics import MDFSource
from batcher.io.formats.robotics.mdf import _epoch_nanos

pytestmark = pytest.mark.unit

_START = dt.datetime(2026, 7, 18, 12, 0, 0, tzinfo=dt.UTC)


def _write_measurement(path, *, seconds: float = 2.0) -> None:
    """Two channel groups at different rasters — the shape that forbids a wide table."""
    fast = np.arange(0, seconds, 0.01)  # 100 Hz
    slow = np.arange(0, seconds, 0.25)  # 4 Hz
    mdf = MDF()
    mdf.append(
        [
            Signal(np.full(len(fast), 50.0), fast, name="VehicleSpeed", unit="km/h"),
            Signal((fast * 1000).astype("f8"), fast, name="EngineRPM", unit="rpm"),
        ]
    )
    mdf.append([Signal(np.arange(len(slow)).astype("i4"), slow, name="SteeringAngle", unit="deg")])
    mdf.start_time = _START
    mdf.save(str(path), overwrite=True)
    mdf.close()


@pytest.fixture
def measurement(tmp_path):
    _write_measurement(tmp_path / "drive.mf4")
    return str(tmp_path)


def _table(src: MDFSource, **kw) -> pa.Table:
    return pa.Table.from_batches(list(src.iter_batches(**kw)), schema=src.schema())


def test_a_row_is_one_sample_of_one_signal(measurement) -> None:
    table = _table(MDFSource(measurement))

    assert table.schema.names == ["signal", "timestamp", "value", "unit"]
    # 200 + 200 fast samples, 8 slow ones — one schema across two rasters.
    assert table.num_rows == 408


def test_signals_at_different_rasters_keep_their_own_sample_counts(measurement) -> None:
    """The reason for long format: widening these would resample or pad with nulls."""
    table = _table(MDFSource(measurement))
    counts: dict[str, int] = {}
    for name in table.column("signal").to_pylist():
        counts[name] = counts.get(name, 0) + 1

    assert counts == {"VehicleSpeed": 200, "EngineRPM": 200, "SteeringAngle": 8}


def test_units_are_carried_per_signal(measurement) -> None:
    table = _table(MDFSource(measurement))
    pairs = set(
        zip(table.column("signal").to_pylist(), table.column("unit").to_pylist(), strict=True)
    )

    assert pairs == {("VehicleSpeed", "km/h"), ("EngineRPM", "rpm"), ("SteeringAngle", "deg")}


def test_timestamps_are_absolute(measurement) -> None:
    """Channel offsets are seconds from the measurement start; absolute time is what lets
    a measurement be joined against a log recorded by a different system."""
    table = _table(MDFSource(measurement)).sort_by([("timestamp", "ascending")])

    assert table.column("timestamp")[0].as_py().replace(tzinfo=dt.UTC) == _START


def test_signals_lists_the_readable_channels(measurement) -> None:
    assert MDFSource(measurement).signals() == ["EngineRPM", "SteeringAngle", "VehicleSpeed"]


def test_a_signal_restriction_reads_only_those_channels(measurement) -> None:
    src = MDFSource(measurement, signals=["EngineRPM"])
    table = _table(src)

    assert table.num_rows == 200
    assert set(table.column("signal").to_pylist()) == {"EngineRPM"}


def test_read_and_iter_batches_agree(measurement) -> None:
    src = MDFSource(measurement)
    assert pa.Table.from_batches(src.read(), schema=src.schema()).equals(_table(src))


def test_projection_is_honored(measurement) -> None:
    batch = next(iter(MDFSource(measurement).iter_batches(["signal", "value"])))
    assert batch.schema.names == ["signal", "value"]


def test_the_row_count_is_reported_as_inexact(measurement) -> None:
    """It is `cycles x channels`, an upper bound once a restriction or a non-numeric
    channel is involved — so it must not be used to answer a `count()`."""
    stats = MDFSource(measurement).statistics()

    assert stats.row_count == 408
    assert not stats.exact_rows


def test_a_signal_restriction_survives_the_split_round_trip(tmp_path) -> None:
    """A worker rebuilds its reader from the pickled split alone. If `signals` is not
    carried, a distributed read returns every channel while a local one returns three."""
    import pickle

    for i in range(2):
        _write_measurement(tmp_path / f"d{i}.mf4")
    src = MDFSource(str(tmp_path), signals=["EngineRPM"])

    shipped = [pickle.loads(pickle.dumps(s)) for s in src.splits()]
    rows = sum(sum(b.num_rows for b in s.read()) for s in shipped)

    assert rows == 400
    assert rows == _table(src).num_rows


def test_an_unreadable_measurement_can_be_skipped(tmp_path) -> None:
    _write_measurement(tmp_path / "good.mf4")
    (tmp_path / "zbad.mf4").write_bytes(b"not an mdf file")

    src = MDFSource(str(tmp_path), on_error="skip")
    assert _table(src).num_rows == 408
    assert [p.rsplit("/", 1)[-1] for p in src.corrupt_files()] == ["zbad.mf4"]


def test_can_signals_fuse_with_a_robot_log_across_formats(tmp_path) -> None:
    """The ADAS query: attach vehicle speed to every LiDAR sweep, MDF4 joined to MCAP.

    Two different files, two different recording systems, two different rates — aligned
    because both carry absolute time and `join_asof` takes the most recent sample at or
    before each sweep.
    """
    pytest.importorskip("mcap")
    from mcap.writer import Writer

    import batcher as bt
    from batcher import col

    can_dir, log_dir = tmp_path / "can", tmp_path / "log"
    can_dir.mkdir()
    log_dir.mkdir()

    # CAN: 100 Hz speed, cruising then braking hard in the second half.
    stamps = np.arange(0, 2, 0.01)
    speed = np.concatenate([np.full(100, 50.0), np.linspace(50, 2, 100)])
    mdf = MDF()
    mdf.append([Signal(speed, stamps, name="VehicleSpeed", unit="km/h")])
    mdf.start_time = _START
    mdf.save(str(can_dir / "drive.mf4"), overwrite=True)
    mdf.close()

    # LiDAR: 10 Hz sweeps over the same window.
    epoch_ns = int(_START.timestamp() * 1_000_000_000)
    with open(log_dir / "drive.mcap", "wb") as fh:
        writer = Writer(fh)
        writer.start()
        schema_id = writer.register_schema(name="PointCloud2", encoding="ros2msg", data=b"x")
        channel = writer.register_channel(
            topic="/lidar/top", message_encoding="cdr", schema_id=schema_id
        )
        for i in range(20):
            stamp = epoch_ns + int(i * 0.1 * 1_000_000_000)
            writer.add_message(
                channel_id=channel, log_time=stamp, publish_time=stamp, sequence=i, data=b"s"
            )
        writer.finish()

    can = bt.read.mdf(str(can_dir), signals=["VehicleSpeed"]).select(
        "timestamp", speed=col("value")
    )
    lidar = (
        bt.read.mcap(str(log_dir))
        .filter(col("topic") == "/lidar/top")
        .select(timestamp=col("log_time"), sweep=col("sequence"))
    )
    fused = lidar.join_asof(can, on="timestamp").collect().to_pydict()

    assert len(fused["sweep"]) == 20, "one row per sweep"
    assert all(v is not None for v in fused["speed"]), "a sweep got no speed"
    by_sweep = dict(zip(fused["sweep"], fused["speed"], strict=True))
    assert by_sweep[0] == pytest.approx(50.0), "cruising at the start"
    assert by_sweep[19] < 10.0, "braking at the end"

    # Scenario extraction — the query an ADAS corpus exists to answer.
    braking = lidar.join_asof(can, on="timestamp").filter(col("speed") < 20)
    assert 0 < braking.count() < 20


def test_the_measurement_origin_is_exact_to_the_nanosecond() -> None:
    """`start_time` converts without going through a float that cannot hold it.

    `start.timestamp() * 1e9` is a `float64` around 1.7e18, where the spacing between
    representable values is 256 ns — so the conversion quantizes the measurement's origin
    and shifts *every* sample in the file by the same amount. The whole point of an
    absolute `timestamp` column is joining against a nanosecond-stamped log from another
    system, so the error is small and still worth not having.

    A whole-second start time cannot see this; the sub-second ones below can.
    """
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
    inexact = []
    for micros in range(0, 1_000_000, 9_973):
        start = dt.datetime(2026, 7, 18, 12, 0, 0, micros, tzinfo=dt.UTC)
        delta = start - epoch
        want = (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000
        got = _epoch_nanos(start)
        if got != want:
            inexact.append((micros, got - want))
    assert not inexact, f"{len(inexact)} start times converted inexactly: {inexact[:4]}"


def test_a_naive_measurement_origin_is_read_as_utc() -> None:
    """Guessing the recorder's local zone would shift a drive by hours, invisibly."""
    naive = dt.datetime(2026, 7, 18, 12, 0, 0, 500_000)
    aware = naive.replace(tzinfo=dt.UTC)
    assert _epoch_nanos(naive) == _epoch_nanos(aware)


def test_a_measurement_before_the_epoch_converts_with_the_right_sign() -> None:
    """`timedelta` normalizes to a negative `days` with non-negative seconds, which the
    arithmetic has to add up rather than treat as two independent signs."""
    assert _epoch_nanos(dt.datetime(1969, 12, 31, 23, 59, 59, tzinfo=dt.UTC)) == -1_000_000_000
