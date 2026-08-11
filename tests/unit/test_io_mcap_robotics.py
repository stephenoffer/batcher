"""MCAP — robot and vehicle logs as a relation, with index-backed pushdown.

MCAP is the ROS 2 recording format and the ADAS interchange format: one file multiplexes
every sensor as timestamped messages on named topics. Two properties make it worth a real
connector rather than a blob reader, and both are tested here:

* the container is **indexed**, so a row count, the log-time bounds, and a topic- or
  time-filtered read come from the summary rather than a scan;
* pushing a filter down must return **exactly** what filtering afterwards returns. The
  boundary semantics are the trap — MCAP's ``start_time`` is inclusive and ``end_time`` is
  *exclusive*, so an off-by-one drops the last message of a window and nothing in a row
  count says so.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

pytest.importorskip("mcap")

from mcap.writer import Writer

from batcher.io.formats.robotics import MCAPSource
from batcher.plan.stats import Provenance

pytestmark = pytest.mark.unit

# A plausible drive-log epoch, in nanoseconds.
_T0 = 1_700_000_000_000_000_000
_TICK = 1_000_000_000  # 1 s


def _write_log(path, topics: tuple[str, ...] = ("/imu", "/lidar", "/gps"), n: int = 100) -> None:
    with open(path, "wb") as fh:
        writer = Writer(fh)
        writer.start()
        schema_id = writer.register_schema(name="S", encoding="ros2msg", data=b"x")
        channels = {
            t: writer.register_channel(topic=t, message_encoding="cdr", schema_id=schema_id)
            for t in topics
        }
        for i in range(n):
            stamp = _T0 + i * _TICK
            for topic, channel in channels.items():
                writer.add_message(
                    channel_id=channel,
                    log_time=stamp,
                    publish_time=stamp,
                    sequence=i,
                    data=f"{topic}{i}".encode(),
                )
        writer.finish()


@pytest.fixture
def log(tmp_path):
    _write_log(tmp_path / "drive.mcap")
    return str(tmp_path)


def _col(name: str) -> dict:
    return {"e": "col", "name": name}


def _str(value: str) -> dict:
    return {"e": "lit", "value": {"str": value}}


def _time(nanos: int) -> dict:
    # The IR carries timestamp literals in microseconds.
    return {"e": "lit", "value": {"timestamp": nanos // 1000}}


def _cmp(op: str, left: dict, right: dict) -> dict:
    return {"e": "binary", "op": op, "left": left, "right": right}


def _table(src: MCAPSource, **kw) -> pa.Table:
    # The schema is passed explicitly so a read that matches nothing still yields a
    # (correctly-typed) empty table rather than failing to construct one.
    return pa.Table.from_batches(list(src.iter_batches(**kw)), schema=src.schema())


# ---- the relation ------------------------------------------------------------


def test_a_row_is_a_message(log) -> None:
    table = _table(MCAPSource(log))

    assert table.num_rows == 300  # 3 topics x 100 messages
    assert table.schema.names == [
        "topic",
        "log_time",
        "publish_time",
        "sequence",
        "schema_name",
        "message_encoding",
        "data",
    ]


def test_payloads_use_64_bit_offsets(log) -> None:
    """A LiDAR sweep or camera frame is megabytes; a batch of them passes 2 GB."""
    assert MCAPSource(log).schema().field("data").type == pa.large_binary()


def test_topics_are_listed_from_the_summary(log) -> None:
    assert MCAPSource(log).topics() == ["/gps", "/imu", "/lidar"]


# ---- index-backed metadata ---------------------------------------------------


def test_row_count_comes_from_the_index(log) -> None:
    stats = MCAPSource(log).statistics()

    assert stats.row_count == 300
    assert stats.exact_rows


def test_the_log_time_zone_map_is_exact(log) -> None:
    """Recorded first/last message times, not bounds over a chunk — so a time-sliced
    query prunes files precisely rather than conservatively."""
    stat = MCAPSource(log).statistics().columns["log_time"]

    assert stat.provenance is Provenance.EXACT
    assert stat.min.as_py() == pa.scalar(_T0, pa.timestamp("ns")).as_py()
    assert stat.max.as_py() == pa.scalar(_T0 + 99 * _TICK, pa.timestamp("ns")).as_py()


def test_row_count_reflects_an_explicit_topic_restriction(log) -> None:
    assert MCAPSource(log, topics=["/imu"]).statistics().row_count == 100


# ---- pushdown soundness ------------------------------------------------------


def _assert_pushdown_matches(src: MCAPSource, predicate: dict, mask) -> None:
    """Pushing a predicate must equal reading everything and filtering after."""
    full = _table(src)
    pushed = _table(src, predicate=predicate)
    expected = full.filter(mask)
    order = [("topic", "ascending"), ("log_time", "ascending")]

    assert pushed.num_rows == expected.num_rows
    assert pushed.sort_by(order).equals(expected.sort_by(order))


def test_topic_equality_pushes_down(log) -> None:
    src = MCAPSource(log)
    _assert_pushdown_matches(
        src,
        _cmp("eq", _col("topic"), _str("/imu")),
        pc.equal(_table(src)["topic"], "/imu"),
    )


@pytest.mark.parametrize(
    ("op", "compare"),
    [
        ("ge", pc.greater_equal),
        ("gt", pc.greater),
        ("le", pc.less_equal),
        ("lt", pc.less),
    ],
)
def test_time_bounds_push_down_on_the_right_side_of_the_boundary(log, op, compare) -> None:
    """`start_time` is inclusive and `end_time` exclusive, so strict and non-strict
    bounds differ by one nanosecond. Getting it wrong loses a boundary message."""
    src = MCAPSource(log)
    cut = _T0 + 50 * _TICK
    _assert_pushdown_matches(
        src,
        _cmp(op, _col("log_time"), _time(cut)),
        compare(_table(src)["log_time"], pa.scalar(cut, pa.timestamp("ns"))),
    )


def test_a_conjunction_pushes_both_terms(log) -> None:
    src = MCAPSource(log)
    cut = _T0 + 50 * _TICK
    full = _table(src)
    predicate = {
        "e": "binary",
        "op": "and",
        "left": _cmp("eq", _col("topic"), _str("/imu")),
        "right": _cmp("ge", _col("log_time"), _time(cut)),
    }
    _assert_pushdown_matches(
        src,
        predicate,
        pc.and_(
            pc.equal(full["topic"], "/imu"),
            pc.greater_equal(full["log_time"], pa.scalar(cut, pa.timestamp("ns"))),
        ),
    )


def test_a_disjunction_is_not_mined(log) -> None:
    """`topic = '/imu' OR log_time > t` must not push the topic term — that would drop
    rows the predicate keeps. Ignoring a term is always safe; narrowing wrongly is not."""
    src = MCAPSource(log)
    predicate = {
        "e": "binary",
        "op": "or",
        "left": _cmp("eq", _col("topic"), _str("/imu")),
        "right": _cmp("ge", _col("log_time"), _time(_T0 + 50 * _TICK)),
    }

    assert _table(src, predicate=predicate).num_rows == 300


def test_an_explicit_topic_restriction_is_never_widened_by_a_predicate(log) -> None:
    src = MCAPSource(log, topics=["/imu"])
    predicate = _cmp("eq", _col("topic"), _str("/lidar"))

    # The two restrictions intersect to nothing; the source must not read `/lidar`.
    assert _table(src, predicate=predicate).num_rows == 0


def test_read_and_iter_batches_agree(log) -> None:
    src = MCAPSource(log)
    assert pa.Table.from_batches(src.read()).equals(_table(src))


# ---- the multi-sensor workflow ----------------------------------------------


def test_sensors_at_different_rates_align_with_an_asof_join(tmp_path) -> None:
    """The step every perception and ADAS pipeline starts with: put the 100 Hz IMU onto
    each 10 Hz LiDAR sweep."""
    import batcher as bt
    from batcher import col

    path = tmp_path / "drive.mcap"
    with open(path, "wb") as fh:
        writer = Writer(fh)
        writer.start()
        schema_id = writer.register_schema(name="S", encoding="ros2msg", data=b"x")
        imu = writer.register_channel(topic="/imu", message_encoding="cdr", schema_id=schema_id)
        lidar = writer.register_channel(topic="/lidar", message_encoding="cdr", schema_id=schema_id)
        for i in range(1000):  # 100 Hz
            stamp = _T0 + i * 10_000_000
            writer.add_message(
                channel_id=imu, log_time=stamp, publish_time=stamp, sequence=i, data=b"i"
            )
        for i in range(100):  # 10 Hz, offset so no timestamp coincides exactly
            stamp = _T0 + i * 100_000_000 + 3_000_000
            writer.add_message(
                channel_id=lidar, log_time=stamp, publish_time=stamp, sequence=i, data=b"l"
            )
        writer.finish()

    log = bt.read.mcap(str(tmp_path))
    imu_ds = log.filter(col("topic") == "/imu").select("log_time", imu_seq=col("sequence"))
    lidar_ds = log.filter(col("topic") == "/lidar").select("log_time", lidar_seq=col("sequence"))
    aligned = lidar_ds.join_asof(imu_ds, on="log_time").to_pydict()

    assert len(aligned["lidar_seq"]) == 100, "one row per sweep"
    assert all(v is not None for v in aligned["imu_seq"]), "a sweep matched no IMU sample"
    # Sweep i sits 3 ms after IMU sample 10*i, which is the most recent one at or before it.
    pairs = dict(zip(aligned["lidar_seq"], aligned["imu_seq"], strict=True))
    assert [pairs[i] for i in range(5)] == [0, 10, 20, 30, 40]


def test_an_unreadable_log_can_be_skipped(tmp_path) -> None:
    """One truncated recording in a drive-day directory must not cost the rest."""
    _write_log(tmp_path / "good.mcap", n=10)
    (tmp_path / "zbad.mcap").write_bytes(b"not an mcap file")

    with pytest.raises(Exception):  # noqa: B017 — the backend's own decode error
        list(MCAPSource(str(tmp_path)).iter_batches())

    src = MCAPSource(str(tmp_path), on_error="skip")
    assert _table(src).num_rows == 30
    assert [p.rsplit("/", 1)[-1] for p in src.corrupt_files()] == ["zbad.mcap"]


# ---- distributed and streaming equivalence -----------------------------------


def test_splits_cover_the_source_exactly_once(log) -> None:
    """A distributed read is its splits; if they do not cover the source, rows vanish."""
    src = MCAPSource(log)
    per_split = sum(sum(b.num_rows for b in s.read()) for s in src.splits())

    assert per_split == _table(src).num_rows


def test_a_topic_restriction_survives_the_split_round_trip(tmp_path) -> None:
    """Invariant #7, and the one this connector could most easily break.

    A worker receives only the *pickled split* — never the source object — and rebuilds a
    reader from it. If `topics` is not carried in `_reader_kwargs`, every worker rebuilds an
    unrestricted reader, and the distributed run silently returns every topic while the
    single-node run returns one. Same query, different answers, no error.
    """
    import pickle

    for shard in range(3):
        _write_log(tmp_path / f"d{shard}.mcap", topics=("/imu", "/lidar"), n=50)
    src = MCAPSource(str(tmp_path), topics=["/imu"])

    shipped = [pickle.loads(pickle.dumps(s)) for s in src.splits()]
    rows = sum(sum(b.num_rows for b in s.read()) for s in shipped)

    assert rows == 150, "the topic restriction was lost on the way to the worker"
    assert rows == _table(src).num_rows


def test_streaming_and_materializing_executors_agree(log) -> None:
    """The engine's two execution paths must produce the same relation."""
    import dataclasses

    import batcher as bt
    from batcher.config import active_config, config_context

    ds = bt.read.mcap(log).select("topic", "log_time", "sequence")
    streamed = ds.collect()
    cfg = active_config()
    with config_context(cfg.replace(execution=dataclasses.replace(cfg.execution, streaming=False))):
        materialized = ds.collect()

    assert streamed.equals(materialized)


def test_a_pushed_predicate_agrees_with_the_engine_filter(log) -> None:
    """End to end: the source-level seek must return what the engine's own `Filter` would.

    This is the property that makes pushdown an optimization rather than a semantic
    change, and it is checked through the public API rather than against the reader.
    """
    import batcher as bt
    from batcher import col

    ds = bt.read.mcap(log)
    pushed = ds.filter(col("topic") == "/imu").count()
    everything = ds.collect().num_rows

    assert pushed == 100
    assert everything == 300


def test_the_handle_fallback_batches_like_the_streaming_route(tmp_path) -> None:
    """`_read_file` shares the streaming route's message loop rather than copying it.

    The copy it used to carry took every message of the file into a single batch. A drive
    log is millions of messages with megabyte payloads, so that is an unbounded
    materialization of the whole recording — and it read differently from the route every
    real query takes, which is how the two drift apart unnoticed.
    """
    from batcher.io.formats.robotics.mcap import _MESSAGES_PER_BATCH

    path = tmp_path / "many.mcap"
    n = _MESSAGES_PER_BATCH + 17
    _write_log(path, topics=("/imu",), n=n)

    src = MCAPSource(str(path))
    with open(path, "rb") as fh:
        batches = src._read_file(fh, None)

    assert sum(b.num_rows for b in batches) == n
    assert len(batches) > 1, "the whole file came back as one batch"
    assert max(b.num_rows for b in batches) <= _MESSAGES_PER_BATCH


def test_both_read_routes_agree_that_an_empty_topic_restriction_selects_nothing(
    tmp_path,
) -> None:
    """`topics=[]` means *no* topics, on whichever route runs.

    A pin rather than a regression test: both routes already answer this correctly, the
    streaming one through its own guard and the handle one because the mcap reader
    happens to treat an empty list as a restriction rather than as "unrestricted". The
    second of those is a property of a third-party library, which is exactly the kind of
    thing worth holding still — the guard reads as if it were load-bearing on both.
    """
    path = tmp_path / "empty_restriction.mcap"
    _write_log(path, topics=("/imu", "/lidar"), n=10)

    src = MCAPSource(str(path), topics=[])
    with open(path, "rb") as fh:
        batches = src._read_file(fh, None)
    assert sum(b.num_rows for b in batches) == 0

    # And the streaming route agrees, which is the property that was broken.
    assert sum(b.num_rows for b in src.iter_batches()) == 0
