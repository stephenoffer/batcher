"""Delta time travel by timestamp, bounded change-feed windows, and table properties.

Three gaps these cover, each of which made a documented Spark/Delta capability
unreachable from Batcher:

* **Time travel accepted only one spelling of a timestamp.** delta-rs wants a
  fully-qualified RFC-3339 instant, so ``"2026-08-05"`` and ``datetime.now().isoformat()``
  — the two forms a person actually writes, and both of which Spark's ``timestampAsOf``
  takes — failed with ``Failed to parse datetime string: premature end of input``.
* **The change feed could only be read as an unbounded stream.** An incremental ETL step
  wants a *closed* window ("every change between the watermark and now") so it can join
  and merge it; an unbounded source cannot be collected, counted, or joined at all.
* **Table properties could not be set.** Which meant a Batcher-created table could never
  have ``delta.enableChangeDataFeed`` turned on — so `read_change_feed` could not read a
  table Batcher wrote, only one Spark had.
"""

from __future__ import annotations

import datetime as dt
import time

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import BackendError

deltalake = pytest.importorskip("deltalake")
pytestmark = pytest.mark.integration

_CDF = {"delta.enableChangeDataFeed": "true"}

#: Seconds between the fixture's commits. Wide enough that the midpoint between two of
#: them is unambiguously after the earlier commit and before the later one, whichever of
#: the two clocks (`commitInfo` or file mtime) delta-rs resolves against.
_COMMIT_GAP = 0.05


def _config(uri: str) -> dict[str, str]:
    """The table's properties as the log records them."""
    return deltalake.DeltaTable(uri).metadata().configuration


def _rows(ids: list[int]) -> pa.Table:
    return pa.table({"id": pa.array(ids, pa.int64()), "v": [f"r{i}" for i in ids]})


def _table(tmp_path, name: str = "t") -> str:
    """A three-commit table with the change feed on, written entirely through Batcher.

    The commits are deliberately spaced. delta-rs resolves a time-travel timestamp against
    each commit file's *modification time*, which lands a few milliseconds after the
    `commitInfo` timestamp `history()` reports — so an instant computed from `history()` and
    only a millisecond before a commit still resolves **to** that commit. Three writes this
    small otherwise land close enough together for that skew to swallow the gap between
    them, which failed the time-travel assertion about one run in two.
    """
    uri = str(tmp_path / name)
    bt.from_arrow(_rows([1, 2])).write.delta(uri, mode="overwrite", table_properties=_CDF)
    time.sleep(_COMMIT_GAP)
    bt.from_arrow(_rows([3])).write.delta(uri, mode="append")
    time.sleep(_COMMIT_GAP)
    bt.from_arrow(_rows([4])).write.delta(uri, mode="append")
    return uri


# --- table properties ------------------------------------------------------


def test_table_properties_are_set_when_the_write_creates_the_table(tmp_path):
    uri = str(tmp_path / "props")
    bt.from_arrow(_rows([1])).write.delta(uri, mode="overwrite", table_properties=_CDF)
    assert _config(uri)["delta.enableChangeDataFeed"] == "true"


def test_table_properties_are_altered_onto_an_existing_table(tmp_path):
    uri = str(tmp_path / "props")
    bt.from_arrow(_rows([1])).write.delta(uri, mode="overwrite")
    assert "delta.enableChangeDataFeed" not in _config(uri)
    bt.from_arrow(_rows([2])).write.delta(uri, mode="append", table_properties=_CDF)
    assert _config(uri)["delta.enableChangeDataFeed"] == "true"


def test_setting_a_property_already_in_force_commits_nothing(tmp_path):
    """A pipeline passes its properties on every run; that must not grow the log."""
    uri = str(tmp_path / "props")
    bt.from_arrow(_rows([1])).write.delta(uri, mode="overwrite", table_properties=_CDF)
    before = deltalake.DeltaTable(uri).version()
    bt.from_arrow(_rows([2])).write.delta(uri, mode="append", table_properties=_CDF)
    # Exactly one version for the append itself — no extra metaData commit beside it.
    assert deltalake.DeltaTable(uri).version() == before + 1


# --- time travel by timestamp ----------------------------------------------


def _after_last_commit(uri: str) -> dt.datetime:
    """A naive local `datetime` a few seconds after the table's newest commit."""
    latest_ms = max(h["timestamp"] for h in deltalake.DeltaTable(uri).history(50))
    return dt.datetime.fromtimestamp(latest_ms / 1000 + 5)


@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param(lambda m: m.isoformat(), id="naive-isoformat"),
        pytest.param(lambda m: m.isoformat(timespec="seconds"), id="naive-seconds"),
        pytest.param(lambda m: m.strftime("%Y-%m-%d %H:%M:%S"), id="space-separator"),
        pytest.param(lambda m: (m + dt.timedelta(days=1)).date().isoformat(), id="date-only"),
        pytest.param(lambda m: m, id="naive-datetime-object"),
        pytest.param(lambda m: m.astimezone(dt.UTC), id="aware-datetime-object"),
        pytest.param(lambda m: m.astimezone(dt.UTC).isoformat(), id="offset-string"),
        pytest.param(
            lambda m: m.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), id="zulu-string"
        ),
    ],
)
def test_time_travel_accepts_every_reasonable_timestamp_spelling(tmp_path, spelling):
    uri = _table(tmp_path)
    moment = spelling(_after_last_commit(uri))
    assert bt.read.delta(uri, timestamp=moment).count() == 4


def test_time_travel_by_timestamp_lands_on_the_version_current_then(tmp_path):
    """The point of time travel: an earlier instant must return the earlier table."""
    uri = _table(tmp_path)
    history = {h["version"]: h["timestamp"] for h in deltalake.DeltaTable(uri).history(50)}
    # Midway between commits 1 and 2, when the table held three rows. Midway rather than a
    # fixed offset from either: see `_table` on the skew between the two commit clocks.
    between = dt.datetime.fromtimestamp((history[1] + history[2]) / 2000, tz=dt.UTC)
    assert bt.read.delta(uri, timestamp=between).count() == 3


@pytest.mark.parametrize("bad", ["not-a-date", "2026/08/05", "", 42])
def test_an_unparseable_timestamp_names_the_argument_and_the_accepted_forms(tmp_path, bad):
    uri = _table(tmp_path)
    with pytest.raises(BackendError) as excinfo:
        bt.read.delta(uri, timestamp=bad).count()
    message = str(excinfo.value)
    assert "timestamp" in message
    assert "YYYY-MM-DD" in message
    # Not reported as a failure to reach the table — it is the caller's argument.
    assert "failed to open Delta table" not in message


# --- bounded change-feed windows -------------------------------------------


def test_a_bound_makes_the_change_feed_a_bounded_relation(tmp_path):
    uri = _table(tmp_path)
    feed = bt.read.read_change_feed(uri, starting_version=0, ending_version=2)
    assert feed.is_streaming is False
    assert feed.count() == 4
    assert set(feed.columns) >= {"id", "v", "_change_type", "_commit_version"}


def test_no_bound_still_yields_the_unbounded_stream(tmp_path):
    uri = _table(tmp_path)
    assert bt.read.read_change_feed(uri, starting_version=0).is_streaming is True


def test_the_window_selects_exactly_its_versions(tmp_path):
    uri = _table(tmp_path)
    got = (
        bt.read.read_change_feed(uri, starting_version=2, ending_version=2)
        .select("id", "_commit_version")
        .collect()
        .to_pydict()
    )
    assert got == {"id": [4], "_commit_version": [2]}


def test_a_bounded_window_composes_with_the_relational_api(tmp_path):
    """The reason the bound exists: an unbounded source can be joined to nothing."""
    uri = _table(tmp_path)
    dim = bt.from_pydict({"id": [3, 4], "label": ["three", "four"]})
    got = (
        bt.read.read_change_feed(uri, starting_version=1, ending_version=2)
        .join(dim, on="id")
        .group_by("label")
        .agg(n=bt.col("id").count())
        .sort("label")
        .collect()
        .to_pydict()
    )
    assert got == {"label": ["four", "three"], "n": [1, 1]}


def test_an_open_ended_window_runs_to_the_latest_commit(tmp_path):
    """`ending_timestamp` alone bounds the read without pinning the far end to a version."""
    uri = _table(tmp_path)
    feed = bt.read.read_change_feed(uri, ending_timestamp=_after_last_commit(uri))
    assert feed.is_streaming is False
    assert feed.count() == 4


def test_two_windows_of_one_table_are_two_different_sources(tmp_path):
    """A cached scan keyed on the table alone would serve one window's rows for the other."""
    uri = _table(tmp_path)
    first = bt.read.read_change_feed(uri, starting_version=0, ending_version=0)
    rest = bt.read.read_change_feed(uri, starting_version=1, ending_version=2)
    assert sorted(first.select("id").collect().to_pydict()["id"]) == [1, 2]
    assert sorted(rest.select("id").collect().to_pydict()["id"]) == [3, 4]


def test_a_bad_window_timestamp_names_the_argument_it_came_from(tmp_path):
    uri = _table(tmp_path)
    with pytest.raises(BackendError, match="ending_timestamp"):
        bt.read.read_change_feed(uri, ending_timestamp="nope").count()


def test_the_window_is_planned_against_a_real_cardinality(tmp_path):
    """A source that reports nothing is planned at ~1e12 rows, and that is not survivable.

    With no estimate, this exact join estimated 97 billion rows, picked its build side
    against that figure, and was refused admission by the memory envelope — for a change
    feed of four rows. The log knows the answer well enough to steer the plan, so it says so.
    """
    uri = _table(tmp_path)
    feed = bt.read.read_change_feed(uri, starting_version=0, ending_version=2)
    plan = feed.join(bt.from_pydict({"id": [3], "label": ["three"]}), on="id").explain()
    scan = next(line for line in plan.splitlines() if "scan" in line)
    estimate = int(scan.split("est≈")[1].split()[0].replace(",", ""))
    # Within an order of magnitude of the four rows actually there — the point is only that
    # it is a cardinality and not the unknown-rows placeholder.
    assert estimate < 1000, scan
    assert "sketch" in scan, "the estimate must be marked estimated, never exact"


def test_an_estimate_never_answers_a_count(tmp_path):
    """`exact_rows=False`: the estimate may steer a plan, never substitute for the rows."""
    uri = _table(tmp_path)
    feed = bt.read.read_change_feed(uri, starting_version=2, ending_version=2)
    assert feed.count() == 1  # the one row commit 2 added, not the log-derived estimate


def test_updates_and_deletes_appear_with_their_change_types(tmp_path):
    """A change feed that only ever reports inserts is not a change feed."""
    uri = _table(tmp_path)
    deltalake.DeltaTable(uri).delete("id = 1")
    latest = deltalake.DeltaTable(uri).version()
    kinds = set(
        bt.read.read_change_feed(uri, starting_version=latest, ending_version=latest)
        .select("_change_type")
        .collect()
        .to_pydict()["_change_type"]
    )
    assert kinds == {"delete"}
