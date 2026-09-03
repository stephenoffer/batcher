"""The durable audit sink: a governance decision that outlives the process that made it.

`GovernanceConfig.audit_path` was a configured field nothing wrote, so the only durable
trace of an authorization decision was a log line. These tests pin the three properties a
compliance reviewer actually depends on -- the record lands, it appends rather than
replaces, and a sink that cannot write stops the read instead of quietly passing it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import stat
from pathlib import Path

import pytest

import batcher as bt
from batcher.config import active_config, set_config
from batcher.governance import GovernanceEvent
from batcher.governance.audit_log import record_governance_event

pytestmark = pytest.mark.unit


@pytest.fixture
def audit_to():
    """Point `governance.audit_path` at a file for one test, restoring the config after."""
    original = active_config()

    def apply(path: Path | None) -> None:
        current = active_config()
        set_config(
            current.replace(
                governance=dataclasses.replace(
                    current.governance, audit_path=None if path is None else str(path)
                )
            )
        )

    try:
        yield apply
    finally:
        set_config(original)


@pytest.fixture
def table(tmp_path: Path) -> str:
    path = str(tmp_path / "t.parquet")
    bt.from_pydict({"id": [1, 2, 3], "email": ["a@x", "b@x", "c@x"]}).write(path, format="parquet")
    return path


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestTheDefaultIsUnchanged:
    def test_no_path_configured_writes_nothing(self, tmp_path: Path) -> None:
        """The default is None, and every existing deployment must keep behaving as it did."""
        assert bt.GovernanceConfig().audit_path is None
        record_governance_event(_event(), None)
        assert list(tmp_path.iterdir()) == []


def _event(table: str = "/data/t.parquet") -> GovernanceEvent:
    return GovernanceEvent(
        principal="ana",
        roles=("analyst",),
        table=table,
        visible=("id",),
        denied=("ssn",),
        masked=("email",),
        row_filters=("region = 'EU'",),
    )


class TestTheRecord:
    def test_one_decision_is_one_json_line(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        record_governance_event(_event(), str(path))
        (row,) = _lines(path)
        assert row["principal"] == "ana"
        assert row["roles"] == ["analyst"]
        assert row["visible"] == ["id"]
        assert row["denied"] == ["ssn"]
        assert row["masked"] == ["email"]
        assert row["row_filters"] == ["region = 'EU'"]
        assert row["allowed"] is True
        assert row["at"].startswith("20")  # an ISO-8601 stamp, added by the sink

    def test_records_append_rather_than_replace(self, tmp_path: Path) -> None:
        """An audit file that truncates keeps only the most recent decision, which is no trail."""
        path = tmp_path / "audit.jsonl"
        record_governance_event(_event("/data/a.parquet"), str(path))
        record_governance_event(_event("/data/b.parquet"), str(path))
        assert [r["table"] for r in _lines(path)] == ["/data/a.parquet", "/data/b.parquet"]

    def test_a_denial_is_recorded_as_not_allowed(self, tmp_path: Path) -> None:
        """The event a security review most wants to find must be distinguishable in the file."""
        path = tmp_path / "audit.jsonl"
        denied = dataclasses.replace(_event(), visible=(), masked=())
        record_governance_event(denied, str(path))
        (row,) = _lines(path)
        assert row["allowed"] is False

    def test_the_file_is_owner_only_from_the_moment_it_exists(self, tmp_path: Path) -> None:
        """Who read what is worth protecting even though no value ever reaches the file."""
        path = tmp_path / "audit.jsonl"
        record_governance_event(_event(), str(path))
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600

    def test_no_column_value_reaches_the_file(self, tmp_path: Path) -> None:
        """The record names columns and policies. A value in an audit log is a leak."""
        path = tmp_path / "audit.jsonl"
        record_governance_event(_event(), str(path))
        assert "a@x" not in path.read_text()


class TestFailingClosed:
    def test_a_sink_that_cannot_write_raises(self, tmp_path: Path) -> None:
        """A full disk must not quietly turn a governed deployment into an ungoverned one."""
        unwritable = tmp_path / "no-such-dir" / "audit.jsonl"
        with pytest.raises(OSError):
            record_governance_event(_event(), str(unwritable))


class TestTheEmitterIsWired:
    def test_a_governed_read_lands_in_the_configured_file(
        self, table: str, tmp_path: Path, audit_to
    ) -> None:
        """The end-to-end proof: without this the sink is a module nothing calls."""
        path = tmp_path / "audit.jsonl"
        audit_to(path)
        catalog = bt.SecurityCatalog().grant("analyst", on=table, select=["id"])
        with bt.security(catalog, bt.Principal("ana", roles=["analyst"])):
            bt.read.parquet(table).collect()
        rows = _lines(path)
        assert rows, "a governed read wrote no audit record"
        assert rows[-1]["principal"] == "ana"
        assert rows[-1]["table"] == table
        assert "email" in rows[-1]["denied"]

    def test_an_ungoverned_read_writes_nothing(self, table: str, tmp_path: Path, audit_to) -> None:
        """A read no policy covers emits no decision, so the file must stay empty."""
        path = tmp_path / "audit.jsonl"
        audit_to(path)
        bt.read.parquet(table).collect()
        assert not path.exists()


class TestConcurrency:
    def test_concurrent_writers_do_not_interleave_a_record(self, tmp_path: Path) -> None:
        """Two queries auditing at once must not splice one JSON object into another.

        An interleaved write is not a lost record, it is a *corrupt* one: the line no longer
        parses, so a reviewer loses both decisions and the file's whole tail is suspect.
        """
        import concurrent.futures as cf

        path = tmp_path / "audit.jsonl"
        writers, per_writer = 8, 40
        # Distinct table names of very different lengths, so a splice shows up as a parse
        # failure rather than hiding inside equally-sized records.
        events = [_event("/data/" + "x" * (1 + i % 60) + f"/{i}.parquet") for i in range(writers)]

        def write(ev: GovernanceEvent) -> None:
            for _ in range(per_writer):
                record_governance_event(ev, str(path))

        with cf.ThreadPoolExecutor(max_workers=writers) as pool:
            list(pool.map(write, events))

        rows = _lines(path)  # raises JSONDecodeError on any spliced line
        assert len(rows) == writers * per_writer
        assert {r["table"] for r in rows} == {e.table for e in events}
