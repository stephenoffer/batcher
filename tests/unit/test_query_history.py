"""`query_history()` — the queries a deployment ran, as a relation rather than a directory.

The engine already wrote a structured document per query. What was missing was the shape
an operator's questions come in: "which queries spilled last night", "what is the p95 per
pipeline shape", "did this get slower after the release" are all relational, and none of
them can be asked of a directory of JSON files. Snowflake answers them with
`QUERY_HISTORY` and Databricks with `system.query.history`.

Two properties are load-bearing and neither is obvious from reading the happy path:

* **The schema does not depend on whether anything has run.** Inference over an empty
  history types every column `null`, so `filter(col("total_elapsed_ms") > 1000)` — the
  first query any dashboard runs — failed on a fresh deployment. The types are declared.
* **The measurements are exposed and the plan is not.** An event-log document carries the
  whole plan including literal predicate constants, which is why the engine writes it into
  an owner-only directory. A history table that republished those would move every
  `WHERE ssn = '...'` into whatever a dashboard can reach.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.history import query_history

pytestmark = pytest.mark.unit


def _document(tmp_path, name: str, **overrides) -> str:
    """Write one event-log-shaped document and return the directory holding it."""
    document = {
        "query_id": name,
        "total_ms": 12.5,
        "rows": 7,
        "distributed": False,
        "measured": True,
        "spilled": False,
        "total_spill_bytes": 0,
        "peak_rss_bytes": 1024,
        "memory_budget_bytes": 2048,
        "cpu_utilization": 0.5,
        "carbonite_summary": "feasible",
        "machine": "test-machine",
        "usage": {"cpu_ms": 3.0, "wall_ms": 12.0, "cores_busy": 0.25},
        "ops": [{"op_id": 0}, {"op_id": 1}],
        "logical_ir": {"op": "filter", "predicate": "ssn = '123-45-6789'"},
    }
    document.update(overrides)
    (tmp_path / f"{name}.json").write_text(json.dumps(document), encoding="utf-8")
    return str(tmp_path)


class TestShape:
    """One row per query, with the measurements the engine took."""

    def test_a_document_becomes_a_row(self, tmp_path):
        history = query_history(_document(tmp_path, "20260101-000000-1-000000"))
        rows = history.to_pydict()
        assert rows["query_id"] == ["20260101-000000-1-000000"]
        assert rows["total_elapsed_ms"] == [12.5]
        assert rows["rows_produced"] == [7]
        assert rows["operator_count"] == [2]

    def test_the_nested_usage_block_is_flattened(self, tmp_path):
        history = query_history(_document(tmp_path, "20260101-000000-1-000000"))
        rows = history.to_pydict()
        assert rows["cpu_ms"] == [3.0]
        assert rows["cores_busy"] == [0.25]

    def test_most_recent_first(self, tmp_path):
        for name in ("20260101-000001-1-000000", "20260101-000003-1-000000"):
            _document(tmp_path, name)
        directory = _document(tmp_path, "20260101-000002-1-000000")
        got = query_history(directory).to_pydict()["query_id"]
        assert got == sorted(got, reverse=True)

    def test_limit_takes_the_most_recent(self, tmp_path):
        for i in range(5):
            directory = _document(tmp_path, f"20260101-00000{i}-1-000000")
        got = query_history(directory, limit=2).to_pydict()["query_id"]
        assert got == ["20260101-000004-1-000000", "20260101-000003-1-000000"]

    def test_the_profile_path_names_the_document(self, tmp_path):
        directory = _document(tmp_path, "20260101-000000-1-000000")
        path = query_history(directory).to_pydict()["profile_path"][0]
        assert os.path.isfile(path)
        assert json.loads(Path(path).read_text(encoding="utf-8"))["query_id"]


class TestEmptyHistoryIsUsable:
    """A fresh deployment must answer a dashboard query with no rows, not an error."""

    def test_the_schema_is_the_same_with_no_documents(self, tmp_path):
        empty = query_history(str(tmp_path))
        (tmp_path / "sub").mkdir()
        populated = query_history(_document(tmp_path / "sub", "20260101-000000-1-000000"))
        assert empty.count() == 0
        assert populated.count() == 1
        assert empty.columns == populated.columns
        assert empty.schema == populated.schema

    def test_no_column_is_typed_null(self, tmp_path):
        """Inference over nothing types everything `null`, and every comparison against a
        `null` column then fails. Declaring the types is what this asserts."""
        schema = query_history(str(tmp_path)).schema
        assert [f.name for f in schema if f.type == "null"] == []

    def test_a_numeric_filter_plans_and_runs_on_an_empty_history(self, tmp_path):
        history = query_history(str(tmp_path))
        assert history.filter(bt.col("total_elapsed_ms") > 1000.0).count() == 0

    def test_an_aggregate_over_an_empty_history_runs(self, tmp_path):
        history = query_history(str(tmp_path))
        assert history.agg(n=bt.col("rows_produced").sum()).to_pydict()["n"] == [None]

    def test_a_missing_directory_is_no_history_not_an_error(self, tmp_path):
        assert query_history(str(tmp_path / "never-created")).count() == 0


class TestItDoesNotRepublishThePlan:
    """The document carries literal predicate constants. The table must not."""

    def test_no_column_carries_the_plan(self, tmp_path):
        directory = _document(tmp_path, "20260101-000000-1-000000")
        history = query_history(directory)
        assert "logical_ir" not in history.columns
        assert "optimized_ir" not in history.columns

    def test_a_literal_in_the_plan_is_not_in_any_value(self, tmp_path):
        """The positive control for the assertion above: the secret really is in the
        document, so a column that leaked it would have something to leak."""
        directory = _document(tmp_path, "20260101-000000-1-000000")
        raw = (tmp_path / "20260101-000000-1-000000.json").read_text(encoding="utf-8")
        assert "123-45-6789" in raw
        rendered = str(query_history(directory).to_pydict())
        assert "123-45-6789" not in rendered


class TestDamagedInput:
    """The engine prunes this directory while writing it, so partial reads are normal."""

    def test_an_unreadable_document_is_skipped_not_fatal(self, tmp_path):
        directory = _document(tmp_path, "20260101-000000-1-000000")
        (tmp_path / "20260101-000001-1-000000.json").write_text("{ truncated", encoding="utf-8")
        assert query_history(directory).count() == 1

    def test_a_non_object_document_is_skipped(self, tmp_path):
        directory = _document(tmp_path, "20260101-000000-1-000000")
        (tmp_path / "20260101-000001-1-000000.json").write_text("[1, 2]", encoding="utf-8")
        assert query_history(directory).count() == 1

    def test_a_non_json_file_is_ignored(self, tmp_path):
        directory = _document(tmp_path, "20260101-000000-1-000000")
        (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
        assert query_history(directory).count() == 1

    def test_a_document_missing_keys_still_becomes_a_row(self, tmp_path):
        """An older build's document is still a query that ran. Dropping it would shorten
        the history at exactly the moment an operator compares across an upgrade."""
        (tmp_path / "20260101-000000-1-000000.json").write_text(
            json.dumps({"query_id": "old", "total_ms": 1.0}), encoding="utf-8"
        )
        rows = query_history(str(tmp_path)).to_pydict()
        assert rows["query_id"] == ["old"]
        assert rows["rows_produced"] == [None]


class TestArgumentChecks:
    """A typo'd path returning an empty history reads as "nothing ran"."""

    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "10"])
    def test_a_bad_limit_is_refused(self, bad, tmp_path):
        with pytest.raises(PlanError, match="positive integer"):
            query_history(str(tmp_path), limit=bad)

    def test_a_file_where_a_directory_belongs_is_refused(self, tmp_path):
        target = tmp_path / "one.json"
        target.write_text("{}", encoding="utf-8")
        with pytest.raises(PlanError, match="not a directory"):
            query_history(str(target))

    def test_an_empty_path_is_refused(self):
        with pytest.raises(PlanError, match="non-empty string"):
            query_history("")


class TestTheDefaultLocation:
    """With no path it reads where the engine actually writes.

    This is the test that makes the rest mean anything: everything above feeds the reader
    documents a test wrote, and would keep passing if the reader and the writer had drifted
    onto different directories or different key names.
    """

    def test_a_query_this_process_ran_is_recorded(self, tmp_path, monkeypatch):
        import dataclasses

        from batcher.config import active_config, set_config

        monkeypatch.setenv("BATCHER_HOME", str(tmp_path))
        previous = active_config()
        # The suite disables the event log (tests/conftest.py); this is one of the tests
        # that has to turn it back on, against a temporary home.
        set_config(
            previous.replace(
                observability=dataclasses.replace(previous.observability, event_log=True)
            )
        )
        try:
            # A shape that actually executes. A keyless aggregate over in-memory data is
            # answered from metadata without reaching the executor, so it writes no
            # document -- and a test built on one would assert the reader against an empty
            # directory while looking like it had run a query.
            bt.from_pydict({"a": [1, 2, 3], "b": ["x", "y", "x"]}).group_by("b").agg(
                s=bt.col("a").sum()
            ).collect()
        finally:
            set_config(previous)
        history = query_history()
        assert history.count() >= 1
        rows = history.to_pydict()
        assert all(p.startswith(str(tmp_path)) for p in rows["profile_path"])
        # The writer's keys reach the reader's columns: a measured query has a real
        # elapsed time and a real operator count, not the None a key mismatch would give.
        assert all(v is not None and v > 0 for v in rows["total_elapsed_ms"])
        assert all(v is not None and v > 0 for v in rows["operator_count"])


class TestTheReaderAndTheWriterAgree:
    """The default location is stated twice. These hold the two statements equal.

    `query_history` cannot call `event_log._resolve_dir`, because that resolution
    `mkdir`s as part of resolving: a reader that creates a directory to answer "has
    anything run?" is wrong on its own terms, and against a configured directory the
    process may not create it raises instead of answering. So the convention is written
    out a second time, and the cost of that is exactly this class.
    """

    @staticmethod
    def _writers_answer() -> str:
        from batcher.api.terminal.event_log import _resolve_dir
        from batcher.config import active_config

        return str(_resolve_dir(active_config().observability.event_log_dir))

    def test_the_two_resolutions_match_under_batcher_home(self, tmp_path, monkeypatch):
        from batcher.api.history import _default_directory

        monkeypatch.setenv("BATCHER_HOME", str(tmp_path))
        assert _default_directory() == self._writers_answer()

    def test_the_two_resolutions_match_with_a_configured_directory(self, tmp_path, monkeypatch):
        import dataclasses

        from batcher.api.history import _default_directory
        from batcher.config import active_config, set_config

        monkeypatch.delenv("BATCHER_HOME", raising=False)
        previous = active_config()
        target = str(tmp_path / "events")
        set_config(
            previous.replace(
                observability=dataclasses.replace(previous.observability, event_log_dir=target)
            )
        )
        try:
            assert _default_directory() == self._writers_answer()
        finally:
            set_config(previous)

    def test_reading_does_not_create_the_directory(self, tmp_path, monkeypatch):
        """The bug this exists to stop: `query_history()` used to `mkdir` its way to an
        answer, which fails outright when the configured directory is somewhere the
        process cannot write."""
        monkeypatch.setenv("BATCHER_HOME", str(tmp_path / "never-created"))
        assert query_history().count() == 0
        assert not (tmp_path / "never-created").exists()
