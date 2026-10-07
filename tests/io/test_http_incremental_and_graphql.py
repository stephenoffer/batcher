"""Resumable incremental API reads (`bt.io.Incremental`) and the GraphQL source.

The incremental tests run the same read twice against a server whose data changes in
between, and pin the two properties AP-462 asks for: a resumed read misses no record at a
page boundary, and it does not deliver a record version twice.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pyarrow as pa
import pytest

import batcher as bt
from _fake_http import FakeApi, FakeResponse
from batcher._internal.errors import BackendError, PlanError
from batcher.io.formats.http.graphql import GraphQLSource, raise_on_errors
from batcher.io.formats.http.source import HttpJsonSource
from batcher.io.formats.http.state import IncrementalRun, shift_back


class _Feed:
    """An API serving ``?since=`` filtered records, ordered by ``updated_at``, 2 per page."""

    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def __call__(self, req) -> FakeResponse:
        since = req.query.get("since")
        rows = sorted(
            (r for r in self.records if since is None or r["updated_at"] >= since),
            key=lambda r: (r["updated_at"], r["id"]),
        )
        page = int(req.query.get("page", 1))
        return FakeResponse(body={"items": rows[(page - 1) * 2 : page * 2]})


def _rec(i: int, minute: int) -> dict:
    return {"id": i, "updated_at": f"2024-01-01T00:{minute:02d}:00Z", "v": f"{i}@{minute}"}


def _source(api, inc) -> HttpJsonSource:
    return HttpJsonSource(
        api.url,
        pagination=bt.io.PagePagination(),
        records_path="items",
        schema=pa.schema([("id", pa.int64()), ("updated_at", pa.string()), ("v", pa.string())]),
        incremental=inc,
    )


def _values(src) -> list[str]:
    return sorted(v for b in src.read() for v in b.column("v").to_pylist())


def test_watermark_resume_loses_nothing_at_the_boundary_and_never_duplicates(tmp_path):
    feed = _Feed([_rec(1, 1), _rec(2, 2), _rec(3, 3), _rec(4, 3)])
    inc = bt.io.Incremental(
        state=str(tmp_path / "s.json"),
        cursor_field="updated_at",
        key="id",
        lookback=timedelta(minutes=1),
        param="since",
    )
    with FakeApi(feed) as api:
        assert _values(_source(api, inc)) == ["1@1", "2@2", "3@3", "4@3"]
        assert inc.load()["watermark"] == "2024-01-01T00:03:00Z"
        # Between runs: a late record lands *at* the watermark, one inside the lookback
        # window is updated, and a new one arrives after.
        feed.records += [_rec(5, 3), _rec(2, 3), _rec(6, 4)]
        second = _source(api, inc)
        got = _values(second)
        assert api.requests[-1].query["since"] == "2024-01-01T00:02:00Z"
    # 5@3 shares the old watermark and is not lost; 2@3 is a new version of id 2; the
    # versions already delivered (2@2, 3@3, 4@3) are not delivered again.
    assert got == ["2@3", "5@3", "6@4"]
    assert inc.load()["watermark"] == "2024-01-01T00:04:00Z"


def test_a_read_that_stops_part_way_leaves_the_state_alone(tmp_path):
    feed = _Feed([_rec(i, i) for i in range(1, 6)])
    inc = bt.io.Incremental(
        state=str(tmp_path / "s.json"), cursor_field="updated_at", key="id", param="since"
    )
    with FakeApi(feed) as api:
        it = _source(api, inc).iter_batches()
        next(it)  # one page consumed, then the consumer dies
        it.close()
        assert inc.load() is None and inc.pending() is None
        assert _values(_source(api, inc)) == [f"{i}@{i}" for i in range(1, 6)]


def test_manual_commit_stages_until_commit(tmp_path):
    feed = _Feed([_rec(1, 1), _rec(2, 2)])
    inc = bt.io.Incremental(
        state=str(tmp_path / "s.json"), cursor_field="updated_at", key="id", auto_commit=False
    )
    with FakeApi(feed) as api:
        _source(api, inc).read()
    assert inc.load() is None
    assert inc.pending()["watermark"] == "2024-01-01T00:02:00Z"
    assert inc.commit() is True
    assert inc.load()["watermark"] == "2024-01-01T00:02:00Z"
    assert not os.path.exists(str(tmp_path / "s.json.pending"))
    assert inc.commit() is False


def test_client_side_bound_filters_when_the_api_ignores_since(tmp_path):
    records = [_rec(1, 1), _rec(2, 2), _rec(3, 3)]
    inc = bt.io.Incremental(state=str(tmp_path / "s.json"), cursor_field="updated_at", key="id")
    with FakeApi(lambda req: FakeResponse(body={"items": records})) as api:
        src = HttpJsonSource(api.url, records_path="items", incremental=inc)
        assert len(_values(src)) == 3
        records.append(_rec(4, 4))
        # No param: the server returns everything; the client keeps only >= watermark,
        # and the boundary record 3@3 is a delivered version, so only 4@4 is new.
        assert _values(HttpJsonSource(api.url, records_path="items", incremental=inc)) == ["4@4"]


def test_cursor_resume_restarts_at_the_last_accepted_page(tmp_path):
    data = [{"id": i, "v": str(i)} for i in range(5)]

    def handler(req):
        start = int(req.query.get("after", 0))
        page = data[start : start + 2]
        nxt = start + 2 if start + 2 < len(data) else None
        return FakeResponse(body={"items": page, "next": nxt})

    inc = bt.io.Incremental(state=str(tmp_path / "c.json"), key="id")
    pagination = bt.io.CursorPagination(cursor_path="next", param="after")
    with FakeApi(handler) as api:
        first = HttpJsonSource(
            api.url, pagination=pagination, records_path="items", incremental=inc
        )
        assert sorted(i for b in first.read() for i in b.column("id").to_pylist()) == [
            0,
            1,
            2,
            3,
            4,
        ]
        assert inc.load()["cursor"] == 4
        data.append({"id": 5, "v": "5"})
        second = HttpJsonSource(
            api.url, pagination=pagination, records_path="items", incremental=inc
        )
        rows = [i for b in second.read() for i in b.column("id").to_pylist()]
        assert api.requests[-1].query["after"] == "4"
    # The last page is re-read (it may have grown); id 4 was delivered, id 5 is new.
    assert rows == [5]


def test_numeric_watermark_and_lookback(tmp_path):
    rows = [{"id": i, "seq": i} for i in range(1, 6)]
    inc = bt.io.Incremental(
        state=str(tmp_path / "n.json"), cursor_field="seq", key="id", lookback=2
    )
    with FakeApi(lambda req: FakeResponse(body=rows)) as api:
        assert HttpJsonSource(api.url, incremental=inc).read()[0].num_rows == 5
        state = inc.load()
        assert state["watermark"] == 5
        # Only the tokens inside the next lookback window (seq >= 3) are kept.
        assert sorted(int(t.split("\x1f")[0]) for t, _ in state["seen"]) == [3, 4, 5]
        rows.append({"id": 6, "seq": 6})
        again = HttpJsonSource(api.url, incremental=inc).read()
        assert [r for b in again for r in b.column("id").to_pylist()] == [6]


def test_incremental_validation():
    with pytest.raises(PlanError):
        bt.io.Incremental(state="")
    with pytest.raises(PlanError):
        bt.io.Incremental(state="x", lookback=1)
    with pytest.raises(PlanError, match="key="):
        HttpJsonSource(
            "http://x",
            pagination=bt.io.OffsetPagination(limit=5),
            incremental=bt.io.Incremental(state="x"),
        )
    assert (
        shift_back("2024-01-01T00:00:00+00:00", timedelta(hours=1)) == "2023-12-31T23:00:00+00:00"
    )


def test_run_without_state_uses_start(tmp_path):
    run = IncrementalRun(
        bt.io.Incremental(
            state=str(tmp_path / "x.json"), cursor_field="t", start="2024-01-01T00:00:00Z"
        )
    )
    assert run.lower_bound == "2024-01-01T00:00:00Z"


# ---- GraphQL --------------------------------------------------------------------------
QUERY = (
    "query($after: String, $owner: String) { repo(owner: $owner) { issues(after: $after) "
    "{ nodes { id title } pageInfo { endCursor hasNextPage } } } }"
)


def _graphql_handler(pages: list[dict]):
    def handler(req):
        body = req.json()
        assert body["query"] == QUERY and body["variables"]["owner"] == "o"
        index = int(body["variables"].get("after") or 0)
        return FakeResponse(body=pages[index])

    return handler


def _page(nodes: list[dict], end: str | None, more: bool) -> dict:
    return {
        "data": {
            "repo": {
                "issues": {"nodes": nodes, "pageInfo": {"endCursor": end, "hasNextPage": more}}
            }
        }
    }


def test_graphql_pages_through_relay_page_info():
    pages = [
        _page([{"id": "a", "title": "x"}], "1", True),
        _page([{"id": "b", "title": "y"}], "2", True),
        _page([{"id": "c", "title": "z"}], None, False),
    ]
    with FakeApi(_graphql_handler(pages)) as api:
        ds = bt.read.graphql(
            api.url,
            QUERY,
            records_path="repo.issues.nodes",
            page_info_path="repo.issues.pageInfo",
            variables={"owner": "o"},
        )
        assert sorted(ds.to_pydict()["id"]) == ["a", "b", "c"]


def test_graphql_errors_with_partial_data_fail_the_read():
    partial = _page([{"id": "a", "title": None}], None, False)
    partial["errors"] = [{"message": "title is forbidden", "path": ["repo", "issues", 0, "title"]}]
    with FakeApi(_graphql_handler([partial])) as api:
        src = GraphQLSource(
            api.url, QUERY, records_path="repo.issues.nodes", variables={"owner": "o"}
        )
        with pytest.raises(BackendError, match=r"title is forbidden.*partial data"):
            src.read()


def test_graphql_error_on_a_later_page_fails_rather_than_truncating():
    pages = [
        _page([{"id": "a", "title": "x"}], "1", True),
        {"errors": [{"message": "rate limited"}]},
    ]
    with FakeApi(_graphql_handler(pages)) as api:
        src = GraphQLSource(
            api.url,
            QUERY,
            records_path="repo.issues.nodes",
            page_info_path="repo.issues.pageInfo",
            variables={"owner": "o"},
            schema=pa.schema([("id", pa.string()), ("title", pa.string())]),
        )
        with pytest.raises(BackendError, match="rate limited"):
            src.read()


def test_raise_on_errors_requires_data():
    with pytest.raises(BackendError, match="no data"):
        raise_on_errors({"data": None}, where="e")
    with pytest.raises(BackendError):
        raise_on_errors([1], where="e")
