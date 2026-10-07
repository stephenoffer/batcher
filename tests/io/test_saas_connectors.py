"""The SaaS connectors against local fakes: GitHub, Salesforce, Google Sheets, SharePoint.

Each fake speaks the documented request and response shapes of the real API over a local
socket (`_fake_http.FakeApi`), so these tests pin what each connector *sends* -- paths,
query parameters, bodies, headers -- and how it maps what comes back. They do not prove the
real services answer that way; the live smoke tests under ``tests/integration/live`` do.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from urllib.parse import unquote

import pyarrow as pa
import pytest

import batcher as bt
from _fake_http import FakeApi, FakeResponse
from batcher._internal.errors import BackendError, FormatError, PlanError
from batcher.io.formats.http import transport
from batcher.io.formats.saas import salesforce as sf_module
from batcher.io.formats.saas.github import GitHubSource
from batcher.io.formats.saas.msgraph import SharePointSource
from batcher.io.formats.saas.salesforce import SalesforceSource
from batcher.io.formats.saas.sheets import GoogleSheetsSink, GoogleSheetsSource, values_to_table


@pytest.fixture
def no_sleep(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(transport, "_sleep", waits.append)
    monkeypatch.setattr(sf_module, "_sleep", waits.append)
    return waits


# ---- GitHub ---------------------------------------------------------------------------
def _issue(i: int, updated: str) -> dict:
    return {
        "id": i,
        "number": i,
        "title": f"t{i}",
        "state": "open",
        "user": {"login": "u", "id": 7, "avatar_url": "x"},
        "labels": [{"name": "bug", "color": "red"}],
        "comments": 0,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": updated,
        "closed_at": None,
        "body": None,
        "html_url": f"https://github.com/o/r/issues/{i}",
        "milestone": None,
    }


def test_github_issues_follow_link_pages_and_wait_out_the_rate_limit(no_sleep, monkeypatch, caplog):
    monkeypatch.setattr(transport, "_now", lambda: 1_000.0)
    issues = [_issue(i, f"2024-01-0{i}T00:00:00Z") for i in range(1, 4)]
    hits = {"n": 0}

    def handler(req):
        assert req.path == "/repos/o/r/issues"
        assert req.headers["authorization"] == "Bearer gh-secret"
        assert req.headers["x-github-api-version"] == "2022-11-28"
        hits["n"] += 1
        if hits["n"] == 1:
            return FakeResponse(
                403,
                {"message": "rate limit"},
                {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1010"},
            )
        page = int(req.query.get("page", 1))
        link = {"Link": f'</repos/o/r/issues?page={page + 1}&per_page=100>; rel="next"'}
        return FakeResponse(
            body=issues[(page - 1) * 2 : page * 2], headers=link if page == 1 else {}
        )

    monkeypatch.setenv("GH_TOKEN_TEST", "gh-secret")
    caplog.set_level(logging.DEBUG)
    with FakeApi(handler) as api:
        src = GitHubSource("o/r", "issues", token="env:GH_TOKEN_TEST", base_url=api.url)
        batches = src.read()
        assert api.requests[1].query == {"per_page": "100", "state": "all"}
    rows = pa.Table.from_batches(batches).to_pydict()
    assert rows["number"] == [1, 2, 3]
    assert rows["user"][0] == {"login": "u", "id": 7}
    assert rows["labels"][0] == [{"name": "bug"}]
    assert rows["updated_at"][0] == datetime(2024, 1, 1, tzinfo=UTC)
    assert no_sleep == [11.0]  # until x-ratelimit-reset, plus a second of slack
    assert "gh-secret" not in caplog.text and "gh-secret" not in src.identity()


def test_github_incremental_sends_since_and_drops_delivered_versions(tmp_path):
    issues = [_issue(1, "2024-01-01T00:00:00Z"), _issue(2, "2024-01-02T00:00:00Z")]

    def handler(req):
        since = req.query.get("since")
        return FakeResponse(body=[i for i in issues if since is None or i["updated_at"] >= since])

    inc = bt.io.Incremental(state=str(tmp_path / "gh.json"))
    with FakeApi(handler) as api:
        first = GitHubSource("o/r", incremental=inc, base_url=api.url).read()
        assert sum(b.num_rows for b in first) == 2
        issues.append(_issue(3, "2024-01-03T00:00:00Z"))
        second = GitHubSource("o/r", incremental=inc, base_url=api.url).read()
        assert api.requests[-1].query["since"] == "2024-01-02T00:00:00Z"
    assert [n for b in second for n in b.column("number").to_pylist()] == [3]


def test_github_rejects_bad_arguments():
    with pytest.raises(PlanError):
        GitHubSource("o/r", "commits")
    with pytest.raises(PlanError):
        GitHubSource("not-a-repo")
    with pytest.raises(PlanError, match="since"):
        GitHubSource("o/r", "pulls", since="2024-01-01T00:00:00Z")


# ---- Salesforce -----------------------------------------------------------------------
_SF_SCHEMA = pa.schema(
    [
        ("Id", pa.string()),
        ("Name", pa.string()),
        ("Amount", pa.float64()),
        ("SystemModstamp", pa.timestamp("ms", tz="UTC")),
    ]
)


class _Salesforce:
    def __init__(self, pages: list[str], *, fail: bool = False) -> None:
        self.pages = pages
        self.fail = fail
        self.polls = 0
        self.jobs: list[dict] = []

    def __call__(self, req) -> FakeResponse:
        base = "/services/data/v62.0/jobs/query"
        if req.method == "POST" and req.path == base:
            self.jobs.append(req.json())
            return FakeResponse(body={"id": "750J1", "state": "UploadComplete"})
        if req.path == f"{base}/750J1":
            self.polls += 1
            if self.fail:
                return FakeResponse(body={"state": "Failed", "errorMessage": "INVALID_FIELD: Nope"})
            return FakeResponse(body={"state": "InProgress" if self.polls == 1 else "JobComplete"})
        if req.path == f"{base}/750J1/results":
            assert req.headers["accept"] == "text/csv"
            index = int(req.query.get("locator", 0))
            nxt = str(index + 1) if index + 1 < len(self.pages) else "null"
            return FakeResponse(body=self.pages[index].encode(), headers={"Sforce-Locator": nxt})
        return FakeResponse(404, b"unexpected")


def test_salesforce_bulk_query_pages_through_locators_with_declared_types(no_sleep):
    pages = [
        'Id,Name,Amount,SystemModstamp\n"001A","Acme",10.5,2024-05-01T10:00:00.000Z\n',
        'Id,Name,Amount,SystemModstamp\n"001B","",,2024-05-02T10:00:00.000Z\n',
    ]
    fake = _Salesforce(pages)
    with FakeApi(fake) as api:
        src = SalesforceSource(
            "Opportunity",
            instance_url=api.url,
            schema=_SF_SCHEMA,
            auth=bt.io.BearerToken("sf-token"),
            where="Amount > 0",
        )
        table = pa.Table.from_batches(src.read())
    assert fake.jobs == [
        {
            "operation": "query",
            "query": "SELECT Id, Name, Amount, SystemModstamp FROM Opportunity WHERE (Amount > 0)",
        }
    ]
    assert table.schema == _SF_SCHEMA
    assert table.column("Id").to_pylist() == ["001A", "001B"]
    assert table.column("Amount").to_pylist() == [10.5, None]
    assert table.column("Name").to_pylist() == ["Acme", None]
    assert src.progress() == {"job": "750J1", "records": 2, "cursor": "1"}
    assert no_sleep == [2.0]  # one InProgress poll


def test_salesforce_query_all_adds_is_deleted_and_resumes_by_system_modstamp(no_sleep, tmp_path):
    page = (
        "Id,Name,Amount,SystemModstamp,IsDeleted\n"
        "001A,Acme,1,2024-05-01T10:00:00.000Z,false\n"
        "001B,Gone,2,2024-05-03T09:30:15.250Z,true\n"
    )
    fake = _Salesforce([page])
    inc = bt.io.Incremental(state=str(tmp_path / "sf.json"))
    with FakeApi(fake) as api:
        src = SalesforceSource(
            "Opportunity",
            instance_url=api.url,
            schema=_SF_SCHEMA,
            include_deleted=True,
            incremental=inc,
        )
        table = pa.Table.from_batches(src.read())
        assert table.column("IsDeleted").to_pylist() == [False, True]
        assert inc.load()["watermark"] == "2024-05-03T09:30:15.250000Z"
        fake.polls = 0
        again = sum(b.num_rows for b in src.read())
    assert fake.jobs[0]["operation"] == "queryAll"
    assert fake.jobs[1]["query"].endswith("WHERE SystemModstamp >= 2024-05-03T09:30:15Z")
    # The boundary record is the same version, so the resumed read delivers nothing new.
    assert again == 0


def test_salesforce_failed_job_and_bad_values_raise(no_sleep):
    with FakeApi(_Salesforce([], fail=True)) as api:
        src = SalesforceSource("Account", instance_url=api.url, schema=_SF_SCHEMA)
        with pytest.raises(BackendError, match="INVALID_FIELD"):
            src.read()
    bad = "Id,Name,Amount,SystemModstamp\n001A,Acme,lots,2024-05-01T10:00:00.000Z\n"
    with FakeApi(_Salesforce([bad])) as api:
        src = SalesforceSource("Account", instance_url=api.url, schema=_SF_SCHEMA)
        with pytest.raises(FormatError, match="Amount"):
            src.read()
    with pytest.raises(PlanError, match="SystemModstamp"):
        SalesforceSource(
            "Account",
            instance_url="http://x",
            schema=pa.schema([("Id", pa.string())]),
            incremental=bt.io.Incremental(state="s"),
        )
    with pytest.raises(PlanError):
        SalesforceSource("Account; DROP", instance_url="http://x", schema=_SF_SCHEMA)


# ---- Google Sheets --------------------------------------------------------------------
def test_values_to_table_header_policies_and_padding():
    rows = [["name", "n", "flag"], ["a", 1, True], ["b", "", False], ["c"]]
    table = values_to_table(rows, header=True, schema=None)
    assert table.to_pydict() == {
        "name": ["a", "b", "c"],
        "n": [1, None, None],
        "flag": [True, False, None],
    }
    named = values_to_table(rows[1:], header=["x", "y", "z"], schema=None)
    assert named.column_names == ["x", "y", "z"]
    assert values_to_table(rows[1:2], header=False, schema=None).column_names == ["c0", "c1", "c2"]
    typed = values_to_table(
        rows, header=True, schema=pa.schema([("n", pa.float64()), ("name", pa.string())])
    )
    assert typed.column_names == ["n", "name"] and typed.column("n").type == pa.float64()
    with pytest.raises(FormatError, match="mixes cell types"):
        values_to_table([["a"], [1], ["x"]], header=True, schema=None)
    with pytest.raises(FormatError, match="repeats"):
        values_to_table([["a", "a"], [1, 2]], header=True, schema=None)


def test_google_sheets_read_sends_the_render_options():
    def handler(req):
        assert unquote(req.path) == "/v4/spreadsheets/S1/values/Sheet1!A1:C"
        assert req.query == {
            "majorDimension": "ROWS",
            "valueRenderOption": "UNFORMATTED_VALUE",
            "dateTimeRenderOption": "FORMATTED_STRING",
        }
        assert req.headers["authorization"] == "Bearer g-token"
        return FakeResponse(
            body={"range": "Sheet1!A1:C3", "values": [["k", "v"], ["a", 1.5], ["b", 2]]}
        )

    with FakeApi(handler) as api:
        src = GoogleSheetsSource(
            "S1", "Sheet1!A1:C", auth=bt.io.BearerToken("g-token"), base_url=api.url
        )
        assert pa.Table.from_batches(src.read()).to_pydict() == {"k": ["a", "b"], "v": [1.5, 2.0]}
    with pytest.raises(PlanError):
        GoogleSheetsSource("S1", "A1", value_render="PRETTY")


def test_google_sheets_overwrite_clears_exactly_the_range_then_writes_in_batches():
    calls: list[tuple[str, dict, object]] = []

    def handler(req):
        calls.append((unquote(req.path), req.query, req.json()))
        return FakeResponse(body={})

    table = pa.table(
        {
            "region": ["e", "w", "n"],
            "sales": [1, 2, 3],
            "day": pa.array([datetime(2024, 1, d) for d in (1, 2, 3)], pa.timestamp("s")),
        }
    )
    with FakeApi(handler) as api:
        sink = GoogleSheetsSink(
            range="Out!A1", batch_rows=2, auth=bt.io.BearerToken("t"), base_url=api.url
        )
        written = sink.write(table, "S1")
    assert written.rows == 3
    assert calls[0][0] == "/v4/spreadsheets/S1/values/Out!A1:clear"
    appends = calls[1:]
    assert [c[0] for c in appends] == ["/v4/spreadsheets/S1/values/Out!A1:append"] * 2
    assert appends[0][1] == {"valueInputOption": "RAW", "insertDataOption": "OVERWRITE"}
    assert appends[0][2]["values"] == [["region", "sales", "day"], ["e", 1, "2024-01-01 00:00:00"]]
    assert appends[1][2]["values"] == [
        ["w", 2, "2024-01-02 00:00:00"],
        ["n", 3, "2024-01-03 00:00:00"],
    ]


def test_google_sheets_append_and_refusals():
    calls: list[tuple[str, dict]] = []

    def handler(req):
        calls.append((unquote(req.path), req.query))
        return FakeResponse(body={})

    with FakeApi(handler) as api:
        sink = GoogleSheetsSink(
            range="Out!A1", mode="append", auth=bt.io.BearerToken("t"), base_url=api.url
        )
        sink.write(pa.table({"a": [1]}), "S1")
        assert calls == [
            (
                "/v4/spreadsheets/S1/values/Out!A1:append",
                {"valueInputOption": "RAW", "insertDataOption": "INSERT_ROWS"},
            )
        ]
        with pytest.raises(FormatError, match="NaN"):
            sink.write(pa.table({"x": [float("nan")]}), "S1")
        over = GoogleSheetsSink(range="Out!A1", auth=bt.io.BearerToken("t"), base_url=api.url)
        with pytest.raises(BackendError, match="distributed"):
            over.write_partitioned(pa.table({"a": [1]}), "S1", file_index=1)
    with pytest.raises(BackendError):
        GoogleSheetsSink(range="A1", mode="upsert")


def test_google_sheets_writer_entry_point_reaches_the_sink(monkeypatch):
    seen: dict = {}

    def fake_write(self, table, path):
        seen.update(path=path, rows=table.num_rows, mode=self.mode)
        from batcher.io.manifest import WrittenFile

        return WrittenFile(path=path, rows=table.num_rows, bytes=0)

    monkeypatch.setattr(GoogleSheetsSink, "write", fake_write)
    bt.from_pydict({"a": [1, 2]}).write.google_sheets("S1", "Out!A1", mode="append")
    assert seen == {"path": "S1", "rows": 2, "mode": "append"}


# ---- SharePoint / OneDrive -------------------------------------------------------------
def _item(i: str, name: str, *, path="/drive/root:/Docs", deleted=False, folder=False) -> dict:
    item = {
        "id": i,
        "name": name,
        "size": 10,
        "eTag": f"e{i}",
        "cTag": f"c{i}",
        "webUrl": f"https://t/{name}",
        "lastModifiedDateTime": "2024-02-01T08:00:00Z",
        "parentReference": {"id": "p", "path": path, "driveId": "d"},
    }
    if deleted:
        item = {"id": i, "deleted": {"state": "deleted"}, "parentReference": {"id": "p"}}
    elif folder:
        item["folder"] = {"childCount": 1}
    else:
        item["file"] = {"mimeType": "text/plain"}
    return item


def test_sharepoint_delta_walk_keeps_the_delta_link_and_reconciles_changes(tmp_path):
    state = {"phase": 1}

    def handler(req):
        if req.path == "/drives/D1/root/delta":
            if "token" in req.query:  # the delta link from the first walk
                return FakeResponse(
                    body={
                        "value": [_item("1", "renamed.txt"), _item("2", "", deleted=True)],
                        "@odata.deltaLink": f"{api.url}/drives/D1/root/delta?token=t2",
                    }
                )
            if req.query.get("page") == "2":
                return FakeResponse(
                    body={
                        "value": [_item("3", "sub", folder=True)],
                        "@odata.deltaLink": f"{api.url}/drives/D1/root/delta?token=t1",
                    }
                )
            return FakeResponse(
                body={
                    "value": [_item("1", "a.txt"), _item("2", "b.txt")],
                    "@odata.nextLink": f"{api.url}/drives/D1/root/delta?page=2",
                }
            )
        return FakeResponse(404, b"")

    path = str(tmp_path / "delta.json")
    with FakeApi(handler) as api:
        first = pa.Table.from_batches(
            SharePointSource(
                drive_id="D1", auth=bt.io.BearerToken("m"), state=path, base_url=api.url
            ).read()
        )
        assert first.column("name").to_pylist() == ["a.txt", "b.txt", "sub"]
        assert first.column("is_folder").to_pylist() == [False, False, True]
        assert first.column("parent_path").to_pylist()[0] == "/drive/root:/Docs"
        state["phase"] = 2
        second_src = SharePointSource(
            drive_id="D1", auth=bt.io.BearerToken("m"), state=path, base_url=api.url
        )
        second = pa.Table.from_batches(second_src.read())
        assert api.requests[-1].query == {"token": "t1"}
    assert second.column("id").to_pylist() == ["1", "2"]
    assert second.column("name").to_pylist()[0] == "renamed.txt"
    assert second.column("deleted").to_pylist() == [False, True]
    assert second_src.progress()["cursor"].endswith("token=t2")


def test_sharepoint_expired_delta_link_and_folder_filter_and_content(tmp_path):
    def handler(req):
        if req.path == "/drives/D1/root/delta" and req.query.get("token") == "old":
            return FakeResponse(410, {"error": {"code": "resyncRequired"}})
        if req.path == "/drives/D1/root/delta":
            return FakeResponse(
                body={
                    "value": [
                        _item("1", "in.txt", path="/drive/root:/Reports"),
                        _item("2", "out.txt", path="/drive/root:/Other"),
                        _item("3", "deep.txt", path="/drive/root:/Reports/2024"),
                        _item("4", "", deleted=True),
                    ],
                    "@odata.deltaLink": "x",
                }
            )
        if req.path == "/drives/D1/items/1/content":
            return FakeResponse(302, b"", {"Location": f"{blob.url}/blob/1"})
        if req.path == "/drives/D1/items/3/content":
            return FakeResponse(body=b"deep-bytes")
        return FakeResponse(404, b"")

    with FakeApi(lambda req: FakeResponse(body=b"in-bytes")) as blob, FakeApi(handler) as api:
        src = SharePointSource(
            drive_id="D1",
            auth=bt.io.BearerToken("m"),
            folder="/Reports",
            include_content=True,
            base_url=api.url,
        )
        table = pa.Table.from_batches(src.read())
        assert table.column("name").to_pylist() == ["in.txt", "deep.txt", None]
        assert table.column("content").to_pylist() == [b"in-bytes", b"deep-bytes", None]
        assert "authorization" not in blob.requests[-1].headers
        state = tmp_path / "s.json"
        state.write_text('{"cursor": "' + api.url + '/drives/D1/root/delta?token=old"}')
        with pytest.raises(BackendError, match="410"):
            SharePointSource(
                drive_id="D1", state=str(state), auth=bt.io.BearerToken("m"), base_url=api.url
            ).read()
    with pytest.raises(PlanError):
        SharePointSource()
