"""`bt.read.http_json` against a real local HTTP server: paging, schema, retries, auth, limits.

Every test drives the source over a socket (`_fake_http.FakeApi`), so what is pinned is the
request actually sent and the rows actually returned.
"""

from __future__ import annotations

import pickle
import threading
import time
from datetime import UTC, datetime

import pyarrow as pa
import pytest

import batcher as bt
from _fake_http import FakeApi, FakeResponse
from batcher._internal.errors import BackendError, FormatError, PlanError
from batcher.io.formats.http import transport
from batcher.io.formats.http.options import next_link_from_header
from batcher.io.formats.http.source import HttpJsonSource

ROWS = [{"id": i, "name": f"n{i}", "score": i * 0.5} for i in range(10)]


@pytest.fixture
def no_sleep(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(transport, "_sleep", waits.append)
    return waits


def _sorted_ids(ds) -> list[int]:
    return sorted(ds.to_pydict()["id"])


def test_cursor_pagination_reads_every_page_and_reports_the_cursor():
    def handler(req):
        start = int(req.query.get("cursor", 0))
        nxt = start + 4 if start + 4 < len(ROWS) else None
        return FakeResponse(body={"data": ROWS[start : start + 4], "meta": {"next": nxt}})

    with FakeApi(handler) as api:
        src = HttpJsonSource(
            f"{api.url}/items",
            pagination=bt.io.CursorPagination(cursor_path="meta.next"),
            records_path="data",
        )
        batches = src.read()
        assert sum(b.num_rows for b in batches) == 10
        assert src.progress() == {"pages": 3, "records": 10, "cursor": 8}
        assert [r.query.get("cursor") for r in api.requests[-3:]] == [None, "4", "8"]
        ds = bt.read.http_json(
            f"{api.url}/items",
            pagination=bt.io.CursorPagination(cursor_path="meta.next"),
            records_path="data",
        )
        assert ds.schema.names == ["id", "name", "score"]
        assert _sorted_ids(ds) == list(range(10))


def test_has_more_false_stops_even_with_a_cursor():
    def handler(req):
        return FakeResponse(body={"items": ROWS[:2], "next": "x", "has_more": False})

    with FakeApi(handler) as api:
        src = HttpJsonSource(
            api.url,
            pagination=bt.io.CursorPagination(cursor_path="next", has_more_path="has_more"),
            records_path="items",
        )
        assert sum(b.num_rows for b in src.read()) == 2


def test_link_header_pagination_follows_relative_next_links():
    def handler(req):
        page = int(req.query.get("page", 1))
        headers = {"Link": f'</items?page={page + 1}>; rel="next", </items?page=1>; rel="first"'}
        return FakeResponse(
            body=ROWS[(page - 1) * 5 : page * 5], headers=headers if page < 2 else {}
        )

    with FakeApi(handler) as api:
        src = HttpJsonSource(
            f"{api.url}/items?page=1", pagination=bt.io.NextLinkPagination(), schema=None
        )
        assert sum(b.num_rows for b in src.read()) == 10
        assert src.progress()["cursor"] == f"{api.url}/items?page=2"


def test_link_header_parsing():
    assert next_link_from_header('<https://a/x?page=2>; rel="next"') == "https://a/x?page=2"
    assert next_link_from_header('<u1>; rel="prev", <u2>; rel="next last"') == "u2"
    assert next_link_from_header('<u1>; rel="prev"') is None
    assert next_link_from_header(None) is None


def test_body_next_link_pagination():
    def handler(req):
        page = int(req.query.get("p", 0))
        nxt = f"/r?p={page + 1}" if page < 1 else None
        return FakeResponse(body={"results": ROWS[page * 5 : page * 5 + 5], "next": nxt})

    with FakeApi(handler) as api:
        ds = bt.read.http_json(
            f"{api.url}/r",
            pagination=bt.io.NextLinkPagination(path="next"),
            records_path="results",
        )
        assert _sorted_ids(ds) == list(range(10))


def test_offset_pagination_stops_on_a_short_page():
    def handler(req):
        off, lim = int(req.query["offset"]), int(req.query["limit"])
        return FakeResponse(body=ROWS[off : off + lim])

    with FakeApi(handler) as api:
        src = HttpJsonSource(api.url, pagination=bt.io.OffsetPagination(limit=4))
        assert sum(b.num_rows for b in src.read()) == 10
        assert [r.query["offset"] for r in api.requests[-3:]] == ["0", "4", "8"]


def test_offset_pagination_honors_a_reported_total():
    def handler(req):
        off = int(req.query["skip"])
        return FakeResponse(body={"rows": ROWS[off : off + 5], "total": 10})

    with FakeApi(handler) as api:
        src = HttpJsonSource(
            api.url,
            pagination=bt.io.OffsetPagination(limit=5, offset_param="skip", total_path="total"),
            records_path="rows",
        )
        assert sum(b.num_rows for b in src.read()) == 10
        # The total says page 3 does not exist, so it is never requested.
        assert len([r for r in api.requests if "skip" in r.query]) == 3  # 1 inference + 2


def test_page_pagination_stops_on_an_empty_page():
    def handler(req):
        page = int(req.query["page"])
        return FakeResponse(body={"items": ROWS[(page - 1) * 3 : page * 3]})

    with FakeApi(handler) as api:
        ds = bt.read.http_json(api.url, pagination=bt.io.PagePagination(), records_path="items")
        assert _sorted_ids(ds) == list(range(10))


def test_declared_schema_casts_timestamps_and_ignores_extra_fields():
    records = [{"id": 1, "at": "2024-03-01T10:00:00Z", "extra": {"x": 1}}, {"id": 2, "at": None}]
    schema = pa.schema([("id", pa.int64()), ("at", pa.timestamp("s", tz="UTC"))])
    with FakeApi(lambda req: FakeResponse(body=records)) as api:
        out = bt.read.http_json(api.url, schema=schema).sort("id").to_pydict()
    assert out["at"][0] == datetime(2024, 3, 1, 10, tzinfo=UTC)
    assert out["at"][1] is None
    assert list(out) == ["id", "at"]


def test_inferred_schema_refuses_a_field_the_first_page_lacked():
    def handler(req):
        page = int(req.query["page"])
        body = [{"id": 1}] if page == 1 else ([{"id": 2, "late": "x"}] if page == 2 else [])
        return FakeResponse(body=body)

    with FakeApi(handler) as api:
        src = HttpJsonSource(api.url, pagination=bt.io.PagePagination())
        with pytest.raises(FormatError, match="late"):
            src.read()


def test_inferred_null_column_names_the_field_when_a_value_arrives():
    def handler(req):
        page = int(req.query["page"])
        body = {1: [{"id": 1, "m": None}], 2: [{"id": 2, "m": {"k": 1}}]}.get(page, [])
        return FakeResponse(body=body)

    with FakeApi(handler) as api:
        src = HttpJsonSource(api.url, pagination=bt.io.PagePagination())
        with pytest.raises(FormatError, match=r"\['m'\]"):
            src.read()


def test_records_path_pointing_at_a_non_list_is_a_format_error():
    with (
        FakeApi(lambda req: FakeResponse(body={"data": "oops"})) as api,
        pytest.raises(FormatError, match="records_path"),
    ):
        HttpJsonSource(api.url, records_path="data", schema=pa.schema([("a", pa.int64())])).read()


def test_empty_first_page_without_a_schema_explains_itself():
    with FakeApi(lambda req: FakeResponse(body=[])) as api:
        with pytest.raises(PlanError, match="declare schema="):
            HttpJsonSource(api.url).schema()
        empty = HttpJsonSource(api.url, schema=pa.schema([("a", pa.int64())])).read()
        assert empty == []


def test_retry_after_is_honored_and_the_request_retried(no_sleep):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse(429, {"error": "slow down"}, {"Retry-After": "7"})
        if calls["n"] == 2:
            return FakeResponse(503, b"busy")
        return FakeResponse(body=[{"a": 1}])

    with FakeApi(handler) as api:
        src = HttpJsonSource(
            api.url,
            schema=pa.schema([("a", pa.int64())]),
            retry=bt.io.RetryPolicy(backoff=0.25),
        )
        assert sum(b.num_rows for b in src.read()) == 1
    assert no_sleep[0] == 7.0
    assert 0.0 <= no_sleep[1] <= 0.5  # second attempt: jittered backoff up to 0.25 * 2


def test_retry_after_longer_than_the_cap_fails_the_read(no_sleep):
    with FakeApi(lambda req: FakeResponse(429, b"", {"Retry-After": "9999"})) as api:
        src = HttpJsonSource(api.url, schema=pa.schema([("a", pa.int64())]))
        with pytest.raises(BackendError, match="max_retry_after"):
            src.read()


def test_exhausted_retries_and_client_errors_never_print_the_query(no_sleep):
    with FakeApi(lambda req: FakeResponse(500, b"boom")) as api:
        src = HttpJsonSource(
            api.url,
            params={"api_key": "SECRET-VALUE"},
            schema=pa.schema([("a", pa.int64())]),
            retry=bt.io.RetryPolicy(max_attempts=3),
        )
        with pytest.raises(BackendError) as err:
            src.read()
        assert len(api.requests) == 3
        assert "SECRET-VALUE" not in str(err.value)
        assert "HTTP 500" in str(err.value)
    with FakeApi(lambda req: FakeResponse(404, b"nope")) as api:
        with pytest.raises(BackendError, match="HTTP 404"):
            HttpJsonSource(api.url, schema=pa.schema([("a", pa.int64())])).read()
        assert len(api.requests) == 1  # a 404 is not retried


def test_bearer_token_resolves_a_secret_reference_and_hides_it(monkeypatch):
    monkeypatch.setenv("XHTTP_TOKEN", "tok-123")
    with FakeApi(lambda req: FakeResponse(body=[{"a": 1}])) as api:
        auth = bt.io.BearerToken("env:XHTTP_TOKEN")
        src = HttpJsonSource(api.url, auth=auth, headers={"X-Key": "env:XHTTP_TOKEN"})
        src.read()
        assert api.requests[-1].headers["authorization"] == "Bearer tok-123"
        assert api.requests[-1].headers["x-key"] == "tok-123"
    assert "XHTTP_TOKEN" not in repr(auth) and "tok-123" not in repr(src.identity())
    assert "tok-123" not in pickle.dumps(src).decode("latin-1")


def test_oauth2_client_credentials_fetches_caches_and_refreshes_on_401():
    issued = {"n": 0}
    revoked = {"first": True}

    def handler(req):
        if req.path == "/token":
            issued["n"] += 1
            form = dict(p.split("=") for p in req.body.decode().split("&"))
            assert form["grant_type"] == "client_credentials"
            assert form["client_secret"] == "s3cret"
            return FakeResponse(body={"access_token": f"t{issued['n']}", "expires_in": 3600})
        if req.headers.get("authorization") == "Bearer t1" and revoked["first"]:
            revoked["first"] = False
            return FakeResponse(401, b"expired")
        return FakeResponse(body=[{"a": 1}])

    with FakeApi(handler) as api:
        auth = bt.io.OAuth2ClientCredentials(
            token_url=f"{api.url}/token", client_id="c", client_secret="s3cret", scope="read"
        )
        schema = pa.schema([("a", pa.int64())])
        HttpJsonSource(f"{api.url}/data", auth=auth, schema=schema).read()
        HttpJsonSource(f"{api.url}/data", auth=auth, schema=schema).read()
        auth.invalidate()
    # t1 answered 401 once, t2 was fetched and then reused by the second read.
    assert issued["n"] == 2
    data_auth = [r.headers["authorization"] for r in api.requests if r.path == "/data"]
    assert data_auth == ["Bearer t1", "Bearer t2", "Bearer t2"]
    assert "s3cret" not in repr(auth)


def test_auth_header_is_not_forwarded_across_a_redirect():
    with FakeApi(lambda req: FakeResponse(body=[{"a": 1}])) as other:
        target = f"{other.url}/download"
        with FakeApi(lambda req: FakeResponse(302, b"", {"Location": target})) as api:
            HttpJsonSource(api.url, auth=bt.io.BearerToken("lit-token")).read()
            assert api.requests[-1].headers["authorization"] == "Bearer lit-token"
        assert "authorization" not in other.requests[-1].headers


def test_max_concurrency_bounds_requests_in_flight_per_process():
    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def handler(req):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return FakeResponse(body=[{"a": 1}])

    with FakeApi(handler) as api:
        schema = pa.schema([("a", pa.int64())])
        sources = [
            HttpJsonSource(f"{api.url}/{i}", schema=schema, max_concurrency=2) for i in range(6)
        ]
        threads = [threading.Thread(target=s.read) for s in sources]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert len(api.requests) == 6
    assert state["peak"] <= 2


def test_construction_rejects_bad_options():
    with pytest.raises(PlanError):
        HttpJsonSource("http://x", max_concurrency=0)
    with pytest.raises(PlanError):
        HttpJsonSource("http://x", method="DELETE")
    with pytest.raises(PlanError):
        bt.io.OffsetPagination(limit=0)
    with pytest.raises(PlanError):
        bt.io.RetryPolicy(max_attempts=0)


def test_source_and_split_survive_pickling():
    def handler(req):
        page = int(req.query["page"])
        return FakeResponse(body=ROWS[(page - 1) * 4 : page * 4])

    with FakeApi(handler) as api:
        src = HttpJsonSource(api.url, pagination=bt.io.PagePagination())
        (split,) = src.splits()
        clone = pickle.loads(pickle.dumps(split))
        assert sum(b.num_rows for b in clone.read()) == 10


def test_post_body_is_sent_with_every_page():
    def handler(req):
        assert req.method == "POST" and req.json() == {"q": "x"}
        page = int(req.query["page"])
        return FakeResponse(body=ROWS[(page - 1) * 6 : page * 6])

    with FakeApi(handler) as api:
        src = HttpJsonSource(
            api.url, method="POST", body={"q": "x"}, pagination=bt.io.PagePagination(size=6)
        )
        assert sum(b.num_rows for b in src.read()) == 10


def test_projection_selects_columns():
    with FakeApi(lambda req: FakeResponse(body=ROWS[:3])) as api:
        ds = bt.read.http_json(api.url).select("name")
        assert sorted(ds.to_pydict()["name"]) == ["n0", "n1", "n2"]
