"""The dashboard answers only requests its own origin could have made (BT-357).

A loopback bind keeps other machines out, not other *sites*: any page the user's browser
opens can send to `127.0.0.1`. Two attacks follow, and each is driven here with a raw
`http.client` request so the headers are exactly what a browser would send:

* **CSRF by a simple request.** A `fetch(..., {mode: "no-cors"})` with a `text/plain` body
  goes out with no preflight, and the handler parsed any body as JSON, so a foreign page
  could rewrite the durable pipeline registry.
* **DNS rebinding.** A domain that re-resolves to `127.0.0.1` becomes same-origin with the
  dashboard and reads `/api/queries`, plans and logs. Its requests carry its own name in
  `Host`, which is the only thing that tells them apart.
"""

from __future__ import annotations

import http.client
import json

import pytest

from batcher.observe.pipelines import PipelineRegistry
from batcher.observe.server import UIServer
from batcher.observe.store import ActivityStore

pytestmark = pytest.mark.unit

_BODY = json.dumps({"pipeline_id": "sig-1", "name": "Pwned"}).encode()


@pytest.fixture
def ui(tmp_path):
    store = ActivityStore(registry=PipelineRegistry(path=tmp_path / "pipelines.json"))
    server = UIServer(store, port=0)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def _request(
    server: UIServer, method: str, path: str, headers: dict[str, str], body: bytes | None = None
) -> int:
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        # `skip_host` so the test, not http.client, decides what `Host` says.
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        for name, value in headers.items():
            conn.putheader(name, value)
        if body is not None:
            conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(body)
        response = conn.getresponse()
        response.read()
        return response.status
    finally:
        conn.close()


def _host(server: UIServer) -> str:
    return f"127.0.0.1:{server.port}"


@pytest.mark.parametrize("name", ["127.0.0.1", "localhost", "[::1]", "LOCALHOST"])
def test_loopback_names_are_served(ui, name) -> None:
    assert _request(ui, "GET", "/api/queries", {"Host": f"{name}:{ui.port}"}) == 200


def test_a_rebound_domain_cannot_read_the_api(ui) -> None:
    assert _request(ui, "GET", "/api/queries", {"Host": f"evil.example:{ui.port}"}) == 403
    assert _request(ui, "HEAD", "/", {"Host": "evil.example"}) == 403


def test_a_request_with_no_host_is_refused(ui) -> None:
    assert _request(ui, "GET", "/api/queries", {}) == 403


def test_a_text_plain_post_cannot_rewrite_the_registry(ui) -> None:
    headers = {"Host": _host(ui), "Content-Type": "text/plain"}
    assert _request(ui, "POST", "/api/pipeline/meta", headers, _BODY) == 403


def test_a_cross_site_origin_cannot_post_even_as_json(ui) -> None:
    headers = {
        "Host": _host(ui),
        "Content-Type": "application/json",
        "Origin": "https://evil.example",
    }
    assert _request(ui, "POST", "/api/pipeline/meta", headers, _BODY) == 403


def test_a_rebound_domain_cannot_post(ui) -> None:
    headers = {
        "Host": f"evil.example:{ui.port}",
        "Content-Type": "application/json",
        "Origin": f"http://evil.example:{ui.port}",
    }
    assert _request(ui, "POST", "/api/pipeline/meta", headers, _BODY) == 403


@pytest.mark.parametrize("origin", [None, "same"])
def test_the_dashboards_own_write_still_lands(ui, origin) -> None:
    """Positive control: without it, every refusal above could be a server that refuses all."""
    headers = {"Host": _host(ui), "Content-Type": "application/json; charset=utf-8"}
    if origin is not None:
        headers["Origin"] = f"http://{_host(ui)}"
    assert _request(ui, "POST", "/api/pipeline/meta", headers, _BODY) == 200
