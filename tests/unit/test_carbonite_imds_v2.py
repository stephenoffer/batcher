"""The spot-preemption probe has to survive IMDSv2, or it never fires at all.

EC2 enforces IMDSv2 whenever an instance is launched with `HttpTokens=required` — the
default for recent launch templates, and commonly mandated org-wide. An unauthenticated
GET to the metadata service then returns 401.

That is the worst possible failure for this probe, because "any error means not draining"
is *correct* behaviour: being off EC2 must not false-positive a drain. So a 401 and a
connection refusal are indistinguishable from inside, and a spot fleet that can never see
a termination notice looks exactly like a fixed on-prem cluster that has none to see. The
proactive-drain path — migrating a worker's shuffle output to a survivor before the node
goes away — is simply off, with nothing to say so.

These tests drive the probe against a fake metadata service, since there is no EC2 here.
They pin that a token is minted and presented, that an IMDSv1-only host still works, and
that nothing about the "an error is not a drain" contract was traded away to get it.
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from batcher.carbonite.resilience import preemption

pytestmark = pytest.mark.unit

_ACTION_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"


@pytest.fixture(autouse=True)
def _fresh_probes():
    """Forget the cached IMDS token and the endpoint circuit-breaker between tests.

    Both are deliberately process-lifetime state — a token is valid for minutes and an
    endpoint that has never answered never will — so a suite that shares them would have one
    test's fake metadata service answer the next test's probe.
    """
    preemption.reset_preemption_probes()
    yield
    preemption.reset_preemption_probes()


class _Response:
    def __init__(self, status: int, body: str = "") -> None:
        self.status = status
        self._body = body.encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def _fake_imds(monkeypatch, *, v2_required: bool, draining: bool, seen: list | None = None):
    """Stand in for the link-local metadata service.

    `v2_required` makes an unauthenticated GET 401, the way a `HttpTokens=required`
    instance does. Everything not on the AWS path refuses the connection, as it would off
    the matching cloud.
    """
    token = "AQAEA-fake-token"

    def urlopen(req, timeout=None):
        url = req.full_url
        headers = {k.lower(): v for k, v in req.headers.items()}
        if seen is not None:
            seen.append((req.get_method(), url, headers))
        if url == preemption._IMDS_TOKEN_URL:
            if req.get_method() != "PUT":
                raise urllib.error.HTTPError(url, 405, "method not allowed", {}, None)
            return _Response(200, token)
        if url == _ACTION_URL:
            if v2_required and headers.get("x-aws-ec2-metadata-token") != token:
                raise urllib.error.HTTPError(url, 401, "unauthorized", {}, None)
            if not draining:
                raise urllib.error.HTTPError(url, 404, "not found", {}, None)
            return _Response(200, '{"action": "terminate"}')
        raise OSError("no route to host")  # the other clouds' endpoints

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


# --- the failure this fixes ---------------------------------------------------


def test_a_drain_is_detected_on_an_imds_v2_instance(monkeypatch) -> None:
    """The whole point: `HttpTokens=required` must not silence the probe."""
    _fake_imds(monkeypatch, v2_required=True, draining=True)
    assert preemption.cloud_preemption_probe() is True


def test_the_token_is_minted_by_put_and_presented_on_the_get(monkeypatch) -> None:
    """Pins the protocol, not just the outcome — a GET for the token would 405 forever."""
    seen: list = []
    _fake_imds(monkeypatch, v2_required=True, draining=True, seen=seen)
    preemption.cloud_preemption_probe()

    mint = next(c for c in seen if c[1] == preemption._IMDS_TOKEN_URL)
    assert mint[0] == "PUT"
    assert mint[2].get("x-aws-ec2-metadata-token-ttl-seconds")

    get = next(c for c in seen if c[1] == _ACTION_URL)
    assert get[2].get("x-aws-ec2-metadata-token")


def test_a_quiet_imds_v2_instance_is_not_draining(monkeypatch) -> None:
    """Authenticating must not turn "no action scheduled" into a drain."""
    _fake_imds(monkeypatch, v2_required=True, draining=False)
    assert preemption.cloud_preemption_probe() is False


# --- what it must not cost ----------------------------------------------------


def test_an_imds_v1_instance_still_works(monkeypatch) -> None:
    """The token is additive: a host that does not need one must behave as before."""
    _fake_imds(monkeypatch, v2_required=False, draining=True)
    assert preemption.cloud_preemption_probe() is True


def test_a_host_that_cannot_mint_a_token_still_probes(monkeypatch) -> None:
    """A refused PUT is normal off EC2 and must not abort the remaining probes."""

    def urlopen(req, timeout=None):
        if req.full_url == preemption._IMDS_TOKEN_URL:
            raise OSError("no route to host")
        if req.full_url == _ACTION_URL:
            return _Response(200, "terminate")
        raise OSError("no route to host")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert preemption.cloud_preemption_probe() is True


def test_being_off_ec2_entirely_is_not_a_drain(monkeypatch) -> None:
    """The contract that makes every error safe, which the fix must not weaken."""

    def urlopen(req, timeout=None):
        raise OSError("no route to host")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert preemption.cloud_preemption_probe() is False


def test_an_empty_token_is_treated_as_no_token(monkeypatch) -> None:
    """A 200 with a blank body must not send an empty credential header."""

    def urlopen(req, timeout=None):
        if req.full_url == preemption._IMDS_TOKEN_URL:
            return _Response(200, "   ")
        raise OSError("no route to host")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert preemption._imds_v2_headers() == {}


def test_the_probe_never_raises(monkeypatch) -> None:
    """It runs on a poll thread; an exception there would kill the drain watcher."""

    def urlopen(req, timeout=None):
        raise RuntimeError("something unexpected")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert preemption.cloud_preemption_probe() is False
