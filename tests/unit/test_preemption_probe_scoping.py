"""What an *unanswerable* preemption endpoint costs, which is the normal case on most hosts.

Only one of the four cloud reclamation endpoints can ever answer on a given node, and on a
neocloud, an HPC cluster or on-prem hardware, none of them can. Each unreachable probe waits
out its timeout, and the poll loop runs for the life of the worker — so the cost of the
endpoints that will never answer is paid forever, to learn nothing.

Two bounds fix that without weakening the contract that makes the probe safe (any error means
"not draining", so being off a cloud never false-positives a drain): the site's own identity
skips the platforms it is not, and an endpoint unreachable three times running is not tried
again, because a metadata service does not appear partway through a job.
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from batcher.carbonite.resilience import preemption

pytestmark = pytest.mark.unit

_AWS_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"
_GCP_URL = "http://metadata.google.internal/computeMetadata/v1/instance/preempted"
_ALIBABA_URL = "http://100.100.100.200/latest/meta-data/instance/spot/termination-time"


@pytest.fixture(autouse=True)
def _fresh_probes(monkeypatch):
    """Forget the circuit-breaker state, and silence the firmware.

    The firmware is a *second* identity, deliberately: a GPU cloud reselling hyperscaler
    capacity exports its own marker over EC2 or GCE hardware, and the platform's endpoint
    really does answer there. This suite is about the environment marker, so DMI is silenced
    by default and the one test that is about it puts it back.
    """
    from batcher._internal.site import provider

    monkeypatch.setattr(provider, "dmi_identity", lambda: ("", "", None))
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


def _record(monkeypatch, handler):
    """Install `handler` as the transport and return the list of URLs it was asked for."""
    tried: list[str] = []

    def urlopen(req, timeout=None):
        tried.append(req.full_url)
        return handler(req)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return tried


def _unreachable(req):
    raise OSError("no route to host")


def test_a_known_platform_does_not_probe_the_other_clouds(monkeypatch):
    monkeypatch.setenv("BATCHER_PROVIDER", "gcp")
    tried = _record(monkeypatch, _unreachable)
    assert preemption.cloud_preemption_probe() is False
    assert tried == [_GCP_URL], "only the platform this node actually is"


def test_an_unidentified_site_still_tries_everything(monkeypatch):
    # The only safe answer when the environment says nothing, and what this did before.
    monkeypatch.setenv("BATCHER_PROVIDER", "unknown")
    tried = _record(monkeypatch, _unreachable)
    assert preemption.cloud_preemption_probe() is False
    assert _AWS_URL in tried and _GCP_URL in tried and _ALIBABA_URL in tried


def test_an_endpoint_that_never_answers_is_given_up_on(monkeypatch):
    monkeypatch.setenv("BATCHER_PROVIDER", "unknown")
    tried = _record(monkeypatch, _unreachable)
    for _ in range(preemption._PROBE_FAILURE_LIMIT):
        preemption.cloud_preemption_probe()
    before = len(tried)
    preemption.cloud_preemption_probe()
    assert len(tried) == before, "every endpoint has been unreachable its limit of times"


def test_a_reachable_endpoint_is_never_given_up_on(monkeypatch):
    # Reachability is what counts, not the answer: a 200 saying "not draining" proves the
    # endpoint is there, and a spot node spends its whole life in that state.
    monkeypatch.setenv("BATCHER_PROVIDER", "gcp")

    def quiet(req):
        return _Response(200, "FALSE")

    tried = _record(monkeypatch, quiet)
    for _ in range(preemption._PROBE_FAILURE_LIMIT + 2):
        assert preemption.cloud_preemption_probe() is False
    assert len(tried) == preemption._PROBE_FAILURE_LIMIT + 2


def test_a_blip_does_not_switch_off_the_only_signal_a_spot_worker_has(monkeypatch):
    monkeypatch.setenv("BATCHER_PROVIDER", "gcp")
    state = {"fail": 2}

    def flaky(req):
        if state["fail"] > 0:
            state["fail"] -= 1
            raise OSError("transient")
        return _Response(200, "TRUE")

    _record(monkeypatch, flaky)
    assert preemption.cloud_preemption_probe() is False
    assert preemption.cloud_preemption_probe() is False
    assert preemption.cloud_preemption_probe() is True, "two failures is under the limit"


def test_an_alibaba_spot_reclamation_is_detected(monkeypatch):
    # Alibaba answers 404 until a reclamation is scheduled, so a non-empty 200 is the notice.
    monkeypatch.setenv("BATCHER_PROVIDER", "alibaba")

    def scheduled(req):
        if req.full_url == _ALIBABA_URL:
            return _Response(200, "2026-08-18T10:00:00Z")
        raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, None)

    _record(monkeypatch, scheduled)
    assert preemption.cloud_preemption_probe() is True


def test_an_alibaba_node_with_nothing_scheduled_is_not_draining(monkeypatch):
    monkeypatch.setenv("BATCHER_PROVIDER", "alibaba")

    def empty(req):
        return _Response(200, "")

    _record(monkeypatch, empty)
    assert preemption.cloud_preemption_probe() is False


def test_the_imds_token_is_minted_once_and_reused(monkeypatch):
    # Re-minting per poll cost a link-local round trip on every poll of a spot worker's life.
    monkeypatch.setenv("BATCHER_PROVIDER", "aws")

    def imds(req):
        if req.full_url == preemption._IMDS_TOKEN_URL:
            return _Response(200, "tok")
        raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, None)

    tried = _record(monkeypatch, imds)
    for _ in range(3):
        preemption.cloud_preemption_probe()
    assert tried.count(preemption._IMDS_TOKEN_URL) == 1


def test_a_reseller_keeps_the_endpoint_its_hardware_answers_on(monkeypatch):
    # CoreWeave, Crusoe and the rest may run on hyperscaler capacity, exporting their own
    # marker over EC2 hardware. Dropping the AWS probe on the strength of the marker alone
    # would take away the only preemption notice such a fleet gets.
    from batcher._internal.site import provider

    monkeypatch.setenv("BATCHER_PROVIDER", "coreweave")
    monkeypatch.setattr(provider, "dmi_identity", lambda: ("aws", "m5.large", True))
    tried = _record(monkeypatch, _unreachable)
    assert preemption.cloud_preemption_probe() is False
    assert tried == [preemption._IMDS_TOKEN_URL, _AWS_URL]


def test_a_neocloud_on_its_own_hardware_probes_nothing(monkeypatch):
    # The firmware names a server vendor rather than a platform, so there is no endpoint to
    # try -- which is the case the scoping exists for.
    from batcher._internal.site import provider

    monkeypatch.setenv("BATCHER_PROVIDER", "coreweave")
    monkeypatch.setattr(provider, "dmi_identity", lambda: ("", "AS-4125GS", False))
    tried = _record(monkeypatch, _unreachable)
    assert preemption.cloud_preemption_probe() is False
    assert tried == []
