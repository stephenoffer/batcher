"""Shuffle output-byte tracking on the Flight server (network-egress metadata capture).

The stubs here replace the module's `engine()` accessor rather than individually-imported
native names: `server.py` reaches the compiled data plane only through
`batcher._internal.native.engine`, so that one seam is what a unit test substitutes. Binding
the native symbols at module import instead would make even the pure-Python `ShuffleTicket`
unimportable on a tree that has not been built.
"""

from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pytest

pytestmark = pytest.mark.unit


def _stub_engine(monkeypatch, srvmod, **attrs):
    """Point `srvmod.engine()` at a namespace exposing only `attrs` from the data plane."""
    monkeypatch.setattr(srvmod, "engine", lambda: SimpleNamespace(**attrs))


def test_flight_server_tracks_published_bytes(monkeypatch):
    from batcher.carbonite.transfer import server as srvmod

    class _StubSrv:  # avoid binding a real Flight port in a unit test
        def __init__(self, *args, **kwargs):
            self.addr = "127.0.0.1:0"

        def publish(self, ticket, batches):
            pass

    _stub_engine(monkeypatch, srvmod, FlightShuffleServer=_StubSrv)
    s = srvmod.FlightShuffleServer()
    assert s.bytes_published == 0  # nothing shuffled yet

    tbl = pa.table({"x": list(range(1000)), "y": list(range(1000))})
    n = sum(b.nbytes for b in tbl.to_batches())
    s.publish("ticket-1", tbl.to_batches())
    assert s.bytes_published == n  # one partition's egress measured
    s.publish("ticket-2", tbl.to_batches())
    assert s.bytes_published == 2 * n  # accumulates across partitions


def test_local_paths_track_locality_bytes(monkeypatch):
    from batcher.carbonite.transfer import server as srvmod

    tbl = pa.table({"x": list(range(200))})
    batches = tbl.to_batches()
    n = sum(b.nbytes for b in batches)

    class _StubSrv:
        def __init__(self, *args, **kwargs):
            self.addr = "127.0.0.1:0"

        def local_fetch(self, ticket):
            return batches

        def shm_fetch(self, addr, ticket):
            return batches

    _stub_engine(monkeypatch, srvmod, FlightShuffleServer=_StubSrv)
    s = srvmod.FlightShuffleServer()
    assert s.bytes_served_locally == 0
    s.local_fetch("t")  # DIRECT_MEMORY
    s.shm_fetch("peer", "t")  # SHARED_MEMORY
    assert s.bytes_served_locally == 2 * n  # same-node bytes (no network hop) measured
