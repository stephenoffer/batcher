"""The Flight transport's measurements reach the driver.

`Core measures, Kyber decides` is a contract, not a single-node convenience. The Flight
worker was the hole in it: it called the engine's *unmetered* entry point at every site, so
a distributed sort, join or window on the transport a real cluster actually uses taught the
cost model and the memory model nothing — while being the path that runs the largest inputs
and the one that spills.

A Flight worker is a long-lived actor whose methods return addresses, tickets and paths, so
unlike a disk-shuffle task it cannot hand its measurements back as a return value. The
driver drains them instead, which is what these cover. Everything here runs against a fake
`ray`, because the property under test is the plumbing, not the cluster.

The import of the module under test is at file scope on purpose: `batcher.dist` pulls in the
Flight worker, whose actor class is built by `@ray.remote` at *import* time. Importing it
after the stub is installed resolves that against the stub and raises a bare
`AttributeError` from a module that has nothing to do with what is being tested.
"""

from __future__ import annotations

import json

import pytest
from tests._fake_ray import install_fake_ray

from batcher.dist.executors.ray_runtime.metering import drain_worker_metrics

pytestmark = pytest.mark.unit


class _Handle:
    """A stand-in for one actor handle's `drain_metrics` remote method."""

    def __init__(self, documents: list[str] | None = None, dead: bool = False) -> None:
        self._documents = documents or []
        self._dead = dead
        self.drain_metrics = self

    def remote(self):
        if self._dead:
            raise RuntimeError("actor died after its work landed")
        return lambda: self._documents


class _Legacy:
    """A worker from an engine build that predates the drain — no such method."""


def _doc(kind: str, rows: int = 10) -> str:
    return json.dumps(
        {"ops": [{"op_id": 0, "kind": kind, "rows_in": rows, "rows_out": rows}], "query": {}}
    )


def test_every_worker_contributes_its_documents(monkeypatch):
    install_fake_ray(monkeypatch)

    out: list[dict] = []
    drain_worker_metrics([_Handle([_doc("sort")]), _Handle([_doc("sort")])], None, out)
    # A W-worker stage legitimately contributes W samples; they are not merged here.
    assert len(out) == 2
    assert [d["ops"][0]["kind"] for d in out] == ["sort", "sort"]


def test_a_dead_worker_costs_only_its_own_contribution(monkeypatch):
    install_fake_ray(monkeypatch)

    out: list[dict] = []
    drain_worker_metrics([_Handle([_doc("hash_join")]), _Handle(dead=True)], None, out)
    # The stage's rows are already computed by the time this runs. No statistic is worth
    # failing a finished query for.
    assert len(out) == 1


def test_a_worker_without_the_drain_is_skipped(monkeypatch):
    install_fake_ray(monkeypatch)

    out: list[dict] = []
    drain_worker_metrics([_Legacy(), _Handle([_doc("window")])], None, out)
    assert [d["ops"][0]["kind"] for d in out] == ["window"]


def test_draining_is_a_no_op_when_nothing_consumes_it(monkeypatch):
    install_fake_ray(monkeypatch)

    handle = _Handle([_doc("sort")])
    # With no hub and no profile channel there is nobody to hand the documents to, and the
    # round trip to every actor is pure cost — so it must not happen at all.
    drain_worker_metrics([handle], None, None)
    assert handle._documents == [_doc("sort")], "the actor should not have been asked"


def test_a_malformed_document_does_not_lose_the_others(monkeypatch):
    install_fake_ray(monkeypatch)

    out: list[dict] = []
    drain_worker_metrics([_Handle(["not json", _doc("distinct")])], None, out)
    assert [d["ops"][0]["kind"] for d in out] == ["distinct"]


def test_the_hub_learns_from_a_flight_stage(monkeypatch):
    install_fake_ray(monkeypatch)
    from batcher.metadata import MetadataHub
    from batcher.metadata.backends import InProcessBackend

    hub = MetadataHub(InProcessBackend())
    drain_worker_metrics([_Handle([_doc("sort"), _doc("hash_join")])], hub)
    # The point of the whole path: the operators a distributed run executed become the
    # observations Kyber calibrates its cost model from on the next run.
    assert {"sort", "hash_join"} <= set(hub.op_stats_by_kind())
