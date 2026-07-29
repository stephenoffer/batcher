"""The repeated shuffle fetches must reuse their channel, not dial per call.

`server.fetch` is documented as a *one-shot* fetch that opens a fresh connection, with
`ShuffleClient` named as the form for repeated fetches. Two of the engine's most repeated
fetches used the one-shot form anyway: the GPU consumer of the streaming pipeline, which
calls it once per morsel, and the materialized-intermediate source an adaptive stage reads.
Neither shows up in a result — only as a connection handshake in front of every morsel.
"""

from __future__ import annotations

import inspect

from batcher.dist.fleet import source as fleet_source


def _body(fn) -> str:
    return inspect.getsource(fn)


def test_the_materialized_intermediate_reads_through_the_pooled_client():
    # `FlightFetchSplit` is the per-handle reader `FlightMaterializedSource` delegates to,
    # so it is where the fetch actually happens.
    body = _body(fleet_source.FlightFetchSplit.read)
    assert "process_client" in body
    assert "from batcher.carbonite.transfer.server import fetch" not in body


def test_the_streaming_gpu_consumer_fetches_through_the_pooled_client():
    """One fetch per morsel — the hottest fetch in the engine, and the one that was paying
    a full gRPC handshake ahead of each morsel it was meant to be overlapping."""
    import batcher.dist.executors.map as map_mod

    body = inspect.getsource(map_mod)
    run_split = body[body.index("def run_split(") :]
    run_split = run_split[: run_split.index("\n    def ")]
    assert "process_client" in run_split
    assert "import fetch" not in run_split


def test_the_pooled_client_is_one_per_process():
    from batcher.carbonite.transfer.lifecycle import process_client

    assert process_client() is process_client()


def test_the_pooled_client_reports_its_live_channels():
    from batcher.carbonite.transfer.lifecycle import process_client

    # No peers dialled in this test process, so the count is a number and not an error —
    # which is all this asserts; the count itself is exercised by the transport tests.
    assert isinstance(process_client().connection_count, int)
