"""Live smoke test: a real Flight SQL driver (ADBC) against `batcher.integrations.flightsql`.

Skipped unless ``BATCHER_LIVE_FLIGHTSQL=1`` and ``adbc_driver_flightsql`` is installed (the
``sql`` extra). Run:
``BATCHER_LIVE_FLIGHTSQL=1 pytest tests/integration/live/test_live_flightsql.py``.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("BATCHER_LIVE_FLIGHTSQL"), reason="set BATCHER_LIVE_FLIGHTSQL=1"
    ),
]


def test_adbc_runs_a_bound_query_against_the_service() -> None:
    adbc = pytest.importorskip("adbc_driver_flightsql.dbapi")
    import batcher as bt
    from batcher.integrations import flightsql

    session = bt.Session()
    session.register("t", bt.from_pydict({"id": [1, 2, 3]}))
    server = flightsql.serve(session, auth="live-token")
    try:
        with (
            adbc.connect(
                f"grpc://127.0.0.1:{server.port}",
                db_kwargs={"adbc.flight.sql.authorization_header": "Bearer live-token"},
            ) as conn,
            conn.cursor() as cur,
        ):
            cur.execute("SELECT id FROM t WHERE id > ? ORDER BY id", parameters=(1,))
            assert cur.fetchall() == [(2,), (3,)]
    finally:
        server.shutdown()
