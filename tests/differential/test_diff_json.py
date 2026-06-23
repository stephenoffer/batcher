"""JSON string extraction (`.json.extract_string`) — structural."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def test_json_extract_string():
    tbl = pa.table(
        {
            "j": pa.array(
                [
                    '{"name": "alice", "age": 30, "addr": {"city": "NYC"}}',
                    '{"name": "bob", "addr": {"city": "LA"}}',
                    "not json",
                    '{"name": "carol"}',
                    None,
                ]
            )
        }
    )
    out = (
        bt.from_arrow(tbl)
        .select(
            name=col("j").json.extract_string("$.name"),
            age=col("j").json.extract_string("$.age"),
            city=col("j").json.extract_string("$.addr.city"),
        )
        .collect()
        .to_pydict()
    )
    assert out["name"] == ["alice", "bob", None, "carol", None]
    assert out["age"] == ["30", None, None, None, None]  # number → text; missing → null
    assert out["city"] == ["NYC", "LA", None, None, None]
