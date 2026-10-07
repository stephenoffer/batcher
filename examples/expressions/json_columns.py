"""Reading JSON held in a string column, without parsing it in Python.

Semi-structured payloads arrive as text. The ``.json`` accessor runs a path query in the
engine and returns a typed column, so you can filter and aggregate on a nested field
without a ``json.loads`` per row.

    python examples/expressions/json_columns.py
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def main() -> None:
    events = bt.from_pydict(
        {
            "payload": [
                '{"user": {"id": 7, "name": "ada"}, "tags": ["a", "b"], "ok": true, "score": 1.5}',
                '{"user": {"id": 9, "name": "bob"}, "tags": [], "ok": false, "score": 0.25}',
            ],
        }
    )

    parsed = events.with_columns(
        # Typed extraction by JSONPath.
        name=col("payload").json.extract_string("$.user.name"),
        uid=col("payload").json.extract_int("$.user.id"),
        score=col("payload").json.extract_float("$.score"),
        ok=col("payload").json.extract_bool("$.ok"),
        # Shape queries.
        n_tags=col("payload").json.array_length("$.tags"),
        top_keys=col("payload").json.keys(),
        kind=col("payload").json.type_of("$.user"),
        # Presence, and a raw value as text.
        has_user=col("payload").json.exists("$.user"),
        has_missing=col("payload").json.exists("$.nope"),
        # `value` returns the raw JSON text of a *scalar* at that path.
        raw_score=col("payload").json.value("$.score"),
        # `structure` summarizes the document's shape and inferred types.
        shape=col("payload").json.structure(),
    )

    result = parsed.to_pydict()
    print(result)

    assert result["name"] == ["ada", "bob"]
    assert result["uid"] == [7, 9]
    assert result["score"] == [1.5, 0.25]
    assert result["ok"] == [True, False]
    assert result["n_tags"] == [2, 0]
    assert sorted(result["top_keys"][0]) == ["ok", "score", "tags", "user"]
    assert result["kind"][0] == "object"
    assert result["has_user"] == [True, True]
    assert result["has_missing"] == [False, False]
    assert result["raw_score"] == ["1.5", "0.25"]
    assert '"user"' in result["shape"][0]

    # The point: filter and aggregate on a nested field with no Python parsing.
    winners = (
        events.filter(col("payload").json.extract_bool("$.ok"))
        .select(name=col("payload").json.extract_string("$.user.name"))
        .to_pydict()
    )
    assert winners["name"] == ["ada"]

    # A key holding a dot is reachable only by quoting it; `$."a.b"` and `$['a.b']` agree.
    dotted = bt.from_pydict({"j": ['{"a.b": 1, "a": {"b": 2}}']})
    quoted = dotted.select(
        q=col("j").json.extract_int('$."a.b"'),
        nested=col("j").json.extract_int("$.a.b"),
    ).to_pydict()
    assert quoted == {"q": [1], "nested": [2]}

    # Decode the whole document into a typed struct once, instead of one extract per field.
    shape = pa.struct(
        [
            ("user", pa.struct([("id", pa.int64()), ("name", pa.string())])),
            ("tags", pa.list_(pa.string())),
        ]
    )
    typed = events.select(doc=col("payload").json.decode(shape))
    decoded = typed.select(
        uid=col("doc").struct.field("user").struct.field("id"),
        first_tag=col("doc").struct.field("tags").list.get(0),
    ).to_pydict()
    assert decoded == {"uid": [7, 9], "first_tag": ["a", None]}

    # ...edit it as a struct, and write it back out as JSON text.
    back = typed.select(j=col("doc").struct.drop_fields("tags").json.encode()).to_pydict()
    assert back["j"][0] == '{"user":{"id":7,"name":"ada"}}'

    # An RFC 7386 merge patch: `null` deletes a key, objects merge, the rest replaces. A key
    # the patch touches moves to the end, as in DuckDB's `json_merge_patch`.
    patched = events.select(
        p=col("payload").json.merge_patch('{"user": {"name": null}, "tags": null}')
    ).to_pydict()
    assert patched["p"][0] == '{"ok":true,"score":1.5,"user":{"id":7}}'


if __name__ == "__main__":
    main()
