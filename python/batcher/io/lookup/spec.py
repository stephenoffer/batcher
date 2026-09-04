"""Building a lookup backend on the worker, from a description that can travel to it.

A live Redis connection cannot be pickled, and a `rocksdict` handle cannot be shared
between processes at all — so what `map_batches` ships to its workers has to be a
*description* of the store, not the store. `build_lookup` is that description evaluated:
`Dataset.lookup_join` records the URI, the schema and the options, and each worker calls
this once when it constructs its enricher.

Which backend a URI names follows the same convention as the shared result cache, for the
same reason: one setting, and no second "which backend" option to keep consistent with it.

    redis:// · rediss:// · unix://   -> RedisLookup
    rocksdb://<path> or a bare path  -> RocksDBLookup
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError

__all__ = ["build_lookup", "lookup_schema"]

_REDIS_SCHEMES = ("redis://", "rediss://", "unix://")
_ROCKSDB_SCHEME = "rocksdb://"


def lookup_schema(schema: pa.Schema | dict[str, str] | None) -> pa.Schema:
    """Coerce a user-supplied lookup schema to an Arrow `Schema`.

    Required rather than inferred from the first batch's replies, and that is a deliberate
    cost: a join's output columns and their types cannot depend on which keys the first
    batch happened to contain, or a batch that matched nothing would produce a different
    schema from the batch before it — and under a distributed run, two workers would
    disagree about the shape of the same result.

    Args:
        schema: An Arrow schema, or a `{column: dtype_name}` mapping using the same dtype
            names `cast` accepts (``"int64"``, ``"string"``, ``"timestamp(us)"``).

    Returns:
        The resolved schema.

    Raises:
        PlanError: If `schema` is missing, empty, or names a dtype the engine does not
            know.
    """
    from batcher.plan.types import resolve_dtype

    if isinstance(schema, pa.Schema):
        if not len(schema):
            raise PlanError("lookup_join(): schema must name at least one column")
        return schema
    if not schema:
        raise PlanError(
            "lookup_join(): schema is required",
            hint=(
                "Name the columns the lookup contributes and their types, for example "
                "schema={'name': 'string', 'tier': 'int64'}. It cannot be inferred: a "
                "batch that matched nothing would otherwise have a different shape from "
                "the batch before it."
            ),
        )
    fields = []
    for name, dtype in schema.items():
        if isinstance(dtype, pa.DataType):
            resolved: pa.DataType | None = dtype
        else:
            resolved = resolve_dtype(str(dtype).lower())
        if resolved is None:
            raise PlanError(
                f"lookup_join(): unknown dtype {dtype!r} for column {name!r}",
                hint="Use the same dtype names cast() accepts, such as 'int64' or 'string'.",
            )
        fields.append(pa.field(name, resolved))
    return pa.schema(fields)


def build_lookup(uri: str, schema: pa.Schema, options: dict[str, Any]) -> Any:
    """Construct the `KeyValueLookup` that `uri` names. Called on the worker.

    Args:
        uri: The store's URI.
        schema: The columns the lookup contributes.
        options: Backend-specific options — ``prefix`` and ``hash_values`` for Redis.

    Returns:
        The live lookup.

    Raises:
        PlanError: If `uri` is not a string.
    """
    if not isinstance(uri, str) or not uri:
        raise PlanError(
            f"lookup_join(): expected a store URI, got {uri!r}",
            hint="Pass 'redis://host:port/db', 'rocksdb:///path/to/db', or a database path.",
        )
    if uri.startswith(_REDIS_SCHEMES):
        from batcher.io.lookup.backends import RedisLookup

        return RedisLookup(
            uri,
            schema,
            prefix=options.get("prefix", ""),
            hash_values=bool(options.get("hash_values", False)),
        )
    from batcher.io.lookup.backends import RocksDBLookup

    path = uri.removeprefix(_ROCKSDB_SCHEME)
    return RocksDBLookup(path, schema)
