"""Shared helpers for the Iceberg connector: the dependency gate, write tokens, and
the catalog identity key."""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.optional import require

__all__ = ["_catalog_key", "_new_write_token", "_require_pyiceberg", "_staged_schema"]


def _require_pyiceberg() -> None:
    """Raise `BackendError` if pyiceberg is not importable."""
    require("pyiceberg", feature="Iceberg support", provides="pyiceberg", extra="iceberg")


def _new_write_token() -> str:
    """A short token identifying one write, so staged file names differ between writes."""
    import uuid

    return uuid.uuid4().hex[:12]


def _staged_schema(path: str) -> pa.Schema:
    """The Arrow schema of a staged Parquet file (used to create the table if missing)."""
    import pyarrow.parquet as pq

    from batcher.io.filesystem import resolve_filesystem

    fs = resolve_filesystem(path)
    target = fs.native_read_target(path)
    if target is not None:
        pafs, in_path = target
        return pq.ParquetFile(in_path, filesystem=pafs).schema_arrow
    with fs.open(path) as fh:
        return pq.ParquetFile(fh).schema_arrow


#: Catalog-property name fragments that authenticate rather than identify.
#:
#: Excluded from the key for a concrete reason: `identity()` is *persisted* as the stats
#: key, so rotating a secret must not change which relation a table's learned statistics
#: belong to. Including them would orphan every accumulated statistic on each rotation,
#: silently returning the optimizer to cold estimates on whatever schedule the credentials
#: turn over — a regression that looks like nothing at all.
#:
#: Fragments, not exact names, because the Iceberg catalog vocabulary is hierarchical and
#: per-backend: ``s3.secret-access-key``, ``gcs.oauth2.token``, ``adls.sas-token``,
#: ``credential``, ``header.Authorization``.
_NON_IDENTIFYING_FRAGMENTS = (
    "secret",
    "credential",
    "token",
    "password",
    "access-key",
    "access_key",
    "api-key",
    "api_key",
    "authorization",
    "passphrase",
    "sas",
)


def _catalog_key(spec: Any) -> str:
    """A stable, non-secret key for a catalog spec, so two catalogs never share an entry.

    This used to be ``";".join(f"{k}={v}")`` over the whole property mapping — which put
    the catalog's **credentials in clear text into `identity()`**, and `identity()` is not
    a debugging string. It is the key a source's learned statistics are filed under, so it
    is written to the statistics store and lives there. A Unity Catalog or REST spec
    carries ``s3.secret-access-key`` and ``credential``; both were being persisted verbatim,
    and both then surfaced anywhere the stats store was inspected, dumped, or shipped.

    `connection_fingerprint` is the existing answer to exactly this and is reused rather
    than re-derived. It is `sha256`, deliberately, not `hash()`: Python salts `hash()` per
    process, so a key built on it would differ on every run and no statistic would ever be
    reused — the feedback loop would appear to work while never improving a plan.

    Credential-ish properties are dropped *before* hashing rather than merely hidden by it.
    Hashing alone would stop the leak but tie the key to the secret's value, so a rotation
    would silently orphan the table's statistics (see `_NON_IDENTIFYING_FRAGMENTS`). What
    remains — type, URI, warehouse — is what actually distinguishes one catalog from
    another, which is the whole job of the key.
    """
    if spec is None:
        return "default"
    if isinstance(spec, str):
        return spec
    if not isinstance(spec, dict):
        return str(spec)
    from batcher.io.formats.sql._common import connection_fingerprint

    identifying = {
        key: value
        for key, value in spec.items()
        if not any(fragment in key.lower() for fragment in _NON_IDENTIFYING_FRAGMENTS)
    }
    return connection_fingerprint(identifying)
