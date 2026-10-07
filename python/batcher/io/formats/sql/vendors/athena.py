"""Athena connection profile — the PyAthena keywords, named the way Athena names them.

Athena is reachable through any PEP 249 driver, so Batcher needs no Athena source of its
own: `DBAPISource` already reads it, pushes the projection and predicate into the SQL, and
probes the schema with a zero-row query. What a user lacks is the *spelling*. PyAthena wants
``s3_staging_dir``, ``work_group``, ``region_name`` and ``schema_name``; the console says
"query result location", "workgroup", "region" and "database". `athena_connect_kwargs` is
that translation and nothing else, so ``bt.read.athena`` is a thin route over the DB-API
source rather than a new framework.

Athena needs somewhere to write each query's result: an S3 output location, or a workgroup
whose configuration enforces one. Naming neither is refused here, before any query is
submitted, because Athena's own error for it arrives only after a round trip.

Not yet verified against a live Athena; see ``tests/PENDING_VERIFICATION.md``.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.errors import BackendError

__all__ = ["ATHENA_DRIVER", "athena_connect_kwargs"]

#: The PEP 249 module `DBAPISource` imports for an Athena read.
ATHENA_DRIVER = "pyathena"


def athena_connect_kwargs(
    *,
    region: str,
    workgroup: str | None = None,
    output_location: str | None = None,
    database: str | None = None,
    catalog: str | None = None,
    profile_name: str | None = None,
) -> dict[str, Any]:
    """PyAthena ``connect()`` keywords for an Athena profile.

    Args:
        region: The AWS region, e.g. ``"us-east-1"``.
        workgroup: The Athena workgroup to run in.
        output_location: The S3 location for query results (``s3://bucket/prefix/``).
            Optional when `workgroup` enforces its own.
        database: The default database (Glue schema) unqualified table names resolve in.
        catalog: The data catalog, when it is not ``AwsDataCatalog``.
        profile_name: A named AWS profile; omit to use the ambient credential chain.

    Returns:
        Keyword arguments for ``pyathena.connect``.

    Raises:
        BackendError: If neither `workgroup` nor `output_location` is given, or
            `output_location` is not an ``s3://`` URI.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.vendors import athena_connect_kwargs
            >>> athena_connect_kwargs(region="us-east-1", workgroup="analytics")
            {'region_name': 'us-east-1', 'work_group': 'analytics'}
    """
    if workgroup is None and output_location is None:
        raise BackendError(
            "an Athena read needs output_location= (an s3:// prefix for query results) or "
            "workgroup= naming a workgroup that enforces one."
        )
    if output_location is not None and not output_location.startswith("s3://"):
        raise BackendError(f"output_location={output_location!r} must be an s3:// URI")
    named = {
        "region_name": region,
        "work_group": workgroup,
        "s3_staging_dir": output_location,
        "schema_name": database,
        "catalog_name": catalog,
        "profile_name": profile_name,
    }
    return {key: value for key, value in named.items() if value is not None}
