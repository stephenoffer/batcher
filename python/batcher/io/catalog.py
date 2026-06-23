"""Unified lakehouse catalog resolver.

A *catalog* is the metadata service that maps a table identifier
(``namespace.table``) to its current snapshot/metadata location. Batcher speaks
to catalogs only through `pyiceberg`'s `load_catalog`, which already abstracts the
concrete backend behind a ``type`` discriminator: Iceberg REST, Databricks Unity
Catalog, and Snowflake Polaris are all REST catalogs; Glue, Hive, SQL/JDBC, and
DynamoDB are their own pyiceberg types.

`resolve_catalog` normalizes a small, friendly spec into the exact
``load_catalog(name, **props)`` call pyiceberg expects, so the Iceberg connector
(and any future catalog-backed connector) has one place to obtain a live catalog.

The `pyiceberg` import is deferred: importing this module never requires the
optional dependency. A missing dependency raises `BackendError` with an
actionable ``pip install`` hint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import BackendError

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog

__all__ = ["CatalogSpec", "resolve_catalog"]

# Friendly aliases → the pyiceberg ``type`` discriminator. REST-based services
# (Unity Catalog, Polaris, Tabular, Lakekeeper, …) all surface as ``rest``.
_TYPE_ALIASES: dict[str, str] = {
    "rest": "rest",
    "iceberg-rest": "rest",
    "unity": "rest",
    "unity-catalog": "rest",
    "databricks": "rest",
    "polaris": "rest",
    "snowflake": "rest",
    "glue": "glue",
    "hive": "hive",
    "hive-metastore": "hive",
    "sql": "sql",
    "jdbc": "sql",
    "dynamodb": "dynamodb",
    "in-memory": "in-memory",
}

# A catalog spec is a mapping of properties; ``type`` (and optional ``name``)
# select the backend, the rest are passed through to pyiceberg verbatim.
CatalogSpec = dict[str, Any]


def _require_pyiceberg() -> Any:
    """Import and return the `pyiceberg.catalog` module or raise `BackendError`."""
    try:
        from pyiceberg import catalog
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BackendError(
            "Iceberg catalogs require pyiceberg: pip install 'batcher-engine[iceberg]'"
        ) from exc
    return catalog


def _normalize_type(raw: str) -> str:
    """Map a friendly catalog-type alias to pyiceberg's ``type`` discriminator."""
    key = raw.strip().lower()
    try:
        return _TYPE_ALIASES[key]
    except KeyError:
        known = ", ".join(sorted(_TYPE_ALIASES))
        raise BackendError(f"unknown catalog type {raw!r}; supported: {known}") from None


def _coerce_spec(spec: CatalogSpec | str) -> CatalogSpec:
    """Accept either a full property mapping or a bare named catalog string.

    A bare string names a catalog already configured in ``~/.pyiceberg.yaml`` or
    the environment, so it is passed through with no inline properties.
    """
    if isinstance(spec, str):
        return {"name": spec}
    if not isinstance(spec, dict):
        raise BackendError(f"catalog spec must be a str or dict, got {type(spec).__name__}")
    return dict(spec)


def resolve_catalog(spec: CatalogSpec | str) -> Catalog:
    """Load a live pyiceberg `Catalog` from a friendly `spec`.

    Args:
        spec: Either the name of a catalog configured in ``~/.pyiceberg.yaml`` /
            environment, or a property mapping. Recognized keys:

            - ``name``: the catalog name (default ``"default"``).
            - ``type``: a friendly alias (``"rest"``, ``"unity"``, ``"polaris"``,
              ``"snowflake"``, ``"glue"``, ``"hive"``, ``"sql"``/``"jdbc"``,
              ``"dynamodb"``) normalized to pyiceberg's discriminator. Optional
              when the named catalog is fully configured externally.
            - ``uri``, ``warehouse``, ``credential``, ``token``, and any other
              key are passed through to pyiceberg unchanged.

    Returns:
        A connected `pyiceberg.catalog.Catalog`.

    Raises:
        BackendError: if pyiceberg is not installed, the type alias is unknown,
            or pyiceberg fails to construct/connect the catalog.

    Examples:
        >>> resolve_catalog(  # doctest: +SKIP
        ...     {"type": "unity", "uri": "https://dbc.cloud.databricks.com/api/2.1/unity-catalog/iceberg",
        ...      "token": "dapi...", "warehouse": "my_catalog"}
        ... )
    """
    catalog_mod = _require_pyiceberg()
    props = _coerce_spec(spec)
    name = props.pop("name", "default")
    if "type" in props:
        props["type"] = _normalize_type(str(props["type"]))
    try:
        return catalog_mod.load_catalog(name, **props)
    except Exception as exc:
        raise BackendError(f"failed to load catalog {name!r}: {exc}") from exc
