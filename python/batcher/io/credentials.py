"""Credential resolution for connectors, plus Databricks Unity Catalog vending.

**Credentials by reference.** Every connector password, token, API key, and
connection URI accepts ``env:NAME`` or ``file:PATH`` in place of the literal secret.
Only the *reference* is stored on the source object and pickled to workers; the secret
is read on the machine that opens the connection, from *its* environment or a mounted
secret file. This mirrors the crypto-key design in `plan.functions.security` and exists
for the same reason: a distributed query must not ship secrets over the wire, and a
secret that never enters the object cannot leak through a traceback, a log line, or a
pickled split.

Resolution is deliberately **lazy** — call `resolve_secret` at connect time, never in a
constructor. Resolving early would put the plaintext back on the object that gets
pickled, defeating the whole point.

Direct-read connectors (delta-rs, pyiceberg's pyarrow scan) need short-lived
cloud storage credentials scoped to a single table's storage location, rather
than broad ambient credentials. Databricks Unity Catalog vends exactly this via
its *temporary table credentials* API: given a table, it returns a presigned
storage URL plus the cloud-specific options (AWS keys, Azure SAS, GCP token) that
an Arrow-native reader can use to read the underlying Parquet directly — no Spark
cluster in the path.

`vend_unity_credentials` wraps that call and normalizes the per-cloud response
into a ``(storage_url, storage_options)`` pair that delta-rs / object-store
accept. The `databricks-sdk` import is deferred; a missing dependency or a vend
failure raises `BackendError`.
"""

from __future__ import annotations

import pathlib
from typing import Any

from batcher._internal.errors import BackendError
from batcher._internal.optional import require

__all__ = ["is_secret_ref", "resolve_secret", "vend_unity_credentials"]

#: Reference schemes resolved on the machine that opens the connection.
_SECRET_REF_SCHEMES = ("env:", "file:")


def is_secret_ref(value: str | None) -> bool:
    """Whether `value` is an ``env:NAME`` / ``file:PATH`` reference rather than a literal."""
    return isinstance(value, str) and value.startswith(_SECRET_REF_SCHEMES)


def resolve_secret(value: str | None, *, what: str = "credential") -> str | None:
    """Resolve an ``env:``/``file:`` reference to its secret; pass a literal through.

    Call this where the connection is opened, not where the source is built — see the
    module docstring. `None` and a plain literal are returned unchanged, so a connector
    that has not been migrated, and a user who passes a raw password, both keep working.

    Args:
        value: A literal secret, an ``env:NAME`` / ``file:PATH`` reference, or None.
        what: What is being resolved, for the error message (e.g. ``"ClickHouse password"``).

    Returns:
        The resolved secret, or `value` unchanged when it is not a reference.

    Raises:
        BackendError: If the reference cannot be resolved. The message names the
            *reference*, never the secret.
    """
    if not is_secret_ref(value):
        return value
    scheme, _, target = str(value).partition(":")
    if scheme == "env":
        import os

        resolved = os.environ.get(target)
        if resolved is None:
            raise BackendError(
                f"{what}: environment variable {target!r} is not set (referenced as {value!r})"
            )
        return resolved
    try:
        return pathlib.Path(target).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise BackendError(f"{what}: cannot read secret file {target!r}: {exc}") from exc


def _require_databricks_sdk() -> Any:
    """Import and return the Databricks `WorkspaceClient` class or raise."""
    return require(
        "databricks.sdk",
        "WorkspaceClient",
        feature="Unity Catalog credential vending",
        provides="the Databricks SDK",
        extra="databricks",
    )


def _storage_options_from_credentials(creds: Any) -> dict[str, str]:
    """Flatten a Unity `TemporaryTableCredentials` into delta-rs storage options.

    Unity returns exactly one cloud-specific credential block per response. We map
    each to the keys delta-rs (object_store) expects, leaving the others absent.
    """
    options: dict[str, str] = {}
    aws = getattr(creds, "aws_temp_credentials", None)
    if aws is not None:
        options["aws_access_key_id"] = aws.access_key_id
        options["aws_secret_access_key"] = aws.secret_access_key
        if getattr(aws, "session_token", None):
            options["aws_session_token"] = aws.session_token
        return options
    azure_sas = getattr(creds, "azure_user_delegation_sas", None)
    if azure_sas is not None:
        options["azure_storage_sas_token"] = azure_sas.sas_token
        return options
    azure_aad = getattr(creds, "azure_aad", None)
    if azure_aad is not None:
        options["azure_storage_token"] = azure_aad.aad_token
        return options
    gcp = getattr(creds, "gcp_oauth_token", None)
    if gcp is not None:
        options["google_service_account_token"] = gcp.oauth_token
        return options
    r2 = getattr(creds, "r2_temp_credentials", None)
    if r2 is not None:
        options["aws_access_key_id"] = r2.access_key_id
        options["aws_secret_access_key"] = r2.secret_access_key
        if getattr(r2, "session_token", None):
            options["aws_session_token"] = r2.session_token
        return options
    raise BackendError("Unity Catalog returned no recognized cloud credentials for the table")


def vend_unity_credentials(
    table: str,
    workspace: str,
    token: str,
    *,
    operation: str = "READ",
) -> tuple[str, dict[str, str]]:
    """Vend short-lived storage credentials for a Unity Catalog table.

    Calls the Databricks ``temporary_table_credentials`` API and returns the
    table's physical storage location together with the cloud storage options a
    delta-rs / object-store reader needs to read it directly.

    Args:
        table: The fully-qualified Unity table id (``catalog.schema.table``).
        workspace: The Databricks workspace URL (``https://<host>``).
        token: A Databricks personal-access / OAuth token for the workspace.
        operation: ``"READ"`` (default) or ``"READ_WRITE"`` — the access level
            requested for the vended credentials.

    Returns:
        ``(storage_url, storage_options)`` — the table's storage URL and a mapping
        suitable for ``DeltaTable(..., storage_options=...)``.

    Raises:
        BackendError: if the Databricks SDK is missing, the table is not found,
            or no recognized cloud credentials are returned.
    """
    workspace_client = _require_databricks_sdk()
    try:
        client = workspace_client(host=workspace, token=token)
        info = client.tables.get(full_name=table)
        creds = client.temporary_table_credentials.generate_temporary_table_credentials(
            operation=operation,
            table_id=info.table_id,
        )
    except BackendError:
        raise
    except Exception as exc:
        raise BackendError(
            f"failed to vend Unity Catalog credentials for {table!r}: {exc}"
        ) from exc
    storage_url = getattr(creds, "url", None) or getattr(info, "storage_location", None)
    if not storage_url:
        raise BackendError(f"Unity Catalog returned no storage location for {table!r}")
    return str(storage_url), _storage_options_from_credentials(creds)
