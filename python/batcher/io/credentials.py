"""Credential resolution for connectors, plus Databricks Unity Catalog vending.

**Credentials by reference.** Every connector password, token, API key, connection URI and
object-store option accepts ``env:NAME``, ``file:PATH`` or ``cmd:NAME`` in place of the
literal secret. Only the *reference* is stored on the source object and pickled to workers;
the secret is read on the machine that opens the connection, from *its* environment, a
mounted secret file, or the operator's own secret-fetching helper. This mirrors the crypto-key
design in `plan.functions.security` — and it is the same three schemes the data plane's own
resolver (`bc-secrets`) answers, so one vocabulary covers both sides. It exists for the reason
that one does: a distributed query must not ship secrets over the wire, and a secret that
never enters the object cannot leak through a traceback, a log line, or a pickled split.

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
from dataclasses import dataclass, field
from typing import Any

from batcher._internal.errors import BackendError
from batcher._internal.optional import require
from batcher.io.secret_backends import BACKEND_SCHEMES, resolve_backend_ref

__all__ = [
    "SECRET_COMMAND_ENV",
    "UnityLease",
    "is_secret_ref",
    "resolve_client_secrets",
    "resolve_secret",
    "unity_lease",
    "vend_unity_credentials",
]

#: Reference schemes resolved on the machine that opens the connection. The first three are
#: the ones the data plane's own resolver (`bc-secrets`) answers, so one vocabulary covers a
#: connector password, a storage option, and an expression-level encryption key. The rest
#: name a key store directly and are answered by `secret_backends`; the data plane reaches
#: the same stores through `cmd:`, which is why it needs no matching vocabulary.
_SECRET_REF_SCHEMES = ("env:", "file:", "cmd:", *BACKEND_SCHEMES)

#: The operator-configured program `cmd:NAME` runs, with `NAME` as its single argument.
#:
#: **`cmd:` is inert unless this is set, and the reference supplies only the argument, never
#: the program.** That asymmetry is the whole security story, and it is the same one
#: `bc-secrets` states: a plan is data, it may arrive from somewhere less trusted than the
#: cluster, and letting it name a program to execute would turn a secret reference into
#: arbitrary code execution. The operator chooses the program; the plan chooses which secret
#: to ask that program for.
#:
#: One knob reaches `vault kv get`, `aws secretsmanager get-secret-value`,
#: `gcloud secrets versions access`, `az keyvault secret show`, or a bespoke fetcher — which
#: is what lets a connector on a neocloud or an on-prem cluster use a key store this package
#: takes no dependency on.
SECRET_COMMAND_ENV = "BATCHER_SECRET_COMMAND"

#: How long the helper may take. Not configurable, because the failure it guards against is a
#: helper that never returns — which would hang the control plane on a connect with no
#: diagnosis — and no real key-store fetch takes anywhere near this long.
_SECRET_COMMAND_TIMEOUT_S = 30.0


def is_secret_ref(value: str | None) -> bool:
    """Whether `value` is a secret reference (``env:``, ``file:``, ``cmd:``, or a key store)."""
    return isinstance(value, str) and value.startswith(_SECRET_REF_SCHEMES)


def resolve_secret(value: str | None, *, what: str = "credential") -> str | None:
    """Resolve a secret reference to its secret material; pass a literal through.

    Call this where the connection is opened, not where the source is built — see the
    module docstring. `None` and a plain literal are returned unchanged, so a connector
    that has not been migrated, and a user who passes a raw password, both keep working.

    Args:
        value: A literal secret, None, or a reference — ``env:NAME``, ``file:PATH``,
            ``cmd:NAME``, or a key store (``vault:``, ``aws-sm:``, ``aws-ssm:``,
            ``gcp-sm:``, ``azure-kv:``); see `secret_backends`.
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
    if scheme == "cmd":
        return _from_command(target, what=what, reference=str(value))
    if f"{scheme}:" in BACKEND_SCHEMES:
        return resolve_backend_ref(scheme, target, what=what, reference=str(value))
    try:
        return pathlib.Path(target).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise BackendError(f"{what}: cannot read secret file {target!r}: {exc}") from exc


def _from_command(name: str, *, what: str, reference: str) -> str:
    """Run the operator's secret-fetching helper for `name` and take its stdout.

    Never resolved through a shell, and the program is never taken from the reference — see
    `SECRET_COMMAND_ENV`. Deliberately not cached: this is called once when a connection is
    opened, not per batch the way the data plane's resolver is, so a cache would buy nothing
    and would keep a rotated secret alive past its rotation.

    Every failure names the reference and the helper's *stderr*, never its stdout, because
    stdout is the secret.
    """
    import os
    import shlex
    import subprocess

    command = os.environ.get(SECRET_COMMAND_ENV, "").strip()
    if not command:
        raise BackendError(
            f"{what}: {reference!r} uses the `cmd:` scheme, but {SECRET_COMMAND_ENV} is not "
            "set; an operator must configure the secret-fetching program (the reference "
            "supplies only its argument, never the program itself)"
        )
    argv = [*shlex.split(command), name]
    try:
        # An argv list, never a shell string: the reference is data and must not be able to
        # inject a second command through a shell metacharacter.
        done = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_SECRET_COMMAND_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BackendError(f"{what}: secret command for {reference!r} failed: {exc}") from exc
    if done.returncode != 0:
        detail = done.stderr.strip()[:400] or f"exit status {done.returncode}"
        raise BackendError(f"{what}: secret command for {reference!r} failed: {detail}")
    secret = done.stdout.strip()
    if not secret:
        raise BackendError(f"{what}: secret command for {reference!r} returned nothing")
    return secret


def resolve_client_secrets(
    options: dict[str, Any], *, what: str, hints: tuple[str, ...]
) -> dict[str, Any]:
    """Resolve secret references in a client-options dict's credential values.

    A broker client (librdkafka, the Event Hubs SDK) takes its options verbatim, so a
    ``sasl_password="env:KAFKA_PASSWORD"`` used to reach it as the literal string
    ``env:KAFKA_PASSWORD`` and fail authentication -- the one credential in the engine that
    the common secret vocabulary did not cover. Call this where the client is built, on the
    worker, so the reference is what travels and the secret exists only there.

    Args:
        options: Client options, keyed by option name in either spelling (``_`` or ``.``).
        what: A label for error messages, naming the connector.
        hints: Substrings that mark an option name as a credential. Only a value under
            such a key is resolved, so a path option that happens to start with ``file:``
            is never read as a secret. Pass the connector family's own list, the one its
            log redaction already uses, so the two cannot disagree.

    Returns:
        A copy of `options` with each credential-keyed secret reference resolved.

    Examples:
        .. doctest::

            >>> import os
            >>> from batcher.io.credentials import resolve_client_secrets
            >>> os.environ["DOC_KAFKA_PW"] = "hunter2"
            >>> resolve_client_secrets(
            ...     {"sasl.password": "env:DOC_KAFKA_PW"}, what="kafka", hints=("password",)
            ... )
            {'sasl.password': 'hunter2'}
    """
    out = dict(options)
    for key, value in options.items():
        lowered = str(key).lower()
        if is_secret_ref(value) and any(hint in lowered for hint in hints):
            out[key] = resolve_secret(value, what=f"{what} {key}")
    return out


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


@dataclass(frozen=True, slots=True)
class UnityLease:
    """Vended Unity Catalog storage credentials and when they stop working.

    `expires_at_s` is a Unix time in seconds, or ``None`` when Unity reported no expiry, in
    which case nothing can tell a reader when to renew and it is not attempted.
    """

    storage_url: str
    storage_options: dict[str, str] = field(repr=False)
    expires_at_s: float | None = None


def unity_lease(
    table: str,
    workspace: str,
    token: str,
    *,
    operation: str = "READ",
) -> UnityLease:
    """Vend short-lived storage credentials for a Unity Catalog table, with their expiry.

    `token` may be a secret reference (``env:``, ``file:``, a key store), resolved here, on
    whichever machine is vending -- which is what lets a worker renew an expiring lease
    without the driver shipping it a literal token.

    Args:
        table: The fully-qualified Unity table id (``catalog.schema.table``).
        workspace: The Databricks workspace URL (``https://<host>``).
        token: A Databricks token for the workspace, or a reference to one.
        operation: ``"READ"`` (default) or ``"READ_WRITE"``.

    Returns:
        The table's storage URL, the storage options a delta-rs reader needs, and the
        credentials' expiry.

    Raises:
        BackendError: If the Databricks SDK is missing, the table is not found, or no
            recognized cloud credentials are returned.
    """
    workspace_client = _require_databricks_sdk()
    resolved = resolve_secret(token, what="Databricks token") or ""
    try:
        client = workspace_client(host=workspace, token=resolved)
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
    # Unity reports `expiration_time` in epoch milliseconds.
    expires_ms = getattr(creds, "expiration_time", None)
    expires_at = float(expires_ms) / 1000.0 if isinstance(expires_ms, int | float) else None
    return UnityLease(str(storage_url), _storage_options_from_credentials(creds), expires_at)


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
    delta-rs / object-store reader needs to read it directly. `unity_lease` is the same
    call with the credentials' expiry attached.

    Args:
        table: The fully-qualified Unity table id (``catalog.schema.table``).
        workspace: The Databricks workspace URL (``https://<host>``).
        token: A Databricks personal-access / OAuth token for the workspace, or a
            secret reference to one.
        operation: ``"READ"`` (default) or ``"READ_WRITE"`` — the access level
            requested for the vended credentials.

    Returns:
        ``(storage_url, storage_options)`` — the table's storage URL and a mapping
        suitable for ``DeltaTable(..., storage_options=...)``.

    Raises:
        BackendError: if the Databricks SDK is missing, the table is not found,
            or no recognized cloud credentials are returned.
    """
    lease = unity_lease(table, workspace, token, operation=operation)
    return lease.storage_url, lease.storage_options
