"""One declared Snowflake authentication strategy, carried unchanged to every worker.

``snowflake.connector.connect`` authenticates four ways, each spelled with different
keywords: a password; key-pair JWT (``authenticator="SNOWFLAKE_JWT"`` plus a private-key
file); OAuth (``authenticator="oauth"`` plus a token); and single sign-on through a browser
(``authenticator="externalbrowser"``). Getting the combination wrong is a connector error
at connect time, often on a worker, long after the plan was built.

`snowflake_options` lets a caller say which one they mean — ``auth="key_pair"`` — with the
role, warehouse, database and schema beside it, validates the combination once on the
driver, and folds it into the single ``connection_kwargs`` dict the source and sink already
carry. That dict is what a worker connects with, so a local run and a distributed one use
the same declared strategy by construction. Secrets stay references (``env:``, ``file:``,
...) until `snowflake._connect` resolves them on the machine that dials.

Browser SSO cannot run on a headless worker, so `SnowflakeSink` refuses it for a
multi-shard write; a distributed *read* is unaffected, because only the driver connects and
the workers fetch result chunks through pre-signed URLs.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.errors import BackendError

__all__ = ["AUTH_METHODS", "is_browser_auth", "snowflake_options"]

#: The strategies `auth=` accepts.
AUTH_METHODS = ("password", "key_pair", "oauth", "externalbrowser")

#: Session keywords passed to the connector under their own names.
_SESSION = ("account", "user", "role", "warehouse", "database", "schema")

#: Credential keywords consumed by the strategy.
_CREDENTIALS = ("password", "private_key_file", "private_key_file_pwd", "token")

#: The connector's ``authenticator`` value per strategy; password needs none.
_AUTHENTICATOR = {
    "key_pair": "SNOWFLAKE_JWT",
    "oauth": "oauth",
    "externalbrowser": "externalbrowser",
}

#: What each strategy must be given, beyond ``account``.
_REQUIRED = {
    "password": ("user", "password"),
    "key_pair": ("user", "private_key_file"),
    "oauth": ("token",),
    "externalbrowser": ("user",),
}


def _infer(given: dict[str, Any]) -> str | None:
    """The strategy implied by the credentials present, when `auth=` was not named."""
    if "private_key_file" in given:
        return "key_pair"
    if "token" in given:
        return "oauth"
    if "password" in given:
        return "password"
    return None


def snowflake_options(opts: dict[str, Any]) -> dict[str, Any]:
    """Fold the unified auth and session keywords in `opts` into ``connection_kwargs``.

    Keywords that are not Snowflake connection options pass through untouched, so this can
    sit in front of any reader or writer option dict. A caller who passes none of them gets
    `opts` back as it was, which keeps an explicit ``connection_kwargs=`` working unchanged.

    Args:
        opts: Reader or writer options. Consumes ``auth``, ``account``, ``user``,
            ``password``, ``private_key_file``, ``private_key_file_pwd``, ``token``,
            ``role``, ``warehouse``, ``database`` and ``schema``; merges them into
            ``connection_kwargs``.

    Returns:
        The options with a merged ``connection_kwargs``.

    Raises:
        BackendError: If `auth` is unknown, a strategy is missing what it needs, a
            credential belongs to a different strategy, or a keyword contradicts the same
            key in an explicit ``connection_kwargs``.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.vendors import snowflake_options
            >>> out = snowflake_options(
            ...     {"account": "acme", "user": "etl", "auth": "key_pair",
            ...      "private_key_file": "/keys/etl.p8", "warehouse": "WH"}
            ... )
            >>> sorted(out["connection_kwargs"])
            ['account', 'authenticator', 'private_key_file', 'user', 'warehouse']
    """
    remaining = dict(opts)
    auth = remaining.pop("auth", None)
    given = {k: remaining.pop(k) for k in (*_SESSION, *_CREDENTIALS) if k in remaining}
    if auth is None and not given:
        return remaining
    if auth is None:
        auth = _infer(given) or "password"
    if auth not in AUTH_METHODS:
        raise BackendError(
            f"auth={auth!r} is not a Snowflake strategy; expected one of {list(AUTH_METHODS)}."
        )
    explicit = dict(remaining.pop("connection_kwargs", None) or {})
    merged = {**explicit, **given}
    missing = [k for k in ("account", *_REQUIRED[auth]) if not merged.get(k)]
    if missing:
        raise BackendError(f"Snowflake auth={auth!r} needs {', '.join(missing)}.")
    stray = [k for k in _CREDENTIALS if k in given and k not in _allowed(auth)]
    if stray:
        raise BackendError(
            f"Snowflake auth={auth!r} does not use {', '.join(stray)}; drop it, or name the "
            "strategy it belongs to with auth=."
        )
    clashes = [k for k in given if k in explicit and explicit[k] != given[k]]
    if clashes:
        raise BackendError(
            f"{', '.join(clashes)} given both as a keyword and in connection_kwargs with "
            "different values; pass it once."
        )
    if auth in _AUTHENTICATOR:
        merged["authenticator"] = _AUTHENTICATOR[auth]
    remaining["connection_kwargs"] = merged
    return remaining


def _allowed(auth: str) -> tuple[str, ...]:
    """The credential keywords strategy `auth` consumes."""
    if auth == "key_pair":
        return ("private_key_file", "private_key_file_pwd")
    return {"password": ("password",), "oauth": ("token",)}.get(auth, ())


def is_browser_auth(connection_kwargs: dict[str, Any]) -> bool:
    """Whether these kwargs authenticate through a browser, which a worker cannot open.

    Args:
        connection_kwargs: Snowflake connector kwargs.

    Returns:
        True for ``authenticator="externalbrowser"``.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.vendors.snowflake_auth import is_browser_auth
            >>> is_browser_auth({"authenticator": "EXTERNALBROWSER"})
            True
    """
    return str(connection_kwargs.get("authenticator", "")).lower() == "externalbrowser"
