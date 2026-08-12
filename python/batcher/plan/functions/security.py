"""Data-protection functions: `mask`, `hmac_sha256`, `aes_encrypt`, `aes_decrypt`.

The three ways to make a sensitive column safe to read, in increasing order of what
they preserve:

===============  ==================  ===========  ==============================
Function         Reversible?         Joinable?    Use it for
===============  ==================  ===========  ==============================
`mask`           no                  no           partial disclosure to humans
`hmac_sha256`    no                  yes          pseudonymized analytics
`aes_encrypt`    yes, with the key   yes          data that must be read back
===============  ==================  ===========  ==============================

"Joinable" means equal inputs produce equal outputs, so the protected column still
groups and equi-joins. That determinism is required of any expression — the
sequential interpreter is the correctness oracle the parallel executor and the JIT
are checked against, and a randomized output would put them at odds — and it is what
makes a protected column analytically useful at all. Its cost is that equality is
observable: an attacker who sees the protected column learns which rows share a value.
Where that is unacceptable, drop the column instead of protecting it.

These are the primitives that `batcher.governance` column-masking policies lower to;
they are also usable directly.

**Keys by reference (recommended).** Pass ``key="env:NAME"`` or ``key="file:PATH"``
instead of the raw key. Only the reference travels in the plan IR — plan logs, the
profile, ``explain()``, and the FFI boundary never see the secret. The data plane reads
the key on the machine that runs the query (from *its* environment or a mounted secret
file), so a distributed query resolves the key on each worker and never ships it over
the wire. An inline literal still works for local development but emits a
`SecurityWarning`, because it embeds the secret in the query and its serialized plan.
"""

from __future__ import annotations

import base64
import binascii
import warnings

from batcher._internal.errors import PlanError, SecurityWarning
from batcher.config.env import env_flag
from batcher.plan.expr_ir.core import Expr, IntoExpr, _wrap
from batcher.plan.expr_ir.func_nodes import StrFunc

__all__ = ["aes_decrypt", "aes_encrypt", "hmac_sha256", "mask"]

_AES_KEY_BYTES = 32

#: Key-reference schemes resolved by the data plane at execution time (never in Python).
#:
#: **This list is a two-sided contract with `bc-secrets`.** Python only decides whether a
#: key is a *reference* (pass it through) or an *inline literal* (warn, and validate that
#: it decodes to 32 bytes); the resolution happens in Rust. A scheme missing here is not
#: merely unsupported — it is validated as raw key material and rejected at plan-build
#: time, so the engine never gets the chance to resolve it. Add a backend in
#: `bc-secrets` and a scheme here in the same change.
_KEY_REF_SCHEMES = ("env:", "file:", "cmd:")


def _is_key_ref(key: str) -> bool:
    """Whether `key` is a reference (`env:NAME` / `file:PATH`) the engine resolves.

    A reference travels in the plan IR verbatim; the raw secret is read on the executing
    node from the environment or a mounted file. A bare value is an inline literal.
    """
    return isinstance(key, str) and key.startswith(_KEY_REF_SCHEMES)


#: Env var that upgrades the inline-key warning to a hard error. Set it (to any truthy
#: value) in a deployment where a key must never enter a plan.
_REQUIRE_KEY_REFS_VAR = "BATCHER_REQUIRE_KEY_REFS"


def _require_key_refs() -> bool:
    """Whether inline keys are forbidden outright in this deployment."""
    return env_flag(_REQUIRE_KEY_REFS_VAR)


def _warn_inline_key(func: str) -> None:
    """Warn that an inline key is embedded in the query and its serialized plan.

    Under `BATCHER_REQUIRE_KEY_REFS` this raises instead. A warning is the right default
    (an inline key is legitimate in a notebook or a test), but it is a weak control for a
    regulated deployment: `SecurityWarning` is a `UserWarning`, so Python prints it once
    per location and any caller that has filtered warnings never sees it at all — while
    the key still travels verbatim in the IR JSON, into `explain(format="json")`, into the
    plan fingerprint, and out to every Ray worker the plan is shipped to. The env var lets
    an operator make that unrepresentable rather than merely discouraged.
    """
    if _require_key_refs():
        raise PlanError(
            f"{func}(): an inline key is refused because {_REQUIRE_KEY_REFS_VAR} is set. "
            f"Use a reference — key='env:MY_KEY' or key='file:/run/secrets/key' — which "
            f"the engine resolves on the executing node, so the secret never enters the "
            f"plan, its serialized IR, or any explain/profile output."
        )
    warnings.warn(
        f"{func}(): an inline key is embedded in the query plan (and any plan log / "
        f"profile / explain output). Prefer a reference — key='env:MY_KEY' or "
        f"key='file:/run/secrets/key' — which the engine resolves at execution time so "
        f"the secret never enters the plan.",
        SecurityWarning,
        stacklevel=3,
    )


def _as_text(e: IntoExpr) -> Expr:
    """Coerce to text, the domain every data-protection function operates on.

    Matches `concat`'s implicit cast: protecting a numeric account number is exactly as
    ordinary as protecting an email, and requiring an explicit `.cast("string")` at
    every call site would be noise.
    """
    return _wrap(e).cast("string")


def _validated_key(func: str, key: str) -> str:
    """Return the `key` (or key reference) to store in the plan IR, validated.

    A reference (`env:`/`file:`) is passed through verbatim — it is resolved on the
    executing node, so it cannot be validated here (the secret may not be present on the
    machine building the plan). An inline literal is validated to decode to 32 bytes now,
    at plan-build time, so a bad key fails before the query is optimized and admitted
    rather than from inside a scan; and a `SecurityWarning` flags that the secret is being
    embedded in the plan. The `PlanError` deliberately does not quote the key.
    """
    if _is_key_ref(key):
        return key
    _warn_inline_key(func)
    if not isinstance(key, str) or not key:
        raise PlanError(f"{func}(): key must be a non-empty string")
    try:
        raw = (
            binascii.unhexlify(key)
            if len(key) == _AES_KEY_BYTES * 2 and all(c in "0123456789abcdefABCDEF" for c in key)
            else base64.b64decode(key, validate=True)
        )
    except (binascii.Error, ValueError) as exc:
        raise PlanError(
            f"{func}(): key must be {_AES_KEY_BYTES} bytes, given as "
            f"{_AES_KEY_BYTES * 2} hex characters or as base64"
        ) from exc
    if len(raw) != _AES_KEY_BYTES:
        raise PlanError(f"{func}(): key must decode to {_AES_KEY_BYTES} bytes, got {len(raw)}")
    return key


def mask(e: IntoExpr, *, show_first: int = 0, show_last: int = 0, char: str = "X") -> Expr:
    """Replace each character of a string with `char`, optionally revealing its ends.

    Irreversible and unkeyed — the tool for showing a human just enough of a value to
    recognize it ("the card ending 1234"). The result has the same character length as
    the input, and when `show_first` and `show_last` together cover the whole value
    nothing is masked. Null → null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"card": ["4111111111111234"]})
            >>> ds.select(m=bt.mask(bt.col("card"), show_last=4)).to_pydict()
            {'m': ['XXXXXXXXXXXX1234']}

            >>> ds.select(m=bt.mask(bt.col("card"), show_first=2, char="*")).to_pydict()
            {'m': ['41**************']}

    Args:
        e: The column or expression to mask; cast to text first.
        show_first: Number of leading characters left in the clear.
        show_last: Number of trailing characters left in the clear.
        char: The single replacement character.

    Returns:
        A text expression of the same character length as `e`.

    Raises:
        PlanError: If `char` is not exactly one character, or a reveal count is negative.
    """
    if len(char) != 1:
        raise PlanError(f"mask(): char must be exactly one character, got {char!r}")
    if show_first < 0 or show_last < 0:
        raise PlanError("mask(): show_first and show_last must be non-negative")
    return StrFunc("mask", _as_text(e), pattern=char, start=show_first, length=show_last)


def hmac_sha256(e: IntoExpr, key: str) -> Expr:
    """Pseudonymize a value as the lowercase-hex HMAC-SHA-256 of its text, keyed by `key`.

    The right default for analytics over sensitive identifiers. It is deterministic, so
    a pseudonymized ``user_id`` still joins across tables and still counts distinct; it
    is irreversible; and unlike a bare :meth:`~batcher.Expr.str.sha256` it resists the
    dictionary attack that a low-entropy domain (emails, phone numbers, national IDs)
    otherwise invites, because the attacker does not hold the key. Null → null.

    Unlike `aes_encrypt`, `key` here is raw key material of any length — it is an HMAC
    key, not an AES key, so it is not required to be 32 bytes.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"email": ["a@x.com", "a@x.com", "b@x.com"]})
            >>> out = ds.select(p=bt.hmac_sha256(bt.col("email"), key="s3cret")).to_pydict()
            >>> out["p"][0] == out["p"][1] != out["p"][2]  # stable, so it still joins
            True
            >>> len(out["p"][0])  # 32 bytes of digest as hex
            64

    Args:
        e: The column or expression to pseudonymize; cast to text first.
        key: Secret key material. See the module warning about key handling.

    Returns:
        A 64-character lowercase-hex text expression.

    Raises:
        PlanError: If `key` is empty.
    """
    if not key:
        raise PlanError("hmac_sha256(): key must be a non-empty string")
    if not _is_key_ref(key):
        _warn_inline_key("hmac_sha256")
    return StrFunc("hmac_sha256", _as_text(e), pattern=key)


def aes_encrypt(e: IntoExpr, key: str) -> Expr:
    """Encrypt a value with AES-256-GCM-SIV under `key`, as base64 of ciphertext‖tag.

    Reversible with `aes_decrypt` and the same key — the choice when the data must be
    read back (a downstream job with the key) rather than merely analyzed. Encryption is
    deterministic, so an encrypted column still joins and groups; the price is that it
    reveals which rows share a value. Null → null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> key = "00" * 32  # 32 bytes as hex; use a secret manager in production
            >>> ds = bt.from_pydict({"ssn": ["123-45-6789"]})
            >>> enc = ds.select(c=bt.aes_encrypt(bt.col("ssn"), key))
            >>> enc.select(s=bt.aes_decrypt(bt.col("c"), key)).to_pydict()
            {'s': ['123-45-6789']}

    Args:
        e: The column or expression to encrypt; cast to text first.
        key: A 32-byte key as 64 hex characters or as base64.

    Returns:
        A base64 text expression.

    Raises:
        PlanError: If `key` does not decode to exactly 32 bytes.
    """
    return StrFunc("aes_encrypt", _as_text(e), pattern=_validated_key("aes_encrypt", key))


def aes_decrypt(e: IntoExpr, key: str) -> Expr:
    """Decrypt an `aes_encrypt` value under `key`; a value it cannot read becomes null.

    A wrong key, a tampered or truncated ciphertext, or a non-base64 value yields NULL
    rather than aborting the query — one unreadable row must not kill a scan of a
    billion. Round-tripping under the correct key is total, so an all-null result is the
    unambiguous signal that the key is wrong. Null → null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> key, other = "00" * 32, "11" * 32
            >>> ds = bt.from_pydict({"ssn": ["123-45-6789"]})
            >>> enc = ds.select(c=bt.aes_encrypt(bt.col("ssn"), key))
            >>> enc.select(s=bt.aes_decrypt(bt.col("c"), other)).to_pydict()
            {'s': [None]}

    Args:
        e: The column or expression holding base64 ciphertext.
        key: The same 32-byte key used to encrypt.

    Returns:
        A text expression: the plaintext, or null where decryption failed.

    Raises:
        PlanError: If `key` does not decode to exactly 32 bytes.
    """
    return StrFunc("aes_decrypt", _as_text(e), pattern=_validated_key("aes_decrypt", key))
