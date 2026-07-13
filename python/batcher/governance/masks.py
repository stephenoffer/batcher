"""Declarative, picklable column-mask factories.

`SecurityCatalog.mask_column` / `mask_tag` accept any ``Expr -> Expr`` callable, which is
maximally flexible in-process. But a lambda cannot be *persisted* — a platform that keeps
its governance policy in an external store (and reconstructs a catalog per session) needs
mask definitions that survive `pickle`/`copy`. These factories return small frozen
dataclasses that are callable **and** picklable, so a catalog built from them round-trips.

They cover the masks an enterprise policy actually uses — the same shapes as a Snowflake
or Databricks masking policy — and lower to the engine's data-protection functions
(`mask`, `hmac_sha256`, `aes_encrypt`), so the redaction runs in the Rust data plane.
Reach for a raw callable only for a one-off in-process mask that never needs persisting.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher.plan.expr_ir import Expr, nullif
from batcher.plan.functions.security import aes_encrypt, hmac_sha256, mask

__all__ = ["Encrypt", "Nullify", "Pseudonymize", "Redact"]


@dataclass(frozen=True, slots=True)
class Redact:
    """Replace characters with `char`, optionally revealing the first/last few.

    ``Redact(show_last=4)`` masks all but the last four characters — the "card ending
    1234" pattern. Lowers to `mask`; length-preserving, irreversible.

    Examples:
        .. doctest::

            >>> from batcher import col
            >>> from batcher.governance import Redact
            >>> Redact(show_last=4)(col("card_number"))
            col('card_number').cast('string').str.mask('X', 0, 4)
    """

    show_first: int = 0
    show_last: int = 0
    char: str = "X"

    def __call__(self, column: Expr) -> Expr:
        return mask(column, show_first=self.show_first, show_last=self.show_last, char=self.char)


@dataclass(frozen=True, slots=True)
class Pseudonymize:
    """Replace a value with its keyed HMAC-SHA-256 pseudonym.

    Deterministic (so pseudonymized columns still join and count distinct) and
    irreversible. `key` is a key reference (``env:NAME`` / ``file:PATH``) resolved by the
    data plane, so the persisted policy carries the reference, not the secret.

    Examples:
        .. doctest::

            >>> from batcher import col
            >>> from batcher.governance import Pseudonymize
            >>> Pseudonymize("env:HMAC_KEY")(col("email"))
            col('email').cast('string').str.hmac_sha256('env:HMAC_KEY')
    """

    key: str

    def __call__(self, column: Expr) -> Expr:
        return hmac_sha256(column, self.key)


@dataclass(frozen=True, slots=True)
class Encrypt:
    """Encrypt a value with AES-256-GCM-SIV (reversible with the key).

    `key` is a key reference (``env:NAME`` / ``file:PATH``). See `Pseudonymize` for why
    the reference, not the secret, is what the policy stores.

    Examples:
        .. doctest::

            >>> from batcher import col
            >>> from batcher.governance import Encrypt
            >>> Encrypt("env:AES_KEY")(col("email"))
            col('email').cast('string').str.aes_encrypt('env:AES_KEY')
    """

    key: str

    def __call__(self, column: Expr) -> Expr:
        return aes_encrypt(column, self.key)


@dataclass(frozen=True, slots=True)
class Nullify:
    """Replace the value with NULL — full redaction that keeps the column's type.

    The strongest mask: a principal sees the column exists but never any value.

    Examples:
        .. doctest::

            >>> from batcher import col
            >>> from batcher.governance import Nullify
            >>> Nullify()(col("salary"))
            NullIf(col('salary'), col('salary'))
    """

    def __call__(self, column: Expr) -> Expr:
        # `nullif(x, x)` is a typed NULL — the engine has no typed null literal, and a
        # bare `lit(None)` would not carry the column's type through the projection.
        return nullif(column, column)
