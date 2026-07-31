"""Leakage monitors — credentials, encoded payloads, and exfiltration channels in text.

The failures here are the ones that are catastrophic once and invisible until then: a live API
key in a generation, a base64 blob smuggling an instruction past a reviewer, a link built to
carry the conversation off to someone else's server. Each is rare per row, which is exactly why
it wants a corpus rate rather than a spot check.

Like the injection monitors, these are surface patterns rather than classifiers. A key monitor
recognizes the well-known credential prefixes and will not recognize a bespoke token format. Add
your own with `contains_any_rate` where you have one.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column
from batcher.plan.functions.metrics.safety._rate import matches_any, rate

__all__ = [
    "credential_leak_rate",
    "data_uri_rate",
    "encoded_payload_rate",
    "private_key_rate",
    "url_exfiltration_rate",
]

# The public, well-known credential prefixes. Each is specific enough that a match is almost
# never a false positive, which is what makes this monitor worth alerting on rather than
# reviewing.
_CREDENTIALS = (
    r"\bsk-[A-Za-z0-9_-]{16,}",  # OpenAI-style secret key
    r"\bsk-ant-[A-Za-z0-9_-]{16,}",  # Anthropic
    r"\bAKIA[0-9A-Z]{16}\b",  # AWS access key id
    r"\bASIA[0-9A-Z]{16}\b",  # AWS temporary key id
    r"\bgh[pousr]_[A-Za-z0-9]{20,}",  # GitHub tokens
    r"\bxox[baprs]-[A-Za-z0-9-]{10,}",  # Slack
    r"\bAIza[0-9A-Za-z_-]{35}\b",  # Google API key
    r"\bglpat-[A-Za-z0-9_-]{20,}",  # GitLab personal access token
    r"\bey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",  # JWT
)

_PRIVATE_KEY = (
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"-----BEGIN OPENSSH PRIVATE KEY-----",
    r"PuTTY-User-Key-File-\d",
)

# A run this long of base64 alphabet with no spaces is not prose. It is the shape an encoded
# instruction or an embedded blob takes.
_ENCODED_PAYLOAD = r"[A-Za-z0-9+/]{80,}={0,2}"

# A URL carrying a long opaque value in its query string, which is what an exfiltration link
# looks like: the payload is the conversation, encoded into a parameter.
_URL_EXFILTRATION = (
    r"https?://[^\s]*[?&][A-Za-z0-9_%-]+=[A-Za-z0-9+/%_-]{60,}",
    r"!\[[^\]]*\]\(https?://[^\s)]*[?&][^\s)]{40,}\)",
)

_DATA_URI = r"data:[a-z]+/[a-z0-9.+-]+;base64,"


def credential_leak_rate(text: IntoExpr) -> Expr:
    """The fraction of texts containing something shaped like a live API credential.

    Recognizes the public token formats — OpenAI and Anthropic secret keys, AWS access key ids,
    GitHub, GitLab and Slack tokens, Google API keys, and JWTs. Each prefix is specific enough
    that a match is rarely accidental, which makes this one of the few monitors here worth
    paging on rather than reviewing in batch.

    Run it on generations to catch a model reciting a key from its training data or its context,
    and on ingested documents to catch a key you are about to embed and store.

    Args:
        text: The text column to inspect (name or expression).

    Returns:
        The credential-pattern rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> out = bt.from_pydict({"o": ["use AKIAIOSFODNN7EXAMPLE", "no secrets here"]})
            >>> out.agg(r=bt.credential_leak_rate("o")).to_pydict()["r"][0]
            0.5
    """
    return rate(matches_any(text, _CREDENTIALS))


def private_key_rate(text: IntoExpr) -> Expr:
    """The fraction of texts containing a PEM or OpenSSH private-key header.

    Narrower and more conclusive than `credential_leak_rate`: a private-key armor line has no
    innocent reason to appear in a generation or a retrieved document. Kept separate so it can
    be alerted on at a different threshold than the token formats, which do occasionally show up
    in documentation examples.

    Args:
        text: The text column to inspect (name or expression).

    Returns:
        The private-key rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> docs = bt.from_pydict({"d": ["-----BEGIN RSA PRIVATE KEY-----", "hello"]})
            >>> docs.agg(r=bt.private_key_rate("d")).to_pydict()["r"][0]
            0.5
    """
    return rate(matches_any(text, _PRIVATE_KEY))


def encoded_payload_rate(text: IntoExpr) -> Expr:
    """The fraction of texts containing a long unbroken base64-alphabet run.

    Encoding is how an instruction gets past a reviewer and past a pattern monitor: the
    injection is not in the text, it is in what the model decodes from it. A run of eighty
    base64 characters with no whitespace is not prose, so this finds the shape without trying to
    guess the content.

    It matches legitimate embedded data too — an inline image, a signature, a serialized blob —
    so on a corpus that carries those it measures volume rather than risk. Read it per source
    with `group_by` and watch the sources that never used to have any.

    Args:
        text: The text column to inspect (name or expression).

    Returns:
        The encoded-payload rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> docs = bt.from_pydict({"d": ["SWdub3JlIGFsbCBw" * 6, "ordinary prose"]})
            >>> docs.agg(r=bt.encoded_payload_rate("d")).to_pydict()["r"][0]
            0.5
    """
    return rate(_as_column(text).str.regexp_matches(_ENCODED_PAYLOAD))


def data_uri_rate(text: IntoExpr) -> Expr:
    """The fraction of texts containing a base64 ``data:`` URI.

    A `data:` URI carries its whole payload inline, which is what makes it useful for embedding
    a small image and what makes it a delivery channel: `data:text/html;base64,...` rendered in
    a browser is a page you did not write, running in your origin. Check generations for it
    before rendering them, and ingested documents for it before storing them.

    Args:
        text: The text column to inspect (name or expression).

    Returns:
        The data-URI rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> out = bt.from_pydict({"o": ["data:text/html;base64,PHNjcmlwdD4=", "plain"]})
            >>> out.agg(r=bt.data_uri_rate("o")).to_pydict()["r"][0]
            0.5
    """
    return rate(_as_column(text).str.regexp_matches(_DATA_URI))


def url_exfiltration_rate(text: IntoExpr) -> Expr:
    """The fraction of texts containing a link that carries a long opaque payload.

    The exfiltration pattern that works against a chat UI: the model is talked into emitting a
    markdown image whose URL encodes the conversation, and the client fetches it on render,
    sending the data to whoever owns the host. No click is needed, which is why it is worth
    catching before the text reaches a renderer.

    A long encoded query parameter is also what a legitimate signed URL looks like, so allowlist
    your own hosts before alerting, or read the rate per destination domain.

    Args:
        text: The text column to inspect (name or expression).

    Returns:
        The exfiltration-link rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> out = bt.from_pydict(
            ...     {"o": ["![x](https://e.co/p?d=" + "A" * 64 + ")", "see https://e.co/p"]}
            ... )
            >>> out.agg(r=bt.url_exfiltration_rate("o")).to_pydict()["r"][0]
            0.5
    """
    return rate(matches_any(text, _URL_EXFILTRATION))
