"""A valid signature says which key set signed a token, never who it was minted for.

`JwtVerifier` defaults `issuer` and `audience` to empty and then skips both checks. That is
not loose configuration, it is two accepted attacks, and the reason is a property of how the
large identity providers are deployed rather than anything about Batcher: they publish one
key set across many tenants and many applications.

* With `iss` unchecked, a token from **another tenant of the same provider** verifies. It is
  a real token, correctly signed by a key in the same JWKS, issued to somebody else.
* With `aud` unchecked, a token minted for **another application of the same tenant**
  verifies. Same shape, one level in.

Both stay legal, because a deployment mid-migration may genuinely not know its audience yet
and refusing would make the verifier unadoptable at the moment it is most needed. Both warn,
which is the pattern this codebase already uses for a usage that works and weakens security
(`batcher.hmac_sha256` with an inline key).

The warning is emitted at **construction**, so it names the line that configured the
verifier rather than a line in the middle of a pipeline, and so a busy deployment gets it
once per verifier instead of once per credential.
"""

from __future__ import annotations

import warnings

import pytest

from batcher._internal.errors import SecurityWarning
from batcher.governance.authn import JwtVerifier

pytestmark = pytest.mark.unit

JWKS = "https://idp.example/.well-known/jwks.json"


def _warnings(**kwargs) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        JwtVerifier(jwks_url=JWKS, **kwargs)
    return [w for w in caught if issubclass(w.category, SecurityWarning)]


class TestItWarnsAboutEachSkippedCheck:
    def test_neither_claim_set(self):
        (warning,) = _warnings()
        assert "issuer (`iss`)" in str(warning.message)
        assert "audience (`aud`)" in str(warning.message)

    def test_only_the_issuer_set(self):
        (warning,) = _warnings(issuer="https://idp.example/")
        assert "audience (`aud`)" in str(warning.message)
        assert "issuer (`iss`)" not in str(warning.message)

    def test_only_the_audience_set(self):
        (warning,) = _warnings(audience="batcher")
        assert "issuer (`iss`)" in str(warning.message)
        assert "audience (`aud`)" not in str(warning.message)

    def test_the_message_names_the_attack_not_just_the_setting(self):
        """ "`aud` is unset" tells an operator nothing they can act on. "a token minted for
        another application of the same tenant verifies" tells them what it costs."""
        (warning,) = _warnings()
        message = str(warning.message)
        assert "another tenant of the same identity provider" in message
        assert "another application of the same tenant" in message

    def test_it_names_the_verifier_it_is_about(self):
        """A deployment builds more than one; the warning has to say which."""
        (warning,) = _warnings()
        assert JWKS in str(warning.message)


class TestItStaysQuietWhenConfigured:
    def test_both_claims_set_warns_about_nothing(self):
        """The half without which "it warns" would be satisfied by warning always."""
        assert _warnings(issuer="https://idp.example/", audience="batcher") == []

    def test_from_issuer_still_needs_an_audience(self, monkeypatch):
        """`from_issuer` always sets `iss` and leaves `aud` to the caller, so it must warn
        about the one it did not set and not about the one it did."""
        from batcher.governance.authn import verifiers

        monkeypatch.setattr(verifiers, "_discover_jwks", lambda issuer, *, timeout: JWKS)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            verifiers.JwtVerifier.from_issuer("https://idp.example/")
        security = [w for w in caught if issubclass(w.category, SecurityWarning)]
        assert len(security) == 1
        assert "audience (`aud`)" in str(security[0].message)
        assert "issuer (`iss`)" not in str(security[0].message)

    def test_from_issuer_with_an_audience_is_silent(self, monkeypatch):
        from batcher.governance.authn import verifiers

        monkeypatch.setattr(verifiers, "_discover_jwks", lambda issuer, *, timeout: JWKS)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            verifiers.JwtVerifier.from_issuer("https://idp.example/", audience="batcher")
        assert [w for w in caught if issubclass(w.category, SecurityWarning)] == []


class TestTheVerifierIsOtherwiseUnchanged:
    """Warning must not become refusing: the unconfigured verifier still works."""

    def test_an_unconfigured_verifier_is_still_constructed_and_usable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SecurityWarning)
            verifier = JwtVerifier(jwks_url=JWKS)
        assert verifier.jwks_url == JWKS
        assert verifier.issuer == ""
        assert verifier.audience == ""

    def test_the_asymmetric_only_default_is_untouched(self):
        """The other load-bearing default on this class: allowing HS256 beside RS256 is the
        algorithm-confusion attack."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SecurityWarning)
            algorithms = JwtVerifier(jwks_url=JWKS).algorithms
        assert not any(alg.startswith("HS") for alg in algorithms)
