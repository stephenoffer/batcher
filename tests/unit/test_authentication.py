"""`bt.Principal("root", roles=["admin"])` must not be a way to become an admin.

That line was the whole of the "no authentication" gap: a `Principal` was a name somebody
typed, and every policy honored it. A deployment could write perfect row filters and column
masks and a caller could step around all of them by claiming a role.

`governance.require_verified_principal` closes it for the code paths a deployment controls:
an identity has to come from a `CredentialVerifier` that checked something, not from a
constructor call.

**What this does not do**, stated here because a test file is where an over-claim gets
found: it does not make Batcher a trust boundary. Code running inside the engine's process
can set `issuer` by hand, and no in-process mechanism can prevent that. The boundary is the
process. This makes "we only accept established identities" *expressible and enforced*,
which is different from making it unbypassable.
"""

from __future__ import annotations

import dataclasses
import time

import pytest

import batcher as bt
from batcher._internal.errors import AccessDeniedError
from batcher.config import active_config, set_config
from batcher.governance.authn import (
    AuthenticationError,
    CredentialVerifier,
    HmacTokenVerifier,
    JwtVerifier,
    ProcessIdentityVerifier,
)

pytestmark = pytest.mark.unit

KEY = "s3cret-signing-key"


@pytest.fixture
def verifier() -> HmacTokenVerifier:
    return HmacTokenVerifier(key=KEY, issuer="gateway")


@pytest.fixture
def require_verified():
    """Turn on `require_verified_principal` for one test."""
    original = active_config()
    current = active_config()
    set_config(
        current.replace(
            governance=dataclasses.replace(current.governance, require_verified_principal=True)
        )
    )
    try:
        yield
    finally:
        set_config(original)


@pytest.fixture(autouse=True)
def _no_leftover_verifier():
    """No test may leave a verifier installed — it is a process-wide global."""
    yield
    bt.set_verifier(None)


class TestTheGapItCloses:
    def test_an_asserted_admin_is_refused(self, require_verified) -> None:
        """The exploit, asserted to fail."""
        catalog = bt.SecurityCatalog().grant("admin", on="/data/t.parquet")
        with (
            pytest.raises(AccessDeniedError, match="asserted, not verified"),
            bt.security(catalog, bt.Principal("root", roles=["admin"])),
        ):
            pass

    def test_a_verified_principal_is_accepted(
        self, require_verified, verifier: HmacTokenVerifier
    ) -> None:
        catalog = bt.SecurityCatalog().grant("analyst", on="/data/t.parquet")
        bt.set_verifier(verifier)
        principal = bt.authenticate(verifier.mint("ana", roles=["analyst"]))
        with bt.security(catalog, principal):
            pass  # no raise

    def test_it_is_off_by_default(self, verifier: HmacTokenVerifier) -> None:
        # A single-user notebook must keep working exactly as before.
        assert active_config().governance.require_verified_principal is False
        catalog = bt.SecurityCatalog().grant("admin", on="/data/t.parquet")
        with bt.security(catalog, bt.Principal("root", roles=["admin"])):
            pass


class TestExpiry:
    def test_expired_claims_are_refused_even_without_the_knob(self) -> None:
        """Expiry is honoured unconditionally, and that is deliberate.

        An `exp` only exists because a verifier read it off a credential. Ignoring it would
        let a long-running process keep acting on an identity whose token lapsed hours ago,
        which is a worse failure than the one `require_verified_principal` guards against.
        """
        assert active_config().governance.require_verified_principal is False
        catalog = bt.SecurityCatalog().grant("analyst", on="/data/t.parquet")
        stale = bt.Principal("ana", roles=["analyst"], issuer="gateway", expires_at=0.0)
        with pytest.raises(AccessDeniedError, match="expired"), bt.security(catalog, stale):
            pass

    def test_a_principal_without_an_expiry_never_expires(self) -> None:
        assert not bt.Principal("ana", issuer="os").expired()

    def test_the_verifier_rejects_an_expired_token(self, verifier: HmacTokenVerifier) -> None:
        token = verifier.mint("ana", roles=["analyst"], ttl_seconds=-3600)
        with pytest.raises(AuthenticationError, match="expired"):
            verifier.verify(token)


class TestHmacTokenVerifier:
    def test_a_minted_token_round_trips_its_claims(self, verifier: HmacTokenVerifier) -> None:
        token = verifier.mint("ana", roles=["analyst", "auditor"], attrs={"region": "EU"})
        principal = verifier.verify(token)
        assert principal.name == "ana"
        assert sorted(principal.roles) == ["analyst", "auditor"]
        assert principal.attrs["region"] == "EU"
        assert principal.issuer == "gateway"
        assert principal.verified

    def test_a_forged_token_is_rejected(self, verifier: HmacTokenVerifier) -> None:
        """Swap the payload for an admin one, keep the signature. Must not verify."""
        import base64
        import json

        good = verifier.mint("ana", roles=["analyst"])
        forged_claims = json.dumps({"sub": "root", "roles": ["admin"]}).encode()
        payload = base64.urlsafe_b64encode(forged_claims).decode().rstrip("=")
        with pytest.raises(AuthenticationError, match="signature"):
            verifier.verify(f"{payload}.{good.rpartition('.')[2]}")

    def test_a_token_signed_with_another_key_is_rejected(self) -> None:
        attacker = HmacTokenVerifier(key="not-the-key", issuer="gateway")
        with pytest.raises(AuthenticationError, match="signature"):
            HmacTokenVerifier(key=KEY).verify(attacker.mint("root", roles=["admin"]))

    @pytest.mark.parametrize("junk", ["", "no-dot", "!!!.!!!", "a.b.c"])
    def test_malformed_credentials_fail_closed(
        self, verifier: HmacTokenVerifier, junk: str
    ) -> None:
        # Every unparseable input must raise, never return a principal built from
        # whatever could be salvaged.
        with pytest.raises(AuthenticationError):
            verifier.verify(junk)

    def test_a_string_roles_claim_is_rejected(self, verifier: HmacTokenVerifier) -> None:
        """`"roles": "analyst"` would iterate into one role per character.

        `Principal` already rejects this, but the message would talk about a constructor
        argument when the real problem is the token's encoding.
        """
        import base64
        import hashlib
        import hmac
        import json

        claims = json.dumps({"sub": "ana", "roles": "analyst", "exp": time.time() + 60})
        payload = base64.urlsafe_b64encode(claims.encode()).decode().rstrip("=")
        sig = hmac.new(KEY.encode(), payload.encode(), hashlib.sha256).digest()
        token = f"{payload}.{base64.urlsafe_b64encode(sig).decode().rstrip('=')}"
        with pytest.raises(AuthenticationError, match="not a list"):
            verifier.verify(token)

    def test_the_key_may_be_a_secret_reference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # So the signing key never has to sit in a config file or a plan.
        monkeypatch.setenv("BATCHER_TEST_TOKEN_KEY", KEY)
        by_reference = HmacTokenVerifier(key="env:BATCHER_TEST_TOKEN_KEY", issuer="gateway")
        principal = by_reference.verify(HmacTokenVerifier(key=KEY).mint("ana"))
        assert principal.name == "ana"


class TestProcessIdentityVerifier:
    def test_it_reports_the_os_user_as_verified(self) -> None:
        principal = ProcessIdentityVerifier(roles={"analyst"}).verify("")
        assert principal.issuer == "os"
        assert principal.verified
        assert principal.has_role("analyst")

    def test_it_ignores_whatever_is_presented(self) -> None:
        # There is nothing to present: the answer is who the kernel says you are.
        one = ProcessIdentityVerifier().verify("")
        two = ProcessIdentityVerifier().verify("a-token-someone-made-up")
        assert one.name == two.name


class TestTheSeam:
    def test_authenticate_without_a_verifier_fails_closed(self) -> None:
        bt.set_verifier(None)
        with pytest.raises(AuthenticationError, match="no credential verifier"):
            bt.authenticate("anything")

    def test_the_installed_verifier_is_what_authenticate_uses(
        self, verifier: HmacTokenVerifier
    ) -> None:
        bt.set_verifier(verifier)
        assert bt.current_verifier() is verifier
        assert bt.authenticate(verifier.mint("ana")).name == "ana"

    def test_all_three_shipped_verifiers_satisfy_the_protocol(self) -> None:
        """The Protocol earns its place only if more than one thing implements it.

        Three do, which is what keeps `CredentialVerifier` from being the "empty
        framework" the anti-speculation rule forbids.
        """
        for candidate in (
            HmacTokenVerifier(key=KEY),
            ProcessIdentityVerifier(),
            JwtVerifier(jwks_url="https://idp/.well-known/jwks.json"),
        ):
            assert isinstance(candidate, CredentialVerifier)


class TestJwtVerifierDefaults:
    def test_only_asymmetric_algorithms_are_permitted_by_default(self) -> None:
        """Allowing HS256 beside RS256 is the algorithm-confusion attack.

        An attacker signs a token with the issuer's *public* key used as an HMAC secret,
        and a verifier that accepts both families verifies it happily. Pinning the default
        to asymmetric-only is the fix, so it is pinned as a test.
        """
        assert JwtVerifier(jwks_url="https://idp/jwks").algorithms == ("RS256", "ES256")
        assert not any(
            alg.startswith("HS") for alg in JwtVerifier(jwks_url="https://idp/jwks").algorithms
        )
