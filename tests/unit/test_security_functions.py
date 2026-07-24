"""Plan-level contracts of the data-protection functions.

The engine-side behavior is covered by `tests/differential/test_diff_security_functions.py`
and the Rust unit tests. What is pinned here is what happens *before* the engine sees the
plan: keys are validated at plan-build time, and a key never escapes through a `repr`.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.fn_names import KEYED_STR_FNS, STR_FNS

pytestmark = pytest.mark.unit

HEX_KEY = "00" * 32
B64_KEY = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="  # the same 32 zero bytes


def test_the_keyed_functions_are_part_of_the_string_vocabulary():
    """`KEYED_STR_FNS` must be a subset of `STR_FNS`, or validation rejects its own nodes."""
    assert KEYED_STR_FNS <= STR_FNS
    assert {"mask", "hmac_sha256", "aes_encrypt", "aes_decrypt"} <= STR_FNS


@pytest.mark.parametrize("key", [HEX_KEY, B64_KEY])
def test_a_32_byte_key_is_accepted_as_hex_or_base64(key):
    assert bt.aes_encrypt(bt.col("x"), key).to_ir()["pattern"] == key


@pytest.mark.parametrize("key", ["", "abc", "00" * 16, "00" * 33, "not base64!!", "z" * 64])
def test_a_key_that_is_not_32_bytes_is_rejected_at_plan_build_time(key):
    """The error must arrive here, not from inside a scan after the query is admitted."""
    with pytest.raises(PlanError):
        bt.aes_encrypt(bt.col("x"), key)
    with pytest.raises(PlanError):
        bt.aes_decrypt(bt.col("x"), key)


def test_a_rejected_key_is_not_quoted_in_the_error():
    with pytest.raises(PlanError) as exc:
        bt.aes_encrypt(bt.col("x"), "hunter2-is-not-a-valid-key")
    assert "hunter2" not in str(exc.value)


def test_hmac_accepts_any_non_empty_key_because_it_is_not_an_aes_key():
    assert bt.hmac_sha256(bt.col("x"), "k").to_ir()["pattern"] == "k"
    with pytest.raises(PlanError):
        bt.hmac_sha256(bt.col("x"), "")


@pytest.mark.parametrize(
    "expr",
    [
        lambda: bt.aes_encrypt(bt.col("x"), HEX_KEY),
        lambda: bt.aes_decrypt(bt.col("x"), HEX_KEY),
        lambda: bt.hmac_sha256(bt.col("x"), "s3cret-key-material"),
    ],
)
def test_repr_redacts_the_key(expr):
    """A `repr` reaches tracebacks, debuggers, and notebook cells — it must not leak."""
    r = repr(expr())
    assert "***" in r
    assert HEX_KEY not in r
    assert "s3cret" not in r


def test_to_ir_still_carries_the_real_key():
    """Redaction is a display concern; the engine must receive the actual key."""
    assert bt.aes_encrypt(bt.col("x"), HEX_KEY).to_ir()["pattern"] == HEX_KEY


def test_mask_repr_is_not_redacted_because_it_holds_no_secret():
    assert "'*'" in repr(bt.mask(bt.col("x"), char="*"))


@pytest.mark.parametrize("char", ["", "ab", "  "])
def test_mask_requires_exactly_one_replacement_character(char):
    with pytest.raises(PlanError):
        bt.mask(bt.col("x"), char=char)


@pytest.mark.parametrize(("first", "last"), [(-1, 0), (0, -1)])
def test_mask_rejects_a_negative_reveal_count(first, last):
    with pytest.raises(PlanError):
        bt.mask(bt.col("x"), show_first=first, show_last=last)


def test_mask_lowers_reveal_counts_into_the_start_and_length_slots():
    """Pins the wire contract: `bc_expr::StrFunc::Mask` reads `start`/`length`."""
    ir = bt.mask(bt.col("x"), show_first=2, show_last=4, char="#").to_ir()
    assert (ir["e"], ir["fn"], ir["pattern"], ir["start"], ir["length"]) == (
        "str",
        "mask",
        "#",
        2,
        4,
    )


# --- Key references (env: / file:) --------------------------------------------
import warnings  # noqa: E402

from batcher._internal.errors import SecurityWarning  # noqa: E402


@pytest.mark.parametrize("ref", ["env:MY_KEY", "file:/run/secrets/aes.key"])
def test_a_key_reference_is_stored_verbatim_and_not_validated(ref):
    """A reference is resolved on the executing node, so it is not decoded here — a
    32-byte check would wrongly reject `env:MY_KEY`."""
    assert bt.aes_encrypt(bt.col("x"), ref).to_ir()["pattern"] == ref
    assert bt.aes_decrypt(bt.col("x"), ref).to_ir()["pattern"] == ref
    assert bt.hmac_sha256(bt.col("x"), ref).to_ir()["pattern"] == ref


def test_a_reference_does_not_warn():
    """A key given by reference is never embedded, so there is nothing to warn about."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", SecurityWarning)
        encrypted = bt.aes_encrypt(bt.col("x"), "env:MY_KEY")
        signed = bt.hmac_sha256(bt.col("x"), "file:/k")
    assert encrypted.to_ir()["pattern"] == "env:MY_KEY"
    assert signed.to_ir()["pattern"] == "file:/k"


def test_an_inline_literal_warns_that_it_is_embedded():
    for build in (
        lambda: bt.aes_encrypt(bt.col("x"), HEX_KEY),
        lambda: bt.aes_decrypt(bt.col("x"), HEX_KEY),
        lambda: bt.hmac_sha256(bt.col("x"), "secret"),
    ):
        with pytest.warns(SecurityWarning, match="embedded in the query plan"):
            build()


def test_repr_shows_a_reference_but_hides_a_literal():
    assert "env:MY_KEY" in repr(bt.aes_encrypt(bt.col("x"), "env:MY_KEY"))
    r = repr(bt.aes_encrypt(bt.col("x"), HEX_KEY))
    assert "***" in r and HEX_KEY not in r


def test_a_bad_inline_key_is_still_rejected_at_plan_build():
    with pytest.raises(PlanError):
        bt.aes_encrypt(bt.col("x"), "too-short")


def test_require_key_refs_makes_an_inline_key_a_hard_error(monkeypatch):
    """`SecurityWarning` is a weak control for a regulated deployment.

    It is a `UserWarning`, so Python shows it once per location and a caller that filtered
    warnings never sees it — while the key still travels verbatim in the IR JSON, into
    `explain(format="json")`, into the plan fingerprint, and out to every Ray worker the
    plan is shipped to. The env var lets an operator make that unrepresentable.
    """
    monkeypatch.setenv("BATCHER_REQUIRE_KEY_REFS", "1")
    for build in (
        lambda: bt.aes_encrypt(bt.col("x"), HEX_KEY),
        lambda: bt.aes_decrypt(bt.col("x"), HEX_KEY),
        lambda: bt.hmac_sha256(bt.col("x"), "secret"),
    ):
        with pytest.raises(PlanError, match="BATCHER_REQUIRE_KEY_REFS"):
            build()


def test_require_key_refs_still_allows_a_reference(monkeypatch):
    """Enforcement must forbid the *inline* form only — references are the fix it points at."""
    monkeypatch.setenv("BATCHER_REQUIRE_KEY_REFS", "1")
    assert bt.aes_encrypt(bt.col("x"), "env:MY_KEY") is not None
    assert bt.aes_encrypt(bt.col("x"), "file:/run/secrets/key") is not None


def test_inline_keys_are_allowed_by_default(monkeypatch):
    """Unset (and falsey) keeps the warn-only default, so notebooks and tests are unaffected."""
    monkeypatch.delenv("BATCHER_REQUIRE_KEY_REFS", raising=False)
    with pytest.warns(SecurityWarning):
        assert bt.aes_encrypt(bt.col("x"), HEX_KEY) is not None
    monkeypatch.setenv("BATCHER_REQUIRE_KEY_REFS", "0")
    with pytest.warns(SecurityWarning):
        assert bt.aes_encrypt(bt.col("x"), HEX_KEY) is not None


def test_cmd_reference_is_treated_as_a_reference_not_an_inline_key():
    """`_KEY_REF_SCHEMES` is a two-sided contract with the `bc-secrets` scheme table.

    A scheme the engine resolves but Python does not recognize is worse than unsupported:
    Python validates it as raw key material, so it is rejected at plan-build time and the
    engine never gets the chance to resolve it. This caught exactly that drift when the
    `cmd:` backend was added on the Rust side only.
    """
    from batcher.plan.functions.security import _is_key_ref

    for ref in ("env:K", "file:/run/secrets/k", "cmd:prod/aes-key"):
        assert _is_key_ref(ref), ref
        # A reference must pass through untouched — no 32-byte validation, no warning.
        assert bt.aes_encrypt(bt.col("x"), ref) is not None
    assert not _is_key_ref("0" * 64)


def test_every_engine_scheme_is_known_to_the_plan_layer():
    """Enumerate the schemes the Rust resolver implements, so adding one there without
    adding it here fails loudly instead of silently rejecting valid references."""
    from batcher.plan.functions.security import _KEY_REF_SCHEMES

    # Mirrors the backends registered in `crates/bc-secrets/src/lib.rs`.
    assert set(_KEY_REF_SCHEMES) == {"env:", "file:", "cmd:"}
