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
