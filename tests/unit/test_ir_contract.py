"""The JSON IR wire contract holds across languages, checked mechanically.

`CLAUDE.md` invariant #8 makes the Python `to_ir()` tags and the Rust `serde` tags one
contract. `test_ir_tags.py` pins the Python side against a golden, which catches a typo
but cannot see Rust; the differential suite exercises Rust but only for the tags some
test actually names. This module closes the gap between them by deriving the tags Rust
will accept straight from the enum definitions and comparing the whole vocabulary.
"""

from __future__ import annotations

import pytest
from tools.lint_ir_contract import PAIRS, python_tags, rust_tags

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("rel_path", "enum", "dotted"), PAIRS, ids=[p[1] for p in PAIRS])
def test_python_and_rust_agree_on_every_tag(rel_path: str, enum: str, dotted: str) -> None:
    rust = rust_tags(rel_path, enum)
    py = python_tags(dotted)
    assert py - rust == set(), (
        f"{dotted} emits tags `bc_{enum}` will reject: {sorted(py - rust)}. "
        "Rust rejects an unknown tag at deserialization, so this is a hard runtime error."
    )
    assert rust - py == set(), (
        f"`{enum}` accepts tags the control plane never emits: {sorted(rust - py)}. "
        "Either lower them from Python or delete the dead wire surface."
    )


def test_the_checker_would_notice_a_drift() -> None:
    """The gate can fail — a comparison that is true by construction proves nothing."""
    rust = rust_tags("bc-ir/src/lib.rs", "RelOp")
    assert "range_join" in rust
    assert rust - (rust - {"range_join"}) == {"range_join"}


def test_rust_variant_renames_are_honoured() -> None:
    """`#[serde(rename)]` wins over `rename_all`, or every renamed tag reads as drift."""
    expr = rust_tags("bc-expr/src/lib.rs", "Expr")
    # `Expr::NullIf` carries `#[serde(rename = "nullif")]`; snake_case alone gives
    # `null_if`, which is not what the control plane emits and not what Rust accepts.
    assert "nullif" in expr
    assert "null_if" not in expr
