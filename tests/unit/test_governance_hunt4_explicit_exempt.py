"""An exemption from an explicit column mask must not bypass an un-exempt tag mask.

The security contract, established for tag-vs-tag composition in
`test_governance_hunt2_tag_composition.py`, is most-restrictive-wins: being exempt from
*one* mask policy must never grant raw access while *another* policy still masks the
column. That contract is broader than tags — it must also hold between an explicit
`mask_column` and a `mask_tag`.

The regression these pin: `SecurityCatalog.mask_for` returned early on an explicit
`ColumnMask`, yielding ``None`` (raw) whenever the principal was exempt from *that*
explicit mask — even when the same column carried a sensitivity tag whose `mask_tag` the
principal was NOT exempt from. A column with an analyst-exempt explicit mask that was also
tagged ``pii`` (masked for everyone) was therefore read RAW by an analyst: an S1 policy
bypass. Adding a narrow explicit exemption silently disabled the broad PII safety net.
"""

from __future__ import annotations

import os
import tempfile

import pytest

import batcher as bt
from batcher.governance import Nullify, Redact, SecurityCatalog

pytestmark = pytest.mark.unit

TABLE = "/data/t.parquet"


def test_exempt_from_explicit_mask_does_not_bypass_an_unexempt_tag() -> None:
    """Exempt from the explicit mask, but not from the column's ``pii`` tag mask."""
    cat = (
        SecurityCatalog()
        .mask_column(TABLE, "ssn", Redact(show_last=4), exempt=["analyst"])
        .tag(TABLE, "ssn", "pii")
        .mask_tag("pii", Nullify())  # masks everyone, no exemption
    )
    analyst = bt.Principal("ana", roles=["analyst"])
    mask = cat.mask_for(TABLE, "ssn", analyst)
    assert mask is not None, "column read raw despite a non-exempt sensitivity tag"


def test_explicit_mask_still_wins_when_not_exempt() -> None:
    """The explicit mask function still takes precedence over the tag when it applies."""

    def explicit(c: bt.Expr) -> bt.Expr:
        return c.str.upper()

    cat = (
        SecurityCatalog()
        .mask_column(TABLE, "ssn", explicit)  # no exemption
        .tag(TABLE, "ssn", "pii")
        .mask_tag("pii", Nullify())
    )
    analyst = bt.Principal("ana", roles=["analyst"])
    assert cat.mask_for(TABLE, "ssn", analyst) is explicit


def test_exempt_from_both_explicit_and_tag_reads_raw() -> None:
    """Exempt from every applicable mask → raw, as intended."""
    cat = (
        SecurityCatalog()
        .mask_column(TABLE, "ssn", Redact(show_last=4), exempt=["analyst"])
        .tag(TABLE, "ssn", "pii")
        .mask_tag("pii", Nullify(), exempt=["analyst"])
    )
    analyst = bt.Principal("ana", roles=["analyst"])
    assert cat.mask_for(TABLE, "ssn", analyst) is None


def test_explicit_exempt_bypass_end_to_end() -> None:
    """The bypass is observable in a real read: the raw value must never appear."""
    d = tempfile.mkdtemp()
    path = os.path.join(d, "t.parquet")
    bt.from_pydict({"id": [1, 2], "ssn": ["alpha", "bravo"]}).write(path, format="parquet")
    cat = (
        SecurityCatalog()
        # A narrow explicit mask that exempts analysts...
        .mask_column(path, "ssn", Redact(show_last=4), exempt=["analyst"])
        # ...must not disable the broad PII safety net masking everyone.
        .tag(path, "ssn", "pii")
        .mask_tag("pii", Redact(show_last=0))
    )
    analyst = bt.Principal("ana", roles=["analyst"])
    with bt.security(cat, analyst):
        ds = bt.read.parquet(path)
    out = ds.sort("id").to_pydict()
    assert "alpha" not in out["ssn"], out
    assert "bravo" not in out["ssn"], out
