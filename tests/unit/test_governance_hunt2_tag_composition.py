"""Composition of multiple column-mask *tags* on one column must be most-restrictive.

A column can carry several sensitivity tags (``pii``, ``confidential``, ...). Each tag
may attach its own `mask_tag`, each with its own exemptions. The security contract is
most-restrictive-wins: being exempt from *one* tag's mask must not grant raw access
while *another* tag still masks the column.

The regression these pin: `SecurityCatalog.mask_for` iterated tags in sorted order and
returned on the first tag that carried a mask — returning ``None`` (raw) if the principal
was exempt from *that* tag, even when a later tag would have masked the column. A column
tagged both ``a`` (analyst-exempt) and ``z`` (not exempt) was therefore read RAW by an
analyst: an S1 policy bypass triggered purely by tag name ordering.
"""

from __future__ import annotations

import os
import tempfile

import pytest

import batcher as bt
from batcher.governance import Nullify, Redact, SecurityCatalog

pytestmark = pytest.mark.unit

TABLE = "/data/t.parquet"


def _multi_tag_catalog() -> SecurityCatalog:
    # `x` carries BOTH tags. The analyst is exempt from `a` (which sorts first) but NOT
    # from `z`. The strictest applicable tag (`z`) must govern.
    return (
        SecurityCatalog()
        .tag(TABLE, "x", "a", "z")
        .mask_tag("a", Nullify(), exempt=["analyst"])
        .mask_tag("z", Redact(show_last=0))
    )


def test_exempt_from_one_tag_does_not_bypass_another() -> None:
    cat = _multi_tag_catalog()
    analyst = bt.Principal("ana", roles=["analyst"])
    # Exempt from `a` but not `z`: the column MUST still be masked (by `z`).
    mask = cat.mask_for(TABLE, "x", analyst)
    assert mask is not None, "column read raw despite a non-exempt sensitivity tag"


def test_exempt_from_all_tags_reads_raw() -> None:
    # Exempt from every applicable tag mask → raw, as before.
    cat = (
        SecurityCatalog()
        .tag(TABLE, "x", "a", "z")
        .mask_tag("a", Nullify(), exempt=["analyst"])
        .mask_tag("z", Redact(), exempt=["analyst"])
    )
    analyst = bt.Principal("ana", roles=["analyst"])
    assert cat.mask_for(TABLE, "x", analyst) is None


def test_single_tag_unaffected() -> None:
    cat = SecurityCatalog().tag(TABLE, "x", "z").mask_tag("z", Redact())
    analyst = bt.Principal("ana", roles=["analyst"])
    assert cat.mask_for(TABLE, "x", analyst) is not None


def test_multi_tag_bypass_end_to_end() -> None:
    """The bypass is observable in a real read: the value must never appear raw."""
    d = tempfile.mkdtemp()
    path = os.path.join(d, "t.parquet")
    bt.from_pydict({"id": [1, 2], "secret": ["alpha", "bravo"]}).write(path, format="parquet")
    cat = (
        SecurityCatalog()
        .tag(path, "secret", "a", "z")  # both tags on the sensitive column
        .mask_tag("a", Nullify(), exempt=["analyst"])  # analyst exempt from `a`
        .mask_tag("z", Redact(show_last=0))  # `z` masks everyone
    )
    analyst = bt.Principal("ana", roles=["analyst"])
    with bt.security(cat, analyst):
        ds = bt.read.parquet(path)
    out = ds.sort("id").to_pydict()
    assert "alpha" not in out["secret"], out
    assert "bravo" not in out["secret"], out
    assert out["secret"] == ["XXXXX", "XXXXX"], out
