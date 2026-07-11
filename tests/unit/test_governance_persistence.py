"""A catalog built from declarative policies survives persistence (pickle round-trip).

A platform that keeps its governance policy in an external store reconstructs a
`SecurityCatalog` per session. The declarative mask/filter factories
(`governance.masks`, `governance.filters`) exist so that a catalog built from them is
picklable — a lambda over `principal.attrs` is not — and enforces identically after a
round-trip. These tests pin exactly that: same visible columns, same masks, same rows.
"""

from __future__ import annotations

import copy
import pickle

import pyarrow as pa
import pytest

import batcher as bt
from batcher.governance import (
    AttributeIn,
    Encrypt,
    MatchesAttribute,
    Nullify,
    Pseudonymize,
    Redact,
    SecurityCatalog,
    enforce,
)
from batcher.governance.principal import Principal
from batcher.plan.logical import Scan
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit

TABLE = "/data/customers.parquet"
COLUMNS = ["id", "email", "region", "salary"]
_SCHEMA = SchemaRef.from_arrow(
    pa.schema([(c, pa.string() if c != "id" else pa.int64()) for c in COLUMNS])
)
ANALYST = Principal("ana", roles=["analyst"], attrs={"region": "EU"})


def _catalog() -> SecurityCatalog:
    return (
        SecurityCatalog()
        .grant("analyst", on=TABLE, select=["id", "email", "region"])
        .grant("admin", on=TABLE)
        .tag(TABLE, "email", "pii")
        .mask_tag("pii", Pseudonymize("env:PII_KEY"), exempt=["admin"])
        .mask_column(TABLE, "salary", Nullify())
        .filter_rows(TABLE, MatchesAttribute("region", "region"), exempt=["admin"])
    )


def _enforced_ir(catalog: SecurityCatalog, principal: Principal) -> dict:
    plan, _ = enforce(Scan(0, _SCHEMA), [TABLE], principal, catalog)
    return plan.to_ir()


@pytest.mark.parametrize(
    "mask", [Redact(show_last=4), Pseudonymize("env:K"), Encrypt("env:K"), Nullify()]
)
def test_declarative_masks_are_picklable_and_call_identically(mask):
    restored = pickle.loads(pickle.dumps(mask))
    assert restored == mask
    # Both produce the same lowered expression over a column.
    assert mask(bt.col("x")).to_ir() == restored(bt.col("x")).to_ir()


@pytest.mark.parametrize(
    "filt", [MatchesAttribute("region", "region"), AttributeIn("region", "regions")]
)
def test_declarative_filters_are_picklable_and_call_identically(filt):
    restored = pickle.loads(pickle.dumps(filt))
    assert restored == filt
    who = Principal("p", attrs={"region": "EU", "regions": "EU,US"})
    assert filt(who).to_ir() == restored(who).to_ir()


def test_a_catalog_of_declarative_policies_pickles():
    catalog = _catalog()
    restored = pickle.loads(pickle.dumps(catalog))
    assert isinstance(restored, SecurityCatalog)


def test_enforcement_is_identical_after_a_pickle_round_trip():
    """The whole point: a persisted-then-loaded catalog governs byte-for-byte the same."""
    catalog = _catalog()
    restored = pickle.loads(pickle.dumps(catalog))
    assert _enforced_ir(restored, ANALYST) == _enforced_ir(catalog, ANALYST)


def test_a_deep_copied_catalog_enforces_identically():
    catalog = _catalog()
    assert _enforced_ir(copy.deepcopy(catalog), ANALYST) == _enforced_ir(catalog, ANALYST)


def test_redact_lowers_to_mask_and_preserves_length():
    ds = bt.from_pydict({"card": ["4111111111111234"]})
    masked = ds.select(m=Redact(show_last=4)(bt.col("card")))
    assert masked.to_pydict() == {"m": ["XXXXXXXXXXXX1234"]}


def test_nullify_produces_a_typed_null():
    ds = bt.from_pydict({"salary": [100, 200]})
    out = ds.select(s=Nullify()(bt.col("salary"))).to_pydict()
    assert out == {"s": [None, None]}


def test_matches_attribute_restricts_to_the_principals_region():
    import os
    import tempfile

    path = str(tempfile.mkdtemp() + "/c.parquet")
    bt.from_pydict({"id": [1, 2, 3], "region": ["EU", "US", "EU"]}).write(path, format="parquet")
    catalog = SecurityCatalog().filter_rows(path, MatchesAttribute("region", "region"))
    with bt.security(catalog, Principal("ana", attrs={"region": "EU"})):
        got = bt.read.parquet(path).sort("id").to_pydict()
    assert got["id"] == [1, 3]
    os.remove(path)


def test_a_filter_referencing_a_missing_attribute_fails_clearly():
    from batcher._internal.errors import PlanError

    catalog = SecurityCatalog().filter_rows(TABLE, MatchesAttribute("region", "region"))
    with pytest.raises(PlanError, match="does not have"):
        enforce(Scan(0, _SCHEMA), [TABLE], Principal("nobody"), catalog)
