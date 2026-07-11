"""Catalog resolution and the shape of the enforced plan (no engine required).

These pin the *decisions* — which columns are visible, which mask applies, which row
filters survive — and the plan shape `enforce` produces. The end-to-end proof that a
principal cannot read what it is denied lives in
`tests/integration/test_governance_enforcement.py`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import AccessDeniedError
from batcher.governance import Principal, SecurityCatalog, enforce
from batcher.plan.expr_ir import Col
from batcher.plan.logical import Filter, Project, Scan
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit

TABLE = "/data/customers.parquet"
COLUMNS = ["id", "email", "region", "salary"]
_SCHEMA = SchemaRef.from_arrow(
    pa.schema([(c, pa.string() if c != "id" else pa.int64()) for c in COLUMNS])
)


def _scan() -> Scan:
    return Scan(0, _SCHEMA)


def _govern(catalog: SecurityCatalog, principal: Principal):
    """The governed plan alone (the audit events are asserted separately)."""
    return enforce(_scan(), [TABLE], principal, catalog)[0]


ANALYST = Principal("ana", roles=["analyst"], attrs={"region": "EU"})
ADMIN = Principal("root", roles=["admin"])


# --- Principal ----------------------------------------------------------------
def test_a_principal_freezes_its_roles_and_attributes():
    """A mutable role set would make authorization depend on when it was read."""
    roles = ["analyst"]
    p = Principal("ana", roles=roles, attrs={"region": "EU"})
    roles.append("admin")
    assert not p.has_role("admin")
    with pytest.raises(TypeError):
        p.attrs["region"] = "US"  # type: ignore[index]


def test_an_empty_exemption_list_exempts_nobody():
    """The failure mode that would silently disable every policy."""
    assert not ANALYST.has_any_role([])
    assert not Principal("anon").has_any_role([])


# --- Access -------------------------------------------------------------------
def test_a_table_with_no_policy_is_untouched():
    """Installing a catalog must not perturb queries about other tables."""
    catalog = SecurityCatalog().grant("analyst", on="/data/other.parquet")
    scan = _scan()
    governed, events = enforce(scan, [TABLE], ANALYST, catalog)
    assert governed is scan
    assert events == ()


def test_a_table_with_no_grant_is_open_for_access():
    """Masks and row filters alone must not lock a table down."""
    catalog = SecurityCatalog().mask_column(TABLE, "email", lambda c: bt.mask(c))
    assert catalog.visible_columns(TABLE, COLUMNS, ANALYST) == COLUMNS


def test_the_first_grant_makes_a_table_deny_by_default():
    catalog = SecurityCatalog().grant("analyst", on=TABLE, select=["id", "email"])
    assert catalog.visible_columns(TABLE, COLUMNS, ANALYST) == ["id", "email"]
    assert catalog.visible_columns(TABLE, COLUMNS, ADMIN) == []


def test_a_grant_without_columns_grants_every_column():
    catalog = SecurityCatalog().grant("analyst", on=TABLE, select=["id"]).grant("admin", on=TABLE)
    assert catalog.visible_columns(TABLE, COLUMNS, ADMIN) == COLUMNS


def test_grants_from_several_roles_union():
    catalog = (
        SecurityCatalog().grant("a", on=TABLE, select=["id"]).grant("b", on=TABLE, select=["email"])
    )
    both = Principal("p", roles=["a", "b"])
    assert catalog.visible_columns(TABLE, COLUMNS, both) == ["id", "email"]


def test_visible_columns_preserve_schema_order():
    catalog = SecurityCatalog().grant("analyst", on=TABLE, select=["salary", "id"])
    assert catalog.visible_columns(TABLE, COLUMNS, ANALYST) == ["id", "salary"]


def test_a_principal_with_no_visible_column_is_denied_the_table():
    catalog = SecurityCatalog().grant("admin", on=TABLE)
    with pytest.raises(AccessDeniedError) as exc:
        _govern(catalog, ANALYST)
    assert exc.value.table == TABLE


# --- Masking ------------------------------------------------------------------
def test_a_tag_mask_governs_every_column_carrying_the_tag():
    catalog = (
        SecurityCatalog()
        .tag(TABLE, "email", "pii")
        .tag(TABLE, "salary", "pii")
        .mask_tag("pii", lambda c: bt.mask(c))
    )
    assert catalog.mask_for(TABLE, "email", ANALYST) is not None
    assert catalog.mask_for(TABLE, "salary", ANALYST) is not None
    assert catalog.mask_for(TABLE, "id", ANALYST) is None


def test_an_explicit_column_mask_overrides_the_tag_mask():
    def explicit(c):
        return c.str.upper()

    catalog = (
        SecurityCatalog()
        .tag(TABLE, "email", "pii")
        .mask_tag("pii", lambda c: bt.mask(c))
        .mask_column(TABLE, "email", explicit)
    )
    assert catalog.mask_for(TABLE, "email", ANALYST) is explicit


def test_an_exempt_role_reads_the_raw_value():
    catalog = (
        SecurityCatalog()
        .tag(TABLE, "email", "pii")
        .mask_tag("pii", lambda c: bt.mask(c), exempt=["admin"])
    )
    assert catalog.mask_for(TABLE, "email", ADMIN) is None
    assert catalog.mask_for(TABLE, "email", ANALYST) is not None


def test_a_tag_with_no_mask_attached_masks_nothing():
    catalog = SecurityCatalog().tag(TABLE, "email", "pii")
    assert catalog.mask_for(TABLE, "email", ANALYST) is None


# --- Row filters --------------------------------------------------------------
def test_row_filters_apply_unless_the_principal_is_exempt():
    catalog = SecurityCatalog().filter_rows(
        TABLE, lambda p: Col("region") == p.attrs["region"], exempt=["admin"]
    )
    assert len(catalog.row_filters_for(TABLE, ANALYST)) == 1
    assert catalog.row_filters_for(TABLE, ADMIN) == []


def test_the_row_filter_sits_below_the_projection_so_it_may_read_denied_columns():
    """A row-access policy runs with the catalog's authority, not the caller's."""
    catalog = (
        SecurityCatalog()
        .grant("analyst", on=TABLE, select=["id"])
        .filter_rows(TABLE, lambda p: Col("region") == p.attrs["region"])
    )
    governed = _govern(catalog, ANALYST)
    assert isinstance(governed, Project)
    assert [i.alias for i in governed.items] == ["id"]
    assert isinstance(governed.input, Filter)
    assert isinstance(governed.input.input, Scan)


def test_several_row_filters_are_conjoined():
    catalog = (
        SecurityCatalog()
        .filter_rows(TABLE, lambda _p: Col("region") == "EU")
        .filter_rows(TABLE, lambda _p: Col("salary") < "500")
    )
    governed = _govern(catalog, ANALYST)
    # No grants and no masks, so no projection is inserted: the Filter is the root.
    assert isinstance(governed, Filter)
    assert "and" in str(governed.predicate.to_ir())


# --- Plan shape ---------------------------------------------------------------
def test_an_all_visible_unmasked_table_gets_no_projection():
    """Governance must not insert an identity `Project` the optimizer has to see through."""
    catalog = SecurityCatalog().grant("analyst", on=TABLE)
    assert isinstance(_govern(catalog, ANALYST), Scan)


def test_masking_is_applied_at_the_scan_not_at_the_output():
    """The raw value must never exist above the leaf, or a filter could recover it."""
    catalog = SecurityCatalog().mask_column(TABLE, "email", lambda c: bt.mask(c))
    governed = _govern(catalog, ANALYST)
    assert isinstance(governed, Project)
    assert isinstance(governed.input, Scan)
    email = next(i for i in governed.items if i.alias == "email")
    assert email.expr.to_ir()["fn"] == "mask"


def test_the_masked_projection_preserves_the_column_order():
    catalog = SecurityCatalog().mask_column(TABLE, "region", lambda c: bt.mask(c))
    governed = _govern(catalog, ANALYST)
    assert [i.alias for i in governed.items] == COLUMNS


# --- Audit ---------------------------------------------------------------------
def test_enforce_reports_what_it_enforced():
    """The event comes out of the traversal that built the plan, so it cannot drift."""
    catalog = (
        SecurityCatalog()
        .grant("analyst", on=TABLE, select=["id", "email"])
        .mask_column(TABLE, "email", lambda c: bt.mask(c))
        .filter_rows(TABLE, lambda _p: Col("region") == "EU", name="own_region")
    )
    _, events = enforce(_scan(), [TABLE], ANALYST, catalog)
    (event,) = events
    assert event.principal == "ana"
    assert event.roles == ("analyst",)
    assert event.table == TABLE
    assert event.visible == ("id", "email")
    assert event.denied == ("region", "salary")
    assert event.masked == ("email",)
    assert event.row_filters == ("own_region",)
    assert event.allowed


def test_an_ungoverned_table_produces_no_event():
    catalog = SecurityCatalog().grant("analyst", on="/data/other.parquet")
    assert enforce(_scan(), [TABLE], ANALYST, catalog)[1] == ()


def test_an_event_names_columns_and_policies_but_never_values():
    catalog = SecurityCatalog().mask_column(TABLE, "email", lambda c: bt.mask(c))
    (event,) = enforce(_scan(), [TABLE], ANALYST, catalog)[1]
    rendered = str(event)
    assert "ALLOW" in rendered and "email" in rendered
    assert "EU" not in rendered  # no attribute values, no predicates, no data


def test_a_denied_event_renders_as_a_denial():
    from batcher.governance import GovernanceEvent

    event = GovernanceEvent("ivy", ("intern",), TABLE, (), ("id",), (), ())
    assert not event.allowed
    assert str(event).startswith("governance: DENY ivy")


# --- Table naming --------------------------------------------------------------
class _FakeSource:
    def __init__(self, identity: str) -> None:
        self._identity = identity

    def identity(self) -> str:
        return self._identity


@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        ("parquet:/data/customers.parquet", "/data/customers.parquet"),
        ("delta:s3://bucket/tbl", "s3://bucket/tbl"),  # only the first colon splits
        ("warehouse.customers", "warehouse.customers"),  # a custom, prefix-less name
        ("mem:schema:4", ""),  # in-memory data has no durable name
        ("stream:schema", ""),
        ("", ""),
    ],
)
def test_table_name_resolution(identity, expected):
    """An unrecognized naming scheme must be governable, not silently exempt."""
    from batcher.api.security import table_name

    assert table_name(_FakeSource(identity)) == expected


def test_a_source_without_an_identity_is_ungoverned():
    from batcher.api.security import table_name

    assert table_name(object()) == ""
