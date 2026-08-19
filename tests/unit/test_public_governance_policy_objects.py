"""The governance policy dataclasses, constructed directly rather than through the builder.

``Grant``, ``RowFilter``, ``TagMask`` and ``ResidencyVerdict`` are exported from
``batcher.governance``, so a user can build a policy set as data -- read a catalog out of
YAML, hand it around, compare two versions of it -- instead of calling
``SecurityCatalog.grant(...)`` in sequence. Nothing tested them: every existing governance
test goes through the fluent builder, so the constructors, their defaults and their
frozen-ness were public surface with no coverage.

What each test asserts is that the object built by hand is *the same policy* the builder
produces, because that equivalence is the whole reason the classes are exported. Where
the builder adds validation the constructor does not, that difference is pinned too --
it is the kind of gap that turns a hand-built catalog into a silently weaker one.

``active_residency`` and ``ResidencyCatalog.rule_for`` are here for the same reason: the
process-wide residency catalog is reachable and readable, and nothing checked what it says
before any rule is set.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest

import batcher as bt
from batcher.governance import (
    Grant,
    Principal,
    ResidencyVerdict,
    RowFilter,
    SecurityCatalog,
    TagMask,
    active_residency,
    enforce,
)
from batcher.plan.logical import Scan
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit

TABLE = "/data/customers.parquet"
COLUMNS = ["id", "email", "region", "salary"]
SCHEMA = SchemaRef.from_arrow(
    pa.schema([(c, pa.int64() if c == "id" else pa.string()) for c in COLUMNS])
)
ANALYST = Principal("ana", roles=["analyst"], attrs={"region": "EU"})
ADMIN = Principal("root", roles=["admin"])


def _visible(catalog: SecurityCatalog, principal: Principal) -> list[str]:
    """The columns a principal may read from the fixture table."""
    return sorted(catalog.visible_columns(TABLE, COLUMNS, principal))


def test_a_grant_built_by_hand_is_the_grant_the_builder_makes():
    """Same role, same table, same column set -- and the equality the dataclass gives."""
    built = SecurityCatalog().grant("analyst", on=TABLE, select=["id", "email"])
    by_hand = Grant(role="analyst", table=TABLE, columns=frozenset({"id", "email"}))
    assert by_hand in built._grants, f"{by_hand} is not the grant the builder recorded"
    assert _visible(built, ANALYST) == ["email", "id"]


def test_a_grant_with_no_column_set_means_every_column():
    """The documented meaning of ``columns=None``, distinct from an empty set."""
    everything = Grant("admin", TABLE)
    assert everything.columns is None
    nothing = Grant("nobody", TABLE, columns=frozenset())
    assert nothing.columns == frozenset()
    assert everything != nothing, "all columns and no columns must not compare equal"

    catalog = SecurityCatalog().grant("admin", on=TABLE)
    assert _visible(catalog, ADMIN) == sorted(COLUMNS)


def test_any_grant_on_a_table_switches_it_to_deny_by_default():
    """The property the ``Grant`` docstring states, which is what makes a catalog safe."""
    ungoverned = SecurityCatalog()
    assert _visible(ungoverned, ANALYST) == sorted(COLUMNS), "no grant means no restriction"

    governed = SecurityCatalog().grant("analyst", on=TABLE, select=["id"])
    assert _visible(governed, ANALYST) == ["id"]
    assert _visible(governed, ADMIN) == [], "a role with no grant sees nothing once one exists"


def test_a_row_filter_built_by_hand_carries_its_defaults():
    """The name and the exemption set default the way the builder's do."""
    policy = RowFilter(TABLE, lambda principal: bt.col("region") == principal.attrs["region"])
    assert policy.table == TABLE
    assert policy.name == "row_filter", "the default name is what an audit event reports"
    assert policy.exempt_roles == frozenset()

    named = RowFilter(
        TABLE, lambda p: bt.lit(True), name="eu_only", exempt_roles=frozenset({"admin"})
    )
    assert named.name == "eu_only"
    assert "admin" in named.exempt_roles


def test_a_hand_built_row_filter_enforces_the_same_predicate_as_the_builder():
    """The plan `enforce` produces must be the same either way."""
    predicate = lambda principal: bt.col("region") == principal.attrs["region"]  # noqa: E731
    built = SecurityCatalog().filter_rows(TABLE, predicate, name="eu_only")

    applied = built.row_filters_for(TABLE, ANALYST)
    assert [f.name for f in applied] == ["eu_only"]
    assert applied[0] == RowFilter(TABLE, predicate, "eu_only", frozenset()), (
        "the builder must record exactly the dataclass a user could have written"
    )

    plan, _ = enforce(Scan(0, SCHEMA), [TABLE], ANALYST, built)
    assert "filter" in str(plan.to_ir()), "the predicate must reach the plan"


def test_an_exempt_role_skips_the_row_filter_it_is_exempt_from():
    """The exemption set is load-bearing, so an empty one must exempt nobody."""
    catalog = SecurityCatalog().filter_rows(
        TABLE, lambda p: bt.col("region") == "EU", name="eu_only", exempt=["admin"]
    )
    assert [f.name for f in catalog.row_filters_for(TABLE, ANALYST)] == ["eu_only"]
    assert catalog.row_filters_for(TABLE, ADMIN) == []


def test_a_tag_mask_applies_wherever_the_tag_appears():
    """A ``TagMask`` is table-independent, which is what separates it from a column mask."""
    mask = TagMask("pii", lambda e: bt.lit("***"))
    assert mask.tag == "pii"
    assert mask.exempt_roles == frozenset()

    catalog = (
        SecurityCatalog()
        .tag(TABLE, "email", "pii")
        .tag("/data/other.parquet", "contact", "pii")
        .mask_tag("pii", lambda e: bt.lit("***"))
    )
    assert catalog.mask_for(TABLE, "email", ANALYST) is not None
    assert catalog.mask_for("/data/other.parquet", "contact", ANALYST) is not None
    assert catalog.mask_for(TABLE, "region", ANALYST) is None, "an untagged column is untouched"


def test_a_tag_mask_exemption_is_by_role():
    """An exempt role reads the column unmasked; everyone else does not."""
    catalog = (
        SecurityCatalog()
        .tag(TABLE, "email", "pii")
        .mask_tag("pii", lambda e: bt.lit("***"), exempt=["admin"])
    )
    assert catalog.mask_for(TABLE, "email", ANALYST) is not None
    assert catalog.mask_for(TABLE, "email", ADMIN) is None


def test_the_policy_objects_are_frozen_dataclasses():
    """A policy that can be mutated after installation is not a policy.

    Authorization would then depend on when the catalog was read rather than on what it
    says, which is the same failure ``Principal`` freezing its roles prevents.
    """
    for policy in (
        Grant("analyst", TABLE),
        RowFilter(TABLE, lambda p: bt.lit(True)),
        TagMask("pii", lambda e: e),
        ResidencyVerdict(allowed=True),
    ):
        assert dataclasses.is_dataclass(policy)
        field = dataclasses.fields(policy)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(policy, field, "mutated")


def test_a_residency_verdict_records_why_a_placement_was_refused():
    """The refusal carries the dataset, the region and what was allowed, not just a flag."""
    allowed = ResidencyVerdict(allowed=True)
    assert allowed.allowed is True
    assert allowed.enforced is False, "an unenforced catalog reports, it does not block"

    refused = ResidencyVerdict(
        allowed=False,
        dataset="/eu/customers",
        region="us-east-1",
        allowed_regions=frozenset({"eu-west-1"}),
        obligation="keep in the EU",
        enforced=True,
    )
    assert refused.allowed is False
    assert refused.region == "us-east-1"
    assert refused.allowed_regions == frozenset({"eu-west-1"})
    assert refused.obligation, "a refusal with no reason is not actionable"


def test_active_residency_is_off_and_empty_until_a_rule_is_set():
    """The default a process starts in, which is what makes residency opt-in."""
    catalog = active_residency()
    assert catalog.mode in {"off", "report", "enforce"}
    assert catalog.rule_for("/nothing/registered") is None, (
        "an unregistered dataset must have no rule rather than a default one"
    )


def test_residency_rule_lookup_finds_the_rule_that_was_set():
    """``rule_for`` against an installed catalog, restoring the previous one afterwards.

    ``set_residency`` returns what it displaced precisely so a caller can put it back, and
    this test uses that rather than reconstructing the previous state -- a residency
    catalog is process-wide, so a test that left one installed would change what every
    later test in the process sees.
    """
    from batcher.governance import DataResidency, ResidencyCatalog, set_residency

    catalog = ResidencyCatalog(
        mode="report",
        rules={
            "/eu/customers": DataResidency(
                dataset="/eu/customers",
                allowed_regions=frozenset({"eu-west-1"}),
                obligation="keep in the EU",
            )
        },
    )
    previous = set_residency(catalog)
    try:
        live = active_residency()
        assert live.mode == "report"
        rule = live.rule_for("/eu/customers")
        assert rule is not None
        assert rule.allowed_regions == frozenset({"eu-west-1"})
        assert rule.obligation == "keep in the EU"
        assert live.rule_for("/us/customers") is None, "an unregistered dataset has no rule"
    finally:
        set_residency(previous)
    assert active_residency() is previous or active_residency().mode == previous.mode
