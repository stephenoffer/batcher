"""A governed column's *values* must not be persisted into the shared statistics store.

Statistics come in two kinds and the distinction is the whole point. A row count, a null
count, a distinct estimate — those describe the *shape* of the data and leak nothing. A
`min`/`max` pair is literally two values out of the column. And a **bloom filter is a
membership oracle**: holding one for an `ssn` column lets anyone test whether a specific
SSN is present, without ever being granted read on the table.

The store they land in is not private by nature. `MetadataHub`'s backends include Redis
and object storage, which is the point — a learned-stats store shared across a fleet. So
anything written there is readable by every principal with hub access, including the ones
the catalog exists to keep out of that column.

`test_a_bloom_over_a_governed_column_is_a_membership_oracle` below is the demonstration
that this matters at all: it builds the bloom Batcher would have persisted and queries it
for a secret value.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.source_stats import _value_bearing_columns_to_redact

pytestmark = pytest.mark.unit

TABLE = "/data/people.parquet"


@pytest.fixture
def catalog() -> bt.SecurityCatalog:
    """An analyst may read `id` and `region`; `ssn` is invisible and `email` is masked."""
    return (
        bt.SecurityCatalog()
        .grant("analyst", on=TABLE, select=["id", "region", "email"])
        .mask_column(TABLE, "email", lambda c: bt.mask(c, show_last=4))
    )


@pytest.fixture
def analyst() -> bt.Principal:
    return bt.Principal("ana", roles=["analyst"])


COLUMNS = ["id", "region", "email", "ssn"]


class TestRedactionDecision:
    def test_nothing_is_redacted_outside_a_security_block(self) -> None:
        """An ungoverned deployment must behave exactly as it did before.

        This is the property that keeps the change from being a silent optimizer
        regression for every user who has never written a policy.
        """
        assert _value_bearing_columns_to_redact(TABLE, COLUMNS) == set()

    def test_an_invisible_column_is_redacted(
        self, catalog: bt.SecurityCatalog, analyst: bt.Principal
    ) -> None:
        with bt.security(catalog, analyst):
            redacted = _value_bearing_columns_to_redact(TABLE, COLUMNS)
        assert "ssn" in redacted, "a column the principal cannot SELECT still leaked its values"

    def test_a_masked_column_is_redacted(
        self, catalog: bt.SecurityCatalog, analyst: bt.Principal
    ) -> None:
        """Masking is the subtler case, and the easier one to get wrong.

        `email` *is* selectable, so a visibility check alone passes it. But the whole
        point of the mask is that the principal never sees the raw value — and a bloom or
        a min/max built before the mask is applied carries exactly that raw value.
        """
        with bt.security(catalog, analyst):
            redacted = _value_bearing_columns_to_redact(TABLE, COLUMNS)
        assert "email" in redacted, "a masked column leaked its unmasked values"

    def test_a_freely_readable_column_keeps_its_statistics(
        self, catalog: bt.SecurityCatalog, analyst: bt.Principal
    ) -> None:
        # Over-redacting is not "safe": it blinds the optimizer on columns the principal
        # is explicitly allowed to read, which costs plans for no security gain.
        with bt.security(catalog, analyst):
            redacted = _value_bearing_columns_to_redact(TABLE, COLUMNS)
        assert "id" not in redacted
        assert "region" not in redacted

    def test_an_ungoverned_table_is_untouched(
        self, catalog: bt.SecurityCatalog, analyst: bt.Principal
    ) -> None:
        """Installing a catalog must not perturb tables no policy mentions."""
        with bt.security(catalog, analyst):
            assert _value_bearing_columns_to_redact("/data/unrelated.parquet", COLUMNS) == set()


def test_a_persisted_bloom_is_an_exact_membership_oracle() -> None:
    """Why redaction is needed at all, rather than just suppressing `min`/`max`.

    This is not an argument about what a bloom filter could in principle reveal. It builds
    the artifact Batcher actually persists (`_build_bloom_index`, i.e. `build_column_bloom`)
    and queries it through the code the optimizer actually uses (`BloomIndex.from_bytes`),
    then asks it about values the querier was never granted. Every real value answers
    PRESENT and every fabricated one answers absent.

    A bloom has false positives but no false negatives, so an `absent` is *proof* of
    absence — which is itself information about a column nobody was allowed to read.
    Persisted into a Redis or object-storage `MetadataHub` shared across a fleet, this is
    a read of a governed column that never touches the table.
    """
    from batcher.api.source_stats import _build_bloom_index
    from batcher.plan.bloom_index import BloomIndex

    present = ["123-45-6789", "999-99-9999"]
    absent = ["000-00-0000", "aaa-bb-cccc", "555-55-5555"]
    table = pa.table({"ssn": pa.array([*present, "111-11-1111"])})

    blooms = _build_bloom_index(table, ["ssn"])
    if not blooms.get("ssn"):
        pytest.skip("bloom index construction is disabled in this build")
    index = BloomIndex.from_bytes(blooms["ssn"])

    for value in present:
        assert index.contains(value), f"{value} is in the column but the bloom denies it"
    for value in absent:
        assert not index.contains(value), (
            f"{value} answered PRESENT — a false positive here weakens the demonstration "
            f"but not the point: a *negative* answer is still proof of absence"
        )


def test_the_oracle_is_not_persisted_for_a_governed_column(
    catalog: bt.SecurityCatalog, analyst: bt.Principal
) -> None:
    """The fix, stated against the leak above: `ssn` keeps no bloom.

    `ndv` survives, deliberately — a distinct *count* is a cardinality, it answers no
    question about any particular value, and the optimizer needs it to order joins.
    """
    with bt.security(catalog, analyst):
        redacted = _value_bearing_columns_to_redact(TABLE, COLUMNS)
    assert "ssn" in redacted
    assert "email" in redacted
    assert {"id", "region"}.isdisjoint(redacted)
