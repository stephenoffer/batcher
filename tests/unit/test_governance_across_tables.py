"""A policy follows its own table through a join, whichever side of it the table is on.

Governance binds when a table is *read* (`api/security/_binding`), one policy per scan, so a
plan holding several governed tables holds several independent rewrites. Nothing tested that.
Every existing governance test reads one table, and one table cannot tell a per-scan binding
apart from a positional one -- pair the first policy with the first scan and a single-table
plan behaves identically either way.

Two tables tell them apart, and only if the tables have the **same schema and different
policies**. That is what this file builds: `secrets_a` is masked and `secrets_b` is not,
both with an `id` and a `secret` column. A binding that drifted onto plan position would
mask whichever table was read first and hand back the other in the clear, and every column
name, row count and type would be exactly right. `test_the_unmasked_table_is_really_readable`
is what stops the file passing under a blanket mask that redacts everything it sees.

The join is run in both orders for the same reason. `A.join(B)` and `B.join(A)` differ only
in which scan the plan reaches first, which is precisely the variable a positional bug is
sensitive to and a per-scan one is not.

The self-join is the case worth keeping past the others. The same governed table appears
twice in one plan, so an implementation that governs "the scan of this table" rather than
"each scan of this table" masks one side and leaks the other -- with the leaked copy sitting
in a `secret_right` column beside the masked one, which looks like an ordinary join artifact.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.api.security import security
from batcher.governance import Principal, Redact, SecurityCatalog

pytestmark = pytest.mark.unit

#: Distinguishable on sight, so an assertion can say *which* table a value came from.
_A_VALUES = [f"A{i:05d}" for i in range(10)]
_B_VALUES = [f"B{i:05d}" for i in range(10)]


@pytest.fixture(scope="module")
def table_a(tmp_path_factory):
    """The governed table: `secret` masked, half the rows outside the principal's region."""
    directory = tmp_path_factory.mktemp("secrets_a")
    pq.write_table(
        pa.table(
            {
                "id": list(range(10)),
                "secret": _A_VALUES,
                "region": ["us" if i % 2 else "eu" for i in range(10)],
            }
        ),
        directory / "p.parquet",
    )
    return str(directory)


@pytest.fixture(scope="module")
def table_b(tmp_path_factory):
    """The same shape, deliberately *not* masked."""
    directory = tmp_path_factory.mktemp("secrets_b")
    pq.write_table(
        pa.table({"id": list(range(10)), "secret": _B_VALUES, "region": ["eu"] * 10}),
        directory / "p.parquet",
    )
    return str(directory)


@pytest.fixture
def catalog(table_a, table_b):
    """`table_a` masked and region-scoped; `table_b` granted and otherwise untouched."""
    cat = SecurityCatalog()
    cat.grant("analyst", on=table_a, select=["id", "secret", "region"])
    cat.grant("analyst", on=table_b, select=["id", "secret", "region"])
    cat.mask_column(table_a, "secret", Redact(show_last=2))
    return cat


@pytest.fixture
def analyst():
    return Principal("ana", roles=["analyst"])


def _read(catalog, analyst, build):
    """Build a plan with every read inside the block, then run it outside.

    Running the terminal operation outside is deliberate: `security()` documents that a
    dataset read inside a block stays governed "for the whole life of the resulting
    `Dataset`, including terminal operations performed after the block exits", and a test
    that collected inside the block would pass just as well if the policy were attached to
    the *execution* rather than to the read.
    """
    with security(catalog, analyst):
        dataset = build()
    return dataset.to_pylist()


def _text(rows) -> str:
    return repr(rows)


class TestEachTableKeepsItsOwnPolicy:
    """The property one table cannot demonstrate."""

    def test_the_masked_table_is_masked_alone(self, catalog, analyst, table_a):
        rows = _read(catalog, analyst, lambda: bt.read.parquet(table_a))
        assert not any(v in _text(rows) for v in _A_VALUES)

    def test_the_unmasked_table_is_really_readable(self, catalog, analyst, table_b):
        """The control. Every other assertion here says a value is absent, and a catalog
        that masked every column it was shown would satisfy all of them while proving the
        opposite of what this file claims."""
        rows = _read(catalog, analyst, lambda: bt.read.parquet(table_b))
        assert any(v in _text(rows) for v in _B_VALUES), (
            "the untouched table came back masked, so the 'masked' assertions elsewhere in "
            "this file do not show that a policy is matched to its own table"
        )

    @pytest.mark.parametrize("order", ["a_then_b", "b_then_a"])
    def test_a_join_masks_only_the_governed_side(self, catalog, analyst, table_a, table_b, order):
        """Both orders, because plan position is the variable a positional bug is sensitive
        to and a per-scan binding is not."""

        def build():
            left, right = (table_a, table_b) if order == "a_then_b" else (table_b, table_a)
            return bt.read.parquet(left).join(bt.read.parquet(right), on="id")

        text = _text(_read(catalog, analyst, build))
        assert not any(v in text for v in _A_VALUES), f"{order}: the masked table leaked"
        assert any(v in text for v in _B_VALUES), (
            f"{order}: the unmasked table was redacted too, so the policies were applied by "
            "position rather than by table"
        )

    def test_a_self_join_masks_both_copies(self, catalog, analyst, table_a):
        """One governed table, two scans. Governing "the scan of this table" rather than
        "each scan of this table" leaves the second copy in the clear, in a `secret_right`
        column that reads as an ordinary join artifact."""
        rows = _read(
            catalog,
            analyst,
            lambda: bt.read.parquet(table_a).join(bt.read.parquet(table_a), on="id"),
        )
        assert rows, "no rows, so the assertion below is vacuous"
        assert "secret_right" in rows[0], "the join did not produce a second copy to check"
        assert not any(v in _text(rows) for v in _A_VALUES)

    @pytest.mark.parametrize("order", ["a_then_b", "b_then_a"])
    def test_a_union_masks_only_the_governed_arm(self, catalog, analyst, table_a, table_b, order):
        def build():
            left, right = (table_a, table_b) if order == "a_then_b" else (table_b, table_a)
            columns = ["id", "secret"]
            return (
                bt.read.parquet(left)
                .select(*columns)
                .union(bt.read.parquet(right).select(*columns))
            )

        text = _text(_read(catalog, analyst, build))
        assert not any(v in text for v in _A_VALUES), f"{order}: the masked arm leaked"
        assert any(v in text for v in _B_VALUES), f"{order}: the unmasked arm was redacted"


class TestARowFilterAlsoFollowsItsTable:
    """A row filter on one table must survive being joined to an ungoverned one."""

    @pytest.fixture
    def scoped_catalog(self, catalog, table_a):
        catalog.filter_rows(table_a, lambda p: bt.col("region") == "eu", name="eu_only")
        return catalog

    def test_the_filter_holds_alone(self, scoped_catalog, analyst, table_a):
        rows = _read(scoped_catalog, analyst, lambda: bt.read.parquet(table_a))
        assert {r["region"] for r in rows} == {"eu"}
        assert len(rows) == 5, "the fixture must hold rows the filter removes, or this is vacuous"

    @pytest.mark.parametrize("order", ["a_then_b", "b_then_a"])
    def test_the_filter_holds_across_a_join(self, scoped_catalog, analyst, table_a, table_b, order):
        def build():
            left, right = (table_a, table_b) if order == "a_then_b" else (table_b, table_a)
            return bt.read.parquet(left).join(bt.read.parquet(right), on="id")

        rows = _read(scoped_catalog, analyst, build)
        regions = {r.get("region") for r in rows} | {r.get("region_right") for r in rows}
        assert "us" not in regions, f"{order}: a row outside the principal's region survived"
        assert len(rows) == 5, f"{order}: expected the filter to halve the join, got {len(rows)}"


class TestTheUngovernedControl:
    """Without a block there is no policy, which is what makes every assertion above mean
    something. `security()` documents that a table read outside any block is ungoverned."""

    def test_reading_outside_a_block_returns_the_raw_value(self, table_a):
        rows = bt.read.parquet(table_a).to_pylist()
        assert any(v in _text(rows) for v in _A_VALUES)

    def test_building_the_plan_outside_the_block_is_not_governed_by_it(
        self, catalog, analyst, table_a
    ):
        """Reading before the block and collecting inside it is *not* a governed read. This
        pins the documented boundary rather than leaving it to be rediscovered: governance
        attaches at the read, so a plan built earlier carries no policy however it is run."""
        dataset = bt.read.parquet(table_a)
        with security(catalog, analyst):
            rows = dataset.to_pylist()
        assert any(v in _text(rows) for v in _A_VALUES)
