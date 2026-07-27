"""Governance can be made mandatory, so "the developer forgot the `with` block" is not a policy.

Row filters and column masks are enforced as a plan rewrite inside a `bt.security(...)`
block. That is the right default for a library — a `Dataset` built outside one behaves
exactly as it did before any catalog existed — and the wrong default for a deployment,
where forgetting the block is the difference between a masked column and a plain one, and
nothing anywhere says so.

`governance.mode` is that switch, and the middle setting is the load-bearing one.
`advisory` is not a half-measure: it is the only way to find every ungoverned read in a
real workload *before* flipping to `strict`. Without it, strict mode cannot be adopted
incrementally, and a control nobody can adopt protects nobody.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import pytest

import batcher as bt
from batcher._internal.errors import AccessDeniedError, SecurityWarning
from batcher.config import active_config, set_config

pytestmark = pytest.mark.unit


@pytest.fixture
def governance_mode():
    """Set `governance.mode` for one test and restore the process config afterwards."""
    original = active_config()

    def apply(mode: str) -> None:
        current = active_config()
        set_config(current.replace(governance=dataclasses.replace(current.governance, mode=mode)))

    try:
        yield apply
    finally:
        set_config(original)


@pytest.fixture
def table(tmp_path: Path) -> str:
    path = str(tmp_path / "t.parquet")
    bt.from_pydict({"id": [1, 2, 3], "email": ["a@x", "b@x", "c@x"]}).write(path, format="parquet")
    return path


class TestDefaultIsUnchanged:
    def test_off_is_the_default(self) -> None:
        # The whole design rests on this: installing the feature must not change how any
        # existing deployment behaves until someone opts in.
        assert bt.GovernanceConfig().mode == "off"
        assert active_config().governance.mode == "off"

    def test_an_ungoverned_read_still_works(self, table: str, governance_mode) -> None:
        governance_mode("off")
        assert len(bt.read.parquet(table).collect()) == 3


class TestStrictMode:
    def test_a_read_outside_a_security_block_is_refused(self, table: str, governance_mode) -> None:
        governance_mode("strict")
        with pytest.raises(AccessDeniedError, match="ungoverned read"):
            bt.read.parquet(table).collect()

    def test_the_error_says_how_to_fix_it(self, table: str, governance_mode) -> None:
        # A refusal that does not name the remedy just moves the confusion.
        governance_mode("strict")
        with pytest.raises(AccessDeniedError) as caught:
            bt.read.parquet(table).collect()
        assert "bt.security" in caught.value.hint

    def test_a_governed_read_proceeds(self, table: str, governance_mode) -> None:
        """Strict mode must refuse *ungoverned* reads, not all reads."""
        governance_mode("strict")
        catalog = bt.SecurityCatalog().grant("analyst", on=table, select=["id", "email"])
        with bt.security(catalog, bt.Principal("ana", roles=["analyst"])):
            rows = bt.read.parquet(table).collect()
        assert rows.num_rows == 3

    def test_an_ungovernable_source_is_refused_inside_a_governed_block(
        self, table: str, governance_mode
    ) -> None:
        """An in-memory table has no durable name, so no policy can be written about it.

        This is the case that matters, and it is *not* the same as reading outside a
        security block: here the caller has done everything right — installed a catalog,
        named a principal — and then joins in a dict. Passing that through silently is how
        an ungoverned read hides inside an otherwise governed pipeline, so strict mode
        refuses it rather than exempting it.
        """
        governance_mode("strict")
        catalog = bt.SecurityCatalog().grant("analyst", on=table, select=["id", "email"])
        with (
            bt.security(catalog, bt.Principal("ana", roles=["analyst"])),
            pytest.raises(AccessDeniedError, match="durable name"),
        ):
            bt.from_pydict({"a": [1, 2]}).collect()

    def test_reading_outside_a_block_reports_the_missing_block(self, governance_mode) -> None:
        """The other refusal, with its own message: no catalog was installed at all."""
        governance_mode("strict")
        with pytest.raises(AccessDeniedError, match="no security\\(\\) block"):
            bt.from_pydict({"a": [1, 2]}).collect()


class TestAdvisoryMode:
    def test_it_warns_and_proceeds(self, table: str, governance_mode) -> None:
        governance_mode("advisory")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rows = bt.read.parquet(table).collect()
        assert rows.num_rows == 3, "advisory mode must not block the read"
        messages = [str(w.message) for w in caught if issubclass(w.category, SecurityWarning)]
        assert messages, "advisory mode warned about nothing"
        assert "ungoverned read" in messages[0]

    def test_the_warning_names_the_source(self, table: str, governance_mode) -> None:
        # An operator sweeping a workload needs to know *which* read to go and govern.
        governance_mode("advisory")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bt.read.parquet(table).collect()
        assert any(table in str(w.message) for w in caught)

    def test_a_governed_read_warns_about_nothing(self, table: str, governance_mode) -> None:
        governance_mode("advisory")
        catalog = bt.SecurityCatalog().grant("analyst", on=table, select=["id", "email"])
        with (
            warnings.catch_warnings(record=True) as caught,
            bt.security(catalog, bt.Principal("ana", roles=["analyst"])),
        ):
            bt.read.parquet(table).collect()
        governance_warnings = [w for w in caught if issubclass(w.category, SecurityWarning)]
        assert not governance_warnings
