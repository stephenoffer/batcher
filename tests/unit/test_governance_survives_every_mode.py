"""A governance rewrite must hold on every execution path, not just `collect()`.

Governance is row filters and column masks expressed as a *plan rewrite*
(`governance.enforce`), which is what should make it mode-independent: the policy becomes
`Filter` and `Project` nodes in the logical plan, and every mode runs that plan.

"Should" is doing real work in that sentence. The out-of-core and streaming paths **peel
row-wise operators off the top of a plan and re-apply them** to a breaker's result -- and a
governance rewrite inserts exactly such operators. A path that peeled the policy's `Filter`
and forgot to re-apply it would return rows the principal may not see, on precisely the entry
points a `collect()`-only test never reaches. That is a data leak, not a slow query, and it
would pass every correctness suite in the repo: the rows are *more* than expected, and nothing
compares a governed result against a policy.

Nothing covered this. Before this file, no test mentioning `RowFilter` or `ColumnMask` also
mentioned `spill=True`, `iter_batches`, or `distributed=True`.

Each shape is governed and then run three ways, over a **breaker** as well as a bare scan,
because peel-and-re-apply is exactly what a breaker triggers. The distributed path is not
here for the usual reason -- CI installs no Ray -- and it composes the same plan.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.api.dataset.frame import Dataset
from batcher.governance import Principal, Redact, SecurityCatalog, enforce

pytestmark = pytest.mark.unit

_TABLE = "people"


@pytest.fixture(scope="module")
def governed_source(tmp_path_factory):
    """Four Parquet parts, half `eu` and half `us`, so a dropped filter doubles the rows."""
    table = pa.table(
        {
            "id": list(range(200)),
            "ssn": [f"{i:09d}" for i in range(200)],
            "region": ["us" if i % 2 else "eu" for i in range(200)],
            "v": list(range(200)),
        }
    )
    directory = tmp_path_factory.mktemp("people")
    for part in range(4):
        pq.write_table(table, directory / f"p{part}.parquet")
    return str(directory)


@pytest.fixture
def catalog():
    """`ssn` redacted to its last four digits; rows scoped to the principal's region."""
    cat = SecurityCatalog().grant("analyst", on=_TABLE, select=["id", "ssn", "region", "v"])
    cat.mask_column(_TABLE, "ssn", Redact(show_last=4))
    cat.filter_rows(_TABLE, lambda p: bt.col("region") == p.attrs["region"], name="region_scope")
    return cat


@pytest.fixture
def analyst():
    return Principal("ana", roles=["analyst"], attrs={"region": "eu"})


def _shape(name: str, path: str):
    ds = bt.read.parquet(path)
    if name == "scan":
        return ds
    if name == "aggregate":
        return ds.group_by("region").agg(n=bt.col("v").count())
    if name == "sort":
        return ds.sort("v")
    if name == "distinct":
        return ds.select("region").distinct()
    raise AssertionError(name)


_SHAPES = ("aggregate", "distinct", "scan", "sort")

_MODES = {
    "collect": lambda ds: ds.collect(),
    "spill": lambda ds: ds.collect(spill=True, num_partitions=4),
    "stream": lambda ds: pa.Table.from_batches(list(ds.iter_batches())),
}


def _govern(shape: str, path: str, analyst, catalog) -> Dataset:
    ds = _shape(shape, path)
    plan, _events = enforce(ds._plan, [_TABLE], analyst, catalog)
    return Dataset(plan, ds._sources)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("mode", sorted(_MODES))
def test_the_row_filter_holds(shape, mode, governed_source, analyst, catalog):
    """No mode may return a region the principal is not scoped to."""
    out = _MODES[mode](_govern(shape, governed_source, analyst, catalog))
    regions = set(out.column("region").to_pylist())
    assert regions == {"eu"}, (
        f"{mode} returned regions {sorted(regions)} for {shape}; the policy's row filter did "
        "not survive this execution path -- rows the principal may not see were returned"
    )


@pytest.mark.parametrize("mode", sorted(_MODES))
def test_the_column_mask_holds(mode, governed_source, analyst, catalog):
    """No mode may return an unredacted `ssn`."""
    out = _MODES[mode](_govern("scan", governed_source, analyst, catalog))
    values = out.column("ssn").to_pylist()
    assert values, "no rows came back, so the mask assertion would be vacuous"
    assert all(v.startswith("XXXXX") for v in values), (
        f"{mode} returned an unmasked ssn: {next(v for v in values if not v.startswith('XXXXX'))}"
    )


@pytest.mark.parametrize("mode", sorted(_MODES))
def test_an_ungoverned_plan_really_does_leak(mode, governed_source):
    """The control, and this file is worthless without it.

    Every assertion above says a governed result contains only `eu`. If the fixture happened
    to hold no `us` rows -- or if the shape dropped `region` -- those would pass against a
    policy that was never applied at all. So run the *same* shapes with no governance and
    require the opposite: both regions present, and `ssn` in the clear.
    """
    out = _MODES[mode](_shape("scan", governed_source))
    assert set(out.column("region").to_pylist()) == {"eu", "us"}
    assert not any(v.startswith("XXXXX") for v in out.column("ssn").to_pylist())
