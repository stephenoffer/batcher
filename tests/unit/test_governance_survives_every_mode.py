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

import contextlib
import io
import pathlib

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


# --------------------------------------------------------------------------------------
# The export and write surface.
#
# The three modes above are the ones that return an Arrow table. They are not the whole
# surface: a `Dataset` has around thirty terminal operations, and a governed read leaks just
# as completely through `to_pandas()`, `iter_rows()` or `write.csv()` as through `collect()`.
# Each of those is a separate code path to the same plan, and the ones that hand data to
# another framework are exactly where a path might reach for batches directly.
#
# So this is a sweep rather than a list of the ones someone thought of. Each entry renders
# the whole result to a string and the assertion is that neither a raw `ssn` nor a `us` row
# appears anywhere in it -- coarse on purpose, because it needs no per-mode knowledge of the
# return type and therefore covers a mode nobody looked at closely.
#
# `show()` is the reason the renderer is `_rendered` and not `str(...)`. It prints and
# returns None, so a sweep that stringified its return value would check the word "None" for
# a social security number and pass for every possible implementation of `show`.
# --------------------------------------------------------------------------------------

_RAW_SSNS = ("000000000", "000000002", "000000004")


def _rendered(fn, ds) -> str:
    """Everything `fn` produces for `ds`, whether returned, printed, or still unevaluated.

    A lazy result is materialized rather than repr'd. `value_counts` returns a `Dataset`,
    whose repr is ``Dataset(columns=['ssn', 'count'])`` -- no rows, so scanning it for a
    social security number found nothing and the governed assertion passed without executing
    anything. The ungoverned control is what caught that, which is the whole reason it is
    here: the governed half of this sweep looked identical before and after the fix.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = fn(ds)
    if isinstance(result, Dataset):
        result = result.to_pylist()
    return buffer.getvalue() + repr(result)


_EXPORTS = {
    "to_pydict": lambda ds: ds.to_pydict(),
    "to_pylist": lambda ds: ds.to_pylist(),
    "to_arrow": lambda ds: ds.to_arrow(),
    "to_pandas": lambda ds: ds.to_pandas(),
    "to_polars": lambda ds: ds.to_polars(),
    "to_numpy": lambda ds: ds.to_numpy(),
    "iter_batches": lambda ds: list(ds.iter_batches()),
    "iter_rows": lambda ds: list(ds.iter_rows()),
    "iter_slices": lambda ds: list(ds.iter_slices()),
    "first": lambda ds: ds.first(),
    "show": lambda ds: ds.show(),
    "value_counts": lambda ds: ds.value_counts("ssn"),
}

_WRITES = {
    "write.parquet": lambda ds, path: ds.write.parquet(path),
    "write.csv": lambda ds, path: ds.write.csv(path),
    "write.json": lambda ds, path: ds.write.json(path),
}


def _bytes_written(path: pathlib.Path) -> str:
    """Every byte under `path`, as text, however the writer chose to lay it out."""
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    return "".join(f.read_bytes().decode("utf-8", "ignore") for f in files if f.is_file())


@pytest.mark.parametrize("mode", sorted(_EXPORTS))
def test_no_export_path_returns_governed_data(mode, governed_source, analyst, catalog):
    """Every way of getting rows out of a governed dataset, not the three that return Arrow."""
    text = _rendered(_EXPORTS[mode], _govern("scan", governed_source, analyst, catalog))
    leaked = [s for s in _RAW_SSNS if s in text]
    assert not leaked, f"{mode} returned unmasked ssn values {leaked}"
    assert "us" not in text, f"{mode} returned rows outside the principal's region"


@pytest.mark.parametrize("mode", sorted(_EXPORTS))
def test_every_export_path_would_have_seen_the_leak(mode, governed_source):
    """The control for the sweep above, and it is not optional.

    Half of those assertions are `not in`, which is the shape that decays into a tautology
    without anyone touching it: a mode that returned an empty string, raised into a swallowed
    exception, or rendered a summary rather than the rows would satisfy every one of them
    while checking nothing. Running the identical renderer against an *ungoverned* plan and
    requiring the raw value to appear is what proves the sweep can see what it is looking for.
    """
    text = _rendered(_EXPORTS[mode], _shape("scan", governed_source))
    assert any(s in text for s in _RAW_SSNS), (
        f"{mode} does not surface an ssn even ungoverned, so the assertion that a governed "
        "one is absent proves nothing about this path"
    )


@pytest.mark.parametrize("mode", sorted(_WRITES))
def test_no_write_path_persists_governed_data(mode, governed_source, analyst, catalog, tmp_path):
    """A governed read written to disk must be governed on disk.

    This is the shape that already destroyed data once here: `compact` read a governed table
    and wrote the result back, so `email` became `XXXXXXX` permanently. The inverse -- a write
    that persists what the principal may not see -- leaves no trace at all, because the file
    looks exactly like a correct one to everybody who reads it afterwards.
    """
    destination = tmp_path / mode.replace(".", "_")
    _WRITES[mode](_govern("scan", governed_source, analyst, catalog), str(destination))
    written = _bytes_written(destination)
    assert written, "nothing was written, so the assertions below are vacuous"
    leaked = [s for s in _RAW_SSNS if s in written]
    assert not leaked, f"{mode} persisted unmasked ssn values {leaked}"


@pytest.mark.parametrize("mode", sorted(_WRITES))
def test_every_write_path_would_have_persisted_the_leak(mode, governed_source, tmp_path):
    """The control for the writes. Parquet is the reason this is measured rather than assumed:
    it is a binary container, so whether a plaintext ssn is findable in its bytes at all is a
    property of the encoding, not something to take on faith."""
    destination = tmp_path / mode.replace(".", "_")
    _WRITES[mode](_shape("scan", governed_source), str(destination))
    written = _bytes_written(destination)
    assert any(s in written for s in _RAW_SSNS), (
        f"{mode} does not persist a findable ssn even ungoverned, so the governed assertion "
        "proves nothing about this path"
    )
