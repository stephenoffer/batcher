"""End-to-end: a principal reads only what the catalog allows, on every code path.

The unit tests pin what the catalog *decides* and what `enforce` *builds*. These prove
the decision survives execution — including the paths that never run the plan (a
metadata-answered `count()`), the paths that rewrite it (the optimizer), and the
obvious attempts to defeat a mask by filtering, grouping, or joining on it.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import AccessDeniedError, PlanError

pytestmark = pytest.mark.integration

ANALYST = bt.Principal("ana", roles=["analyst"], attrs={"region": "EU"})
ADMIN = bt.Principal("root", roles=["admin"], attrs={"region": "EU"})
INTERN = bt.Principal("ivy", roles=["intern"])


@pytest.fixture
def customers(tmp_path):
    """A customers table: 2 EU rows, 2 US rows, with a PII column and a denied one."""
    path = str(tmp_path / "customers.parquet")
    bt.from_pydict(
        {
            "id": [1, 2, 3, 4],
            "email": ["a@x.com", "b@x.com", "c@y.com", "d@y.com"],
            "region": ["EU", "EU", "US", "US"],
            "salary": [100, 200, 300, 400],
        }
    ).write(path, format="parquet")
    return path


@pytest.fixture
def catalog(customers):
    """Analysts see id/email/region of their own region, with email masked."""
    return (
        bt.SecurityCatalog()
        .grant("analyst", on=customers, select=["id", "email", "region"])
        .grant("admin", on=customers)
        .tag(customers, "email", "pii")
        .mask_tag("pii", lambda c: bt.mask(c, show_last=6), exempt=["admin"])
        .filter_rows(customers, lambda p: bt.col("region") == p.attrs["region"], exempt=["admin"])
    )


def _read(catalog, principal, path):
    with bt.security(catalog, principal):
        return bt.read.parquet(path)


def test_an_analyst_sees_masked_columns_and_only_its_own_rows(catalog, customers):
    got = _read(catalog, ANALYST, customers).sort("id").to_pydict()
    assert got == {"id": [1, 2], "email": ["X@x.com", "X@x.com"], "region": ["EU", "EU"]}


def test_an_exempt_admin_sees_everything(catalog, customers):
    got = _read(catalog, ADMIN, customers).sort("id").to_pydict()
    assert got["salary"] == [100, 200, 300, 400]
    assert got["email"] == ["a@x.com", "b@x.com", "c@y.com", "d@y.com"]


def test_reading_outside_a_security_block_is_ungoverned(customers):
    """Policy is installed for a scope; there is no ambient catalog."""
    assert bt.read.parquet(customers).count() == 4


def test_a_principal_granted_no_column_cannot_open_the_table(catalog, customers):
    with pytest.raises(AccessDeniedError) as exc:
        _read(catalog, INTERN, customers)
    assert exc.value.table == customers


def test_a_denied_column_does_not_exist_for_the_principal(catalog, customers):
    """Fail-closed, and the error does not confirm that `salary` exists."""
    ds = _read(catalog, ANALYST, customers)
    assert "salary" not in ds.columns
    with pytest.raises(PlanError):
        ds.select("salary")


def test_a_denied_column_cannot_be_reached_through_a_filter(catalog, customers):
    ds = _read(catalog, ANALYST, customers)
    with pytest.raises(PlanError):
        ds.filter(bt.col("salary") > 150)


# --- The mask cannot be defeated ----------------------------------------------
def test_filtering_on_a_masked_column_sees_the_mask_not_the_value(catalog, customers):
    """The raw value never exists above the scan, so no predicate can observe it."""
    ds = _read(catalog, ANALYST, customers)
    assert ds.filter(bt.col("email") == "a@x.com").count() == 0
    assert ds.filter(bt.col("email") == "X@x.com").count() == 2


def test_grouping_by_a_masked_column_groups_by_the_mask(catalog, customers):
    ds = _read(catalog, ANALYST, customers)
    got = ds.group_by("email").agg(n=bt.count()).to_pydict()
    assert got == {"email": ["X@x.com"], "n": [2]}


def test_joining_on_a_masked_column_joins_on_the_mask(catalog, customers, tmp_path):
    """A join against known plaintext must not re-identify a masked row."""
    probe = str(tmp_path / "probe.parquet")
    bt.from_pydict({"email": ["a@x.com"], "hit": [1]}).write(probe, format="parquet")
    with bt.security(catalog, ANALYST):
        ds = bt.read.parquet(customers)
        p = bt.read.parquet(probe)
    assert ds.join(p, on="email", how="inner").count() == 0


# --- The row filter survives every path ---------------------------------------
def test_the_row_filter_is_honored_by_a_metadata_answered_count(catalog, customers):
    """`count()` can skip execution entirely — the filter must already be in the plan."""
    assert _read(catalog, ANALYST, customers).count() == 2


def test_the_row_filter_is_honored_by_is_empty(catalog, customers):
    other = bt.Principal("oli", roles=["analyst"], attrs={"region": "APAC"})
    assert _read(catalog, other, customers).is_empty()
    assert not _read(catalog, ANALYST, customers).is_empty()


def test_the_row_filter_is_honored_by_an_aggregate(catalog, customers):
    got = _read(catalog, ANALYST, customers).agg(n=bt.count(), lo=bt.col("id").min()).to_pydict()
    assert got == {"n": [2], "lo": [1]}


def test_the_row_filter_is_honored_when_writing(catalog, customers, tmp_path):
    """An export cannot smuggle out rows the principal may not see."""
    out = str(tmp_path / "export.parquet")
    _read(catalog, ANALYST, customers).write(out, format="parquet")
    assert bt.read.parquet(out).count() == 2


def test_the_row_filter_is_honored_by_iter_batches(catalog, customers):
    ds = _read(catalog, ANALYST, customers)
    assert sum(b.num_rows for b in ds.iter_batches()) == 2


def test_a_join_restricts_the_governed_side_only(catalog, customers, tmp_path):
    orders = str(tmp_path / "orders.parquet")
    bt.from_pydict({"id": [1, 2, 3, 4], "amt": [10, 20, 30, 40]}).write(orders, format="parquet")
    with bt.security(catalog, ANALYST):
        g = bt.read.parquet(customers)
        o = bt.read.parquet(orders)
    got = g.join(o, on="id").sort("id").to_pydict()
    assert got["id"] == [1, 2] and got["amt"] == [10, 20]


# --- Scoping ------------------------------------------------------------------
def test_a_dataset_keeps_the_policy_of_the_block_it_was_read_in(catalog, customers):
    """Leaving the block must not un-govern a `Dataset` already built inside it."""
    ds = _read(catalog, ANALYST, customers)
    assert ds.count() == 2  # terminal op runs outside the `security()` block


def test_nested_security_blocks_restore_the_outer_policy(catalog, customers):
    with bt.security(catalog, ADMIN):
        with bt.security(catalog, ANALYST):
            inner = bt.read.parquet(customers)
        outer = bt.read.parquet(customers)
    assert inner.count() == 2
    assert outer.count() == 4


def test_two_principals_reading_the_same_table_get_different_plans(catalog, customers):
    with bt.security(catalog, ANALYST):
        a = bt.read.parquet(customers)
    with bt.security(catalog, ADMIN):
        b = bt.read.parquet(customers)
    assert a.columns == ["id", "email", "region"]
    assert b.columns == ["id", "email", "region", "salary"]


def test_an_in_memory_source_has_no_durable_name_and_is_not_governed(catalog):
    """Honest: you cannot write a policy about a dict you are already holding."""
    with bt.security(catalog, INTERN):
        ds = bt.from_pydict({"x": [1, 2]})
    assert ds.count() == 2


# --- Audit --------------------------------------------------------------------
def test_every_governed_read_is_audited(catalog, customers):
    seen = []
    with bt.security(catalog, ANALYST, audit=seen.append):
        bt.read.parquet(customers)
    (event,) = seen
    assert event.principal == "ana"
    assert event.table == customers
    assert event.visible == ("id", "email", "region")
    assert event.denied == ("salary",)
    assert event.masked == ("email",)
    assert event.row_filters == ("row_filter",)


def test_a_denial_is_audited_before_it_is_raised(catalog, customers):
    """The access a compliance review most wants to find is the one that was refused."""
    seen = []
    with pytest.raises(AccessDeniedError), bt.security(catalog, INTERN, audit=seen.append):
        bt.read.parquet(customers)
    (event,) = seen
    assert not event.allowed
    assert event.table == customers


def test_an_ungoverned_read_emits_nothing(catalog, tmp_path):
    other = str(tmp_path / "other.parquet")
    bt.from_pydict({"x": [1]}).write(other, format="parquet")
    seen = []
    with bt.security(catalog, ANALYST, audit=seen.append):
        bt.read.parquet(other)
    assert seen == []


def test_each_read_of_a_governed_table_emits_its_own_event(catalog, customers):
    seen = []
    with bt.security(catalog, ANALYST, audit=seen.append):
        bt.read.parquet(customers)
        bt.read.parquet(customers)
    assert len(seen) == 2


# --- ...including the two schedulings the PR gate never runs -------------------
#
# The header claims "every code path", and the paths above are the ones a default
# `collect()` takes. `collect(spill=True)` and `collect(distributed=True)` are *other
# schedulings* of the same plan, each re-rooting it on relabelled scans and re-applying the
# operators above a breaker (`dist.spill.aggregate`, `dist.executors`). Neither was pinned
# here, and CI installs no Ray, so nothing exercised them: a rewrite that lost the
# governance `Filter`/mask on either would leak rows or PII while this file stayed green.


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda ds: ds.collect(spill=True), id="collect"),
        pytest.param(lambda ds: ds.sort(bt.col("id")).limit(2).collect(spill=True), id="top_n"),
        pytest.param(lambda ds: ds.distinct().collect(spill=True), id="distinct"),
        pytest.param(
            lambda ds: ds.group_by("region").agg(n=bt.count()).collect(spill=True), id="group_by"
        ),
        pytest.param(
            lambda ds: ds.with_columns(
                r=bt.col("id").rank().over(partition_by="region", order_by="id")
            ).collect(spill=True),
            id="window",
        ),
    ],
)
def test_the_row_filter_and_mask_survive_the_out_of_core_path(catalog, customers, build):
    """Spilling is a scheduling, never a way around the policy."""
    with bt.security(catalog, ANALYST):
        got = build(bt.read.parquet(customers)).to_pydict()
    assert "salary" not in got, "a denied column reached the out-of-core result"
    if "region" in got:
        assert set(got["region"]) == {"EU"}, "the row filter was lost under spill"
    if "email" in got:
        assert not ({"a@x.com", "b@x.com", "c@y.com", "d@y.com"} & set(got["email"])), (
            "the PII mask was lost under spill"
        )


def test_the_row_filter_and_mask_survive_the_distributed_path(catalog, customers):
    """Same guarantee across Ray tasks, which the PR gate never executes."""
    ray = pytest.importorskip("ray", reason="distributed path needs Ray")
    if not ray.is_initialized():
        ray.init(address="local", ignore_reinit_error=True, include_dashboard=False)
    with bt.security(catalog, ANALYST):
        got = bt.read.parquet(customers).collect(distributed=True).to_pydict()
    assert "salary" not in got
    assert set(got["region"]) == {"EU"}
    assert not ({"a@x.com", "b@x.com", "c@y.com", "d@y.com"} & set(got["email"]))


def _witness(batch):
    """Report what a `map_batches` fn was actually handed, as data.

    Module level and returning its observation as columns, so the same function works on a
    Ray worker: a closure appending to a list proves nothing once the `fn` runs in another
    process, because the list it appended to stays there.
    """
    import pyarrow as pa

    names = batch.schema.names
    return pa.RecordBatch.from_pydict(
        {
            "saw_cols": [",".join(names)] * batch.num_rows,
            "saw_email": (
                batch.column("email").to_pylist() if "email" in names else [None] * batch.num_rows
            ),
        }
    )


RAW_EMAILS = {"a@x.com", "b@x.com", "c@y.com", "d@y.com"}


def test_a_map_batches_fn_is_handed_the_masked_column_not_the_value(catalog, customers):
    """The policy is a plan rewrite, so a user function reads the rewritten plan's output.

    Worth pinning because `map_batches` is the one place user code touches a batch directly:
    every other path in this file observes the policy through a *result*, where a leak would
    have to survive the whole pipeline. Here the `fn` sees the batch, and if the mask were
    applied after the call it would see the raw column and could copy it out.
    """
    with bt.security(catalog, ANALYST):
        got = bt.read.parquet(customers).map_batches(_witness, batch_format="pyarrow").collect()
    seen = got.to_pydict()
    assert set(seen["saw_cols"]) == {"id,email,region"}, "the fn was handed a denied column"
    assert not (RAW_EMAILS & set(seen["saw_email"])), "the fn was handed the unmasked value"
    assert len(seen["saw_email"]) == 2, "the fn was handed rows the row filter excludes"


def test_a_distributed_map_batches_fn_is_handed_the_same_masked_batch(catalog, customers):
    """The worker re-reads the source, so this is where an ungoverned re-scan would show.

    A Ray task scans the file itself rather than receiving the driver's batches. If the task
    were given the raw scan instead of the governed plan, the `fn` on the worker would see
    every column and every row while the driver's own result still looked correct.
    """
    ray = pytest.importorskip("ray", reason="distributed path needs Ray")
    if not ray.is_initialized():
        ray.init(address="local", ignore_reinit_error=True, include_dashboard=False)

    # Defined *inside* the test on purpose. cloudpickle sends a local function to the worker
    # by value; a module-level one it sends by reference, and the worker then cannot import
    # `test_governance_enforcement` to resolve it (`ModuleNotFoundError`, which reads like a
    # governance failure and is not one).
    def witness(batch):
        import pyarrow as pa

        names = batch.schema.names
        return pa.RecordBatch.from_pydict(
            {
                "saw_cols": [",".join(names)] * batch.num_rows,
                "saw_email": (
                    batch.column("email").to_pylist()
                    if "email" in names
                    else [None] * batch.num_rows
                ),
            }
        )

    with bt.security(catalog, ANALYST):
        got = (
            bt.read.parquet(customers)
            .map_batches(witness, batch_format="pyarrow")
            .collect(distributed=True)
        )
    seen = got.to_pydict()
    assert set(seen["saw_cols"]) == {"id,email,region"}, "a worker handed the fn a denied column"
    assert not (RAW_EMAILS & set(seen["saw_email"])), "a worker handed the fn the unmasked value"
