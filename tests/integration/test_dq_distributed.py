"""A data contract answers the same on one node and on many.

Every `ds.dq` terminal lowers to relational operators that already distribute — a FILTER, a
keyless AGGREGATE, ``count() OVER (PARTITION BY keys)``, and a LEFT JOIN against distinct
reference keys — so distribution should be a scheduling detail and nothing else. That is
exactly the kind of claim that holds until an operator is added that partitions the state
badly, and then fails as *wrong counts* rather than as an error.

The two constraints that carry state across partitions are the ones under test: uniqueness,
whose window count is meaningless if a key's rows land on different workers, and referential
integrity, whose join must not multiply the rows being checked. The relation-level checks
come along because their aggregates merge, and a mean that reassociates differently per
partition would show up here first.

CI installs no Ray, so this suite never runs in the PR gate — see `just lint-skips`.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


def _orders(n: int = 50_000) -> pa.Table:
    """Keys that repeat, amounts that go negative, and customer ids that do not all resolve."""
    rng = np.random.default_rng(11)
    return pa.table(
        {
            # A small key space over many rows, so every key spans several partitions.
            "order_id": rng.integers(0, n // 4, n).astype("int64"),
            "customer_id": rng.integers(0, 200, n).astype("int64"),
            "amount": (rng.normal(50.0, 20.0, n)).astype("float64"),
        }
    )


def _customers() -> bt.Dataset:
    return bt.from_arrow(pa.table({"customer_id": np.arange(0, 150, dtype="int64")}))


def _rows(table: pa.Table) -> set:
    return {
        tuple(round(v, 6) if isinstance(v, float) else v for v in row.values())
        for row in table.to_pylist()
    }


def test_row_level_split_is_identical_distributed():
    ds = bt.from_arrow(_orders())
    gate = ds.dq.positive("amount").not_null("customer_id")
    local = gate.drop().collect()
    remote = gate.drop().collect(distributed=True)
    assert local.num_rows == remote.num_rows
    assert _rows(local) == _rows(remote)


def test_uniqueness_split_is_identical_distributed():
    """The window count must see a key's whole partition, wherever its rows landed."""
    ds = bt.from_arrow(_orders())
    gate = ds.dq.unique("order_id")
    clean_local, bad_local = gate.quarantine()
    clean_remote, bad_remote = gate.quarantine()
    local = clean_local.collect(), bad_local.collect()
    remote = clean_remote.collect(distributed=True), bad_remote.collect(distributed=True)
    assert local[0].num_rows == remote[0].num_rows
    assert local[1].num_rows == remote[1].num_rows
    assert _rows(local[0]) == _rows(remote[0])
    # The split stays total on both paths.
    assert remote[0].num_rows + remote[1].num_rows == ds.count()


def test_referential_integrity_is_identical_distributed():
    """A LEFT JOIN against distinct reference keys must not multiply the checked rows."""
    ds = bt.from_arrow(_orders())
    gate = ds.dq.references("customer_id", to=_customers())
    local = gate.drop().collect()
    remote = gate.drop().collect(distributed=True)
    assert local.num_rows == remote.num_rows
    assert _rows(local) == _rows(remote)


def test_violation_counts_are_identical_distributed():
    ds = bt.from_arrow(_orders())
    gate = ds.dq.positive("amount").unique("order_id").references("customer_id", to=_customers())
    local = gate.validate()
    # `validate` runs its own aggregates; distribute the same relation and re-measure.
    remote_source = bt.from_arrow(ds.collect(distributed=True))
    remote = gate.on(remote_source).validate()
    assert local.violations == remote.violations
    assert local.rows == remote.rows


def test_relation_level_measurements_agree_distributed():
    """Aggregates merge, so a bound holds on both paths; floats only up to reassociation."""
    ds = bt.from_arrow(_orders())
    gate = ds.dq.row_count_between(1).mean_between("amount", 40.0, 60.0)
    local = gate.validate()
    remote = gate.on(bt.from_arrow(ds.collect(distributed=True))).validate()
    assert local.ok == remote.ok
    assert local.result("row_count_between(1, None)").value == (
        remote.result("row_count_between(1, None)").value
    )
    local_mean = local.result("mean_between(amount, 40.0, 60.0)").value
    remote_mean = remote.result("mean_between(amount, 40.0, 60.0)").value
    assert local_mean == pytest.approx(remote_mean, rel=1e-9)


def test_annotation_labels_the_same_rows_distributed():
    ds = bt.from_arrow(_orders())
    gate = ds.dq.positive("amount").unique("order_id")
    local = gate.annotate().collect()
    remote = gate.annotate().collect(distributed=True)
    assert _rows(local) == _rows(remote)
