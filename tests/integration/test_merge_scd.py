"""Generic file merge, SCD type 1/2/3, and the surrogate-key pattern."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def _rows(table: pa.Table) -> list[dict]:
    d = table.to_pydict()
    return [dict(zip(d.keys(), vals, strict=True)) for vals in zip(*d.values(), strict=True)]


def test_file_merge_update_and_insert(tmp_path):
    tgt = f"{tmp_path}/t.parquet"
    bt.from_arrow(pa.table({"id": [1, 2, 3], "v": ["a", "b", "c"]})).write.parquet(tgt)
    bt.from_arrow(pa.table({"id": [2, 4], "v": ["B", "d"]})).write.merge(tgt, on="id")
    out = bt.read.parquet(tgt).collect().to_pydict()
    assert dict(zip(out["id"], out["v"], strict=True)) == {1: "a", 2: "B", 3: "c", 4: "d"}


def test_file_merge_update_only_no_insert(tmp_path):
    tgt = f"{tmp_path}/t.parquet"
    bt.from_arrow(pa.table({"id": [1, 2], "v": ["a", "b"]})).write.parquet(tgt)
    bt.from_arrow(pa.table({"id": [2, 9], "v": ["B", "z"]})).write.merge(
        tgt, on="id", when_not_matched="ignore"
    )
    out = bt.read.parquet(tgt).collect().to_pydict()
    assert dict(zip(out["id"], out["v"], strict=True)) == {1: "a", 2: "B"}  # id 9 not inserted


def test_file_merge_delete(tmp_path):
    tgt = f"{tmp_path}/t.parquet"
    bt.from_arrow(pa.table({"id": [1, 2, 3], "v": ["a", "b", "c"]})).write.parquet(tgt)
    bt.from_arrow(pa.table({"id": [2], "v": ["x"]})).write.merge(
        tgt, on="id", when_matched="delete", when_not_matched="ignore"
    )
    out = bt.read.parquet(tgt).collect().to_pydict()
    assert sorted(out["id"]) == [1, 3]


def test_merge_into_missing_target_inserts_all(tmp_path):
    tgt = f"{tmp_path}/new.parquet"
    bt.from_arrow(pa.table({"id": [1, 2], "v": ["a", "b"]})).write.merge(tgt, on="id")
    assert bt.read.parquet(tgt).collect().num_rows == 2


def test_scd_type1_is_upsert(tmp_path):
    tgt = f"{tmp_path}/dim.parquet"
    bt.from_arrow(pa.table({"id": [1, 2], "name": ["Ann", "Bob"]})).scd.type1(tgt, keys="id")
    bt.from_arrow(pa.table({"id": [2, 3], "name": ["Bobby", "Cara"]})).scd.type1(tgt, keys="id")
    out = bt.read.parquet(tgt).collect().to_pydict()
    assert dict(zip(out["id"], out["name"], strict=True)) == {1: "Ann", 2: "Bobby", 3: "Cara"}


def test_scd_type2_history(tmp_path):
    tgt = f"{tmp_path}/dim.parquet"
    bt.from_arrow(pa.table({"id": [1, 2], "name": ["Ann", "Bob"]})).scd.type2(
        tgt, keys="id", track=["name"], as_of="2024-01-01"
    )
    bt.from_arrow(pa.table({"id": [1, 3], "name": ["Alice", "Cara"]})).scd.type2(
        tgt, keys="id", track=["name"], as_of="2024-02-01"
    )
    rows = {(r["id"], r["name"]): r for r in _rows(bt.read.parquet(tgt).collect())}
    # id 1 old version expired with a closed valid_to; new version is current.
    assert rows[(1, "Ann")]["is_current"] is False
    assert rows[(1, "Ann")]["valid_from"] == "2024-01-01"
    assert rows[(1, "Ann")]["valid_to"] == "2024-02-01"
    assert rows[(1, "Alice")]["is_current"] is True
    assert rows[(1, "Alice")]["valid_to"] is None
    # id 2 untouched (still current from the first load); id 3 inserted as current.
    assert rows[(2, "Bob")]["is_current"] is True and rows[(2, "Bob")]["valid_from"] == "2024-01-01"
    assert rows[(3, "Cara")]["is_current"] is True


def test_scd_type3_previous_value(tmp_path):
    tgt = f"{tmp_path}/dim.parquet"
    bt.from_arrow(pa.table({"id": [1, 2], "city": ["NYC", "LA"]})).scd.type3(
        tgt, keys="id", track=["city"]
    )
    bt.from_arrow(pa.table({"id": [1, 3], "city": ["SF", "TX"]})).scd.type3(
        tgt, keys="id", track=["city"]
    )
    rows = {r["id"]: r for r in _rows(bt.read.parquet(tgt).collect())}
    assert rows[1] == {"id": 1, "city": "SF", "city_prev": "NYC"}  # changed: prev captured
    assert rows[2] == {"id": 2, "city": "LA", "city_prev": None}  # untouched survivor
    assert rows[3] == {"id": 3, "city": "TX", "city_prev": None}  # new


def test_surrogate_key_via_hash64_is_stable():
    """The documented surrogate-key pattern: a deterministic hash of the business
    key, stable across runs/partitions (here a single composite key column)."""
    t = pa.table({"bk": ["us|a", "eu|b", "us|a"]})
    sk = bt.from_arrow(t).with_columns(sk=col("bk").str.hash64()).collect().to_pydict()
    assert sk["sk"][0] == sk["sk"][2]  # same business key → same surrogate id
    assert sk["sk"][0] != sk["sk"][1]
    assert all(isinstance(v, int) for v in sk["sk"])


# --- the first load, on a transactional target -----------------------------------------


def test_scd_type1_first_load_creates_a_delta_target(tmp_path):
    """An absent target is the dimension's first load, whatever the format.

    `test_scd_type1_is_upsert` above also starts from a fresh target, but a Parquet one — and
    a file merge into a missing path happens to work. `type1` delegated straight to
    `ds.write.merge`, which opens the Delta table, so the *first* run of a type-1 load into a
    transactional target failed with delta-rs's ``Local path "..." does not exist or you don't
    have access!``: a message about permissions for a table that simply was not there yet.

    `type2`, `type3` and `apply_changes` have always had this branch; `type1` was the one
    sibling without it.
    """
    pytest = __import__("pytest")
    pytest.importorskip("deltalake")
    tgt = f"{tmp_path}/dim"

    bt.from_arrow(pa.table({"id": [1, 2], "name": ["Ann", "Bob"]})).scd.type1(
        tgt, keys="id", format="delta"
    )
    first = bt.read.delta(tgt).collect().to_pydict()
    assert dict(zip(first["id"], first["name"], strict=True)) == {1: "Ann", 2: "Bob"}

    # ...and the second load is still an upsert, not a replace.
    bt.from_arrow(pa.table({"id": [2, 3], "name": ["Bobby", "Cara"]})).scd.type1(
        tgt, keys="id", format="delta"
    )
    out = bt.read.delta(tgt).collect().to_pydict()
    assert dict(zip(out["id"], out["name"], strict=True)) == {1: "Ann", 2: "Bobby", 3: "Cara"}


def test_every_scd_mode_bootstraps_its_target(tmp_path):
    """The four `ds.scd` entry points agree about an absent target: it is the first load."""
    pytest = __import__("pytest")
    pytest.importorskip("deltalake")
    incoming = pa.table(
        {
            "id": pa.array([1, 2], pa.int64()),
            "region": pa.array(["us", "eu"]),
            "seq": pa.array([1, 1], pa.int64()),
        }
    )
    ds = bt.from_arrow(incoming)
    loads = {
        "type1": lambda p: ds.scd.type1(p, keys="id", format="delta"),
        "type2": lambda p: ds.scd.type2(
            p, keys="id", track=["region"], as_of="2024-01-01", format="delta"
        ),
        "type3": lambda p: ds.scd.type3(p, keys="id", track=["region"], format="delta"),
        "apply_changes": lambda p: ds.scd.apply_changes(
            p, keys="id", sequence_by="seq", format="delta"
        ),
    }
    for name, load in loads.items():
        target = f"{tmp_path}/{name}"
        load(target)
        got = bt.read.delta(target).collect()
        assert sorted(got.to_pydict()["id"]) == [1, 2], f"{name} did not load both rows"
