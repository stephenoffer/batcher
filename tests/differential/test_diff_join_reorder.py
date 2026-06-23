"""Multi-table inner joins reordered by cost still match DuckDB (W3).

The reorder rule rebuilds the join tree in a greedy size-minimizing order; these
confirm it is result-transparent for 3- and 4-way joins, with shared (`on=`) and
distinct (`left_on`/`right_on`) keys, and composed with a filter.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def _emp_dept_region():
    emp = pa.table(
        {
            "emp_id": list(range(1, 13)),
            "dept_id": [10, 10, 20, 20, 30, 30, 10, 20, 30, 10, 20, 30],
            "emp_name": [f"e{i}" for i in range(1, 13)],
        }
    )
    dept = pa.table({"dept_id": [10, 20, 30], "region_id": [1, 1, 2], "dept_name": ["x", "y", "z"]})
    region = pa.table({"region_id": [1, 2], "region_name": ["west", "east"]})
    return emp, dept, region


def test_three_way_join_on_keys_vs_duckdb(duck):
    from conftest import assert_same

    emp, dept, region = _emp_dept_region()
    for name, t in [("emp", emp), ("dept", dept), ("region", region)]:
        duck.register(name, t)
    out = (
        bt.from_arrow(emp)
        .join(bt.from_arrow(dept), on="dept_id")
        .join(bt.from_arrow(region), on="region_id")
        .collect()
    )
    expected = duck.sql("SELECT * FROM emp JOIN dept USING (dept_id) JOIN region USING (region_id)")
    assert_same(out, expected)


def test_three_way_join_with_filter_vs_duckdb(duck):
    from conftest import assert_same

    emp, dept, region = _emp_dept_region()
    for name, t in [("emp", emp), ("dept", dept), ("region", region)]:
        duck.register(name, t)
    out = (
        bt.from_arrow(emp)
        .join(bt.from_arrow(dept), on="dept_id")
        .join(bt.from_arrow(region), on="region_id")
        .filter(col("region_name") == "west")
        .collect()
    )
    expected = duck.sql(
        "SELECT * FROM emp JOIN dept USING (dept_id) JOIN region USING (region_id) "
        "WHERE region_name = 'west'"
    )
    assert_same(out, expected)


def test_three_way_join_distinct_key_names_vs_duckdb(duck):
    from conftest import assert_same

    # Distinct key names (left_on/right_on), disjoint schemas.
    orders = pa.table(
        {"order_id": [1, 2, 3, 4, 5], "cust": [1, 2, 1, 3, 2], "amt": [9, 8, 7, 6, 5]}
    )
    customers = pa.table({"cid": [1, 2, 3], "city": [100, 100, 200]})
    cities = pa.table({"cityid": [100, 200], "cname": ["NY", "LA"]})
    duck.register("orders", orders)
    duck.register("customers", customers)
    duck.register("cities", cities)
    out = (
        bt.from_arrow(orders)
        .join(bt.from_arrow(customers), left_on="cust", right_on="cid")
        .join(bt.from_arrow(cities), left_on="city", right_on="cityid")
        .collect()
    )
    # Batcher's left_on/right_on join drops the right key column, so select exactly
    # the columns Batcher keeps (not the right keys cid / cityid that SELECT * adds).
    expected = duck.sql(
        "SELECT order_id, cust, amt, city, cname "
        "FROM orders JOIN customers ON orders.cust = customers.cid "
        "JOIN cities ON customers.city = cities.cityid"
    )
    assert_same(out, expected)


def test_four_way_join_vs_duckdb(duck):
    from conftest import assert_same

    emp, dept, region = _emp_dept_region()
    grade = pa.table({"region_id": [1, 2], "grade": ["A", "B"]})
    for name, t in [("emp", emp), ("dept", dept), ("region", region), ("grade", grade)]:
        duck.register(name, t)
    out = (
        bt.from_arrow(emp)
        .join(bt.from_arrow(dept), on="dept_id")
        .join(bt.from_arrow(region), on="region_id")
        .join(bt.from_arrow(grade), on="region_id")
        .collect()
    )
    expected = duck.sql(
        "SELECT * FROM emp JOIN dept USING (dept_id) JOIN region USING (region_id) "
        "JOIN grade USING (region_id)"
    )
    assert_same(out, expected)
