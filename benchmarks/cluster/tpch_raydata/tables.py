"""Tables, helpers and the query registry for the Ray Data TPC-H port (see the package)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import batcher as bt
from batcher import col, lit

__all__ = ["LARGE_RESULT", "QUERIES", "TABLE_COLUMNS", "d", "f64", "load_tables", "register"]

# Ray's `common.py` TABLE_COLUMNS: positional parquet names -> TPC-H names.
_NAMES = {
    "region": ("r_regionkey", "r_name", "r_comment"),
    "nation": ("n_nationkey", "n_name", "n_regionkey", "n_comment"),
    "supplier": (
        "s_suppkey",
        "s_name",
        "s_address",
        "s_nationkey",
        "s_phone",
        "s_acctbal",
        "s_comment",
    ),
    "customer": (
        "c_custkey",
        "c_name",
        "c_address",
        "c_nationkey",
        "c_phone",
        "c_acctbal",
        "c_mktsegment",
        "c_comment",
    ),
    "orders": (
        "o_orderkey",
        "o_custkey",
        "o_orderstatus",
        "o_totalprice",
        "o_orderdate",
        "o_orderpriority",
        "o_clerk",
        "o_shippriority",
        "o_comment",
    ),
    "part": (
        "p_partkey",
        "p_name",
        "p_mfgr",
        "p_brand",
        "p_type",
        "p_size",
        "p_container",
        "p_retailprice",
        "p_comment",
    ),
    "partsupp": ("ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"),
    "lineitem": (
        "l_orderkey",
        "l_partkey",
        "l_suppkey",
        "l_linenumber",
        "l_quantity",
        "l_extendedprice",
        "l_discount",
        "l_tax",
        "l_returnflag",
        "l_linestatus",
        "l_shipdate",
        "l_commitdate",
        "l_receiptdate",
        "l_shipinstruct",
        "l_shipmode",
        "l_comment",
    ),
}
TABLE_COLUMNS: dict[str, dict[str, str]] = {
    t: {(f"column{i:02d}" if t == "lineitem" else f"column{i}"): n for i, n in enumerate(names)}
    for t, names in _NAMES.items()
}

# Ray materializes these multi-million-row sorted results rather than pulling them to the
# driver; the harness counts them without holding them twice.
LARGE_RESULT = frozenset({"q3", "q10", "q11", "q18"})


def load_tables(base_uri: str, sf: int) -> dict[str, bt.Dataset]:
    """Lazy scans of every table under ``{base_uri}/sf{sf}``, columns renamed as Ray does."""
    return {
        t: bt.read.parquet(f"{base_uri}/sf{sf}/{t}/*.parquet").rename(mapping)
        for t, mapping in TABLE_COLUMNS.items()
    }


def f64(name: str) -> bt.Expr:
    """Ray's `to_f64`: every measure is cast to float64 before arithmetic."""
    return col(name).cast("float64")


def d(y: int, m: int, day: int) -> bt.Expr:
    return lit(date(y, m, day))


Query = Callable[[dict[str, bt.Dataset], int], bt.Dataset]
QUERIES: dict[str, Query] = {}


def register(fn: Query) -> Query:
    QUERIES[fn.__name__] = fn
    return fn
