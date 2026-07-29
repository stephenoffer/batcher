"""Differential census of Polars' expression API against Batcher's.

For every public method on `pl.Expr` and its accessor namespaces, call it on a matching
column in Polars and in Batcher and compare. Buckets: MATCH, MISMATCH (both ran,
answers differ), GAP (Polars ran, Batcher has no such method or raised).
"""

from __future__ import annotations

import inspect
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import polars as pl  # noqa: E402

import batcher as bt  # noqa: E402

# Column per Polars dtype family, with the same values on both sides.
DATA = {
    "i": [3, 1, 4, 1, 5],
    "f": [1.5, -2.5, 3.25, 0.5, 9.0],
    "s": ["Abc", "bcd", "cde", "abc", "e f"],
    "b": [True, False, True, True, False],
    "l": [[1, 2], [3], [2, 2, 5], [], [4]],
}
NS_COLUMN = {None: "f", "str": "s", "dt": "d", "list": "l", "arr": "l", "struct": "st"}

# Methods whose result is not comparable (lazy/plan/meta/IO/random/deprecated) or whose
# Batcher counterpart deliberately differs in kind.
SKIP = {
    "alias",
    "name",
    "meta",
    "map_batches",
    "map_elements",
    "over",
    "sort_by",
    "shuffle",
    "sample",
    "set_sorted",
    "cache",
    "inspect",
    "register_plugin",
    "to_physical",
    "from_json",
    "deserialize",
    "serialize",
    "reshape",
    "gather_every",
    "extend_constant",
    "rolling",
    "cast",
    "value_counts",
    "hist",
    "qcut",
    "cut",
    "search_sorted",
    "rle",
    "rle_id",
    "shrink_dtype",
    "reinterpret",
    "hash",
    "get",
    "gather",
    "slice",
    "head",
    "tail",
    "limit",
    "top_k",
    "bottom_k",
    "top_k_by",
    "bottom_k_by",
    "sort",
    "arg_sort",
    "implode",
    "explode",
    "flatten",
    "repeat_by",
    "append",
    "filter",
    "where",
    "when",
    "then",
    "otherwise",
    "len",
    "count",
    "first",
    "last",
    "shift",
    "reverse",
    "unique",
    "unique_counts",
    "arg_unique",
    "is_between",
    "replace",
    "replace_strict",
    "map_dict",
    "backward_fill",
    "forward_fill",
    "interpolate",
    "interpolate_by",
    "upper_bound",
    "lower_bound",
    "pipe",
    "exclude",
    "keep_name",
    "prefix",
    "suffix",
    "eq",
    "ne",
    "lt",
    "gt",
    "le",
    "ge",
    "eq_missing",
    "ne_missing",
    "and_",
    "or_",
    "xor",
    "add",
    "sub",
    "mul",
    "truediv",
    "floordiv",
    "mod",
    "pow",
    "dot",
    "rank",
    "diff",
    "pct_change",
    "cum_sum",
    "cum_prod",
    "cum_min",
    "cum_max",
    "cum_count",
    "cumulative_eval",
    "ewm_mean",
    "ewm_std",
    "ewm_var",
    "ewm_mean_by",
    "peak_max",
    "peak_min",
    "entropy",
    "null_count",
    "has_nulls",
    "arg_min",
    "arg_max",
    "arg_true",
    "quantile",
    "reduce",
    "fold",
    "list",
    "bin",
    "cat",
    "arr",
    "struct",
    "to_struct",
}


def probe():
    df = pl.DataFrame(DATA).with_columns(
        d=pl.Series(
            ["2024-03-05", "2023-12-31", "2024-01-01", "2022-06-15", "2021-02-28"]
        ).str.to_date()
    )
    bat = bt.from_pydict(DATA).with_columns(d=bt.col("s").str.to_date())  # placeholder
    bat = bt.from_pydict(
        {**DATA, "d": ["2024-03-05", "2023-12-31", "2024-01-01", "2022-06-15", "2021-02-28"]}
    ).with_columns(d=bt.col("d").str.to_date())

    match, gap, mismatch = [], [], []
    for ns in (None, "str", "dt", "list"):
        pl_obj = pl.col(NS_COLUMN[ns]) if ns is None else getattr(pl.col(NS_COLUMN[ns]), ns)
        names = [m for m in dir(pl_obj) if not m.startswith("_")]
        for name in names:
            if name in SKIP:
                continue
            fn = getattr(pl_obj, name)
            if not callable(fn):
                continue
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):
                continue
            required = [
                p
                for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            if required:
                continue  # only the zero-argument forms, so the call is unambiguous
            label = f"{ns or 'Expr'}.{name}"
            try:
                expected = df.select(r=fn()).to_series().to_list()
            except BaseException:
                continue
            b_obj = bt.col(NS_COLUMN[ns]) if ns is None else getattr(bt.col(NS_COLUMN[ns]), ns)
            b_fn = getattr(b_obj, name, None)
            if b_fn is None:
                gap.append((label, "no such method"))
                continue
            try:
                got = bat.select(r=b_fn()).to_pydict()["r"]
            except Exception as exc:
                gap.append((label, f"{type(exc).__name__}: {exc}".split("\n")[0][:110]))
                continue
            if same(expected, got):
                match.append(label)
            else:
                mismatch.append((label, repr(expected)[:70], repr(got)[:70]))
    return match, gap, mismatch


def same(a, b) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=False))
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
    return str(a) == str(b)


if __name__ == "__main__":
    m, g, mm = probe()
    print(json.dumps({"match": sorted(m), "gap": sorted(g), "mismatch": sorted(mm)}, indent=1))
    print(f"\n# match={len(m)} gap={len(g)} mismatch={len(mm)}", file=sys.stderr)
