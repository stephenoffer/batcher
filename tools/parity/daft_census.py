"""Daft expression census: every public method on `daft.Expression`, resolved against
Batcher's *whole* surface.

The point of this one is the resolution step. Daft keeps everything on a flat namespace
(`list_sum`, `upper`, `to_snake_case`), Batcher pushes breadth onto accessor namespaces
(`.list.sum`, `.str.to_uppercase`, `.str.to_case("snake")`). A naive `dir()` difference
therefore reports Batcher as missing ~220 functions it has, which is the same denominator
error that made the DuckDB census read 54% instead of 79%.

So a Daft name counts as present when *any* of these resolves in Batcher:

* the same name at the top level or in any accessor namespace;
* the name with a `list_`/`str_`/`dt_`/… prefix stripped (`list_sum` → `.list.sum`);
* a recorded alias, where the two engines simply chose different words for one thing
  (`avg`/`mean`, `stddev`/`std`, `is_inf`/`is_infinite`, `eq_null_safe`/`eq_missing`);
* a recorded *parameterization*, where Daft spells as N functions what Batcher spells as
  one function with an argument — the eight case converters against
  `.str.to_case(style)`, and Daft's 35 `as_<type>` casts against `.cast(dtype)`.

That last category is the one worth reading twice. `as_int8`…`as_uuid` is 35 names for one
capability, exactly like DuckDB's 135 `icu_collate_*`. Counting them as 35 missing
functions would say more about Daft's API style than about either engine.

Output mirrors the other two censuses: match / gap / out_of_scope, with the denominator
being match + gap.
"""

from __future__ import annotations

import json
import sys
import warnings

warnings.filterwarnings("ignore")

import daft  # noqa: E402

import batcher as bt  # noqa: E402

NAMESPACES = ("str", "dt", "list", "struct", "json", "map", "image", "audio", "video")

# Daft's name → the Batcher name for the same thing. Only where both engines have the
# capability and merely disagree about the word.
ALIASES = {
    "avg": "mean",
    "stddev": "std",
    "is_inf": "is_infinite",
    "not_nan": "is_nan",
    "not_null": "is_not_null",
    "eq_null_safe": "eq_missing",
    "power": "pow",
    "negate": "neg",
    "length": "len",
    "find": "position",
    "explode": "flatten",
    "percentile": "quantile",
    "approx_percentiles": "approx_quantile",
    "string_agg": "join",
    "value_counts": "histogram",
    "pearson_correlation": "corr",
    "levenshtein_distance": "levenshtein",
    "damerau_levenshtein_distance": "damerau_levenshtein",
    "jaccard_similarity": "jaccard",
    "date_trunc": "truncate",
    "day_of_month": "day",
    "unix_date": "epoch_days",
    "convert_time_zone": "convert_timezone",
    "replace_time_zone": "replace_timezone",
    "shift_left": "shl",
    "shift_right": "shr",
    "dot_product": "dot",
    "list_distinct": "unique",
    "list_agg": "agg_list",
    "parse_url": "parse_uri",
    "lag": "shift",
    "first_value": "first",
    "last_value": "last",
    "regexp": "regexp_matches",
    "length_bytes": "octet_length",
}

# Daft spells these as N functions where Batcher takes an argument. Present in Batcher iff
# the named method exists — the styles/types themselves are checked by that method's tests.
PARAMETERIZED = {
    "to_case": (
        "to_snake_case", "to_camel_case", "to_kebab_case", "to_title_case",
        "to_upper_camel_case", "to_upper_snake_case", "to_upper_kebab_case",
    ),
    "cast": tuple(n for n in dir(daft.expressions.Expression) if n.startswith("as_")),
}  # fmt: skip

# Names that are not expression functions at all: plan/API plumbing, or Daft's own
# extension points. Excluding them keeps the denominator about expression surface.
PLUMBING = frozenset(
    {
        "apply", "udf", "over", "alias", "cast", "if_else", "is_column", "is_literal",
        "to_arrow_expr", "column_name", "name", "eq", "ne", "lt", "le", "gt", "ge",
        "serialize", "deserialize", "try_deserialize", "try_serialize",
    }
)  # fmt: skip


def batcher_surface() -> set[str]:
    """Every public name a Daft method could correspond to.

    Both the `Expr` surface with its accessor namespaces flattened, *and* the module-level
    functions — Batcher spells `coalesce` and `atan2` as `bt.coalesce`/`bt.atan2` where
    Daft has them as methods, and an `Expr`-only comparison reports both as missing.
    """
    expr = bt.col("x")
    names = {n for n in dir(expr) if not n.startswith("_")}
    for namespace in NAMESPACES:
        try:
            names |= {n for n in dir(getattr(expr, namespace)) if not n.startswith("_")}
        except AttributeError:
            continue
    return names | {n for n in dir(bt) if not n.startswith("_")}


def candidates(name: str) -> set[str]:
    """Every Batcher spelling that would satisfy `name`."""
    out = {name, name.replace("_", "")}
    for prefix in ("list_", "str_", "dt_", "image_", "map_", "struct_", "json_", "to_"):
        if name.startswith(prefix):
            out.add(name[len(prefix) :])
    if name in ALIASES:
        out.add(ALIASES[name])
    return out


def main() -> None:
    surface = batcher_surface()
    expression = daft.expressions.Expression
    daft_names = sorted(
        n for n in dir(expression) if not n.startswith("_") and callable(getattr(expression, n))
    )

    parameterized = {n: m for m, names in PARAMETERIZED.items() for n in names}
    match, gap, scope = [], [], []
    for name in daft_names:
        if name in PLUMBING:
            scope.append((name, "API plumbing, not an expression function"))
            continue
        covered_by = parameterized.get(name)
        if covered_by is not None:
            if covered_by in surface:
                scope.append((name, f"one argument of Batcher's `{covered_by}`"))
            else:
                gap.append((name, f"no `{covered_by}` to parameterize"))
            continue
        (match if candidates(name) & surface else gap).append(name)

    print(
        json.dumps(
            {"match": sorted(m for m in match), "gap": sorted(gap, key=str), "out_of_scope": scope},
            indent=1,
            default=str,
        )
    )
    scored = len(match) + len(gap)
    pct = 100.0 * len(match) / scored if scored else 0.0
    print(
        f"\n# scored={scored} match={len(match)} ({pct:.0f}%) gap={len(gap)} "
        f"| out_of_scope={len(scope)} (of {len(daft_names)} listed)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
