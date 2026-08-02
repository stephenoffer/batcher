"""Group-by aggregation on a dataframe backend, matching the CPU engine's null semantics.

Aggregation is where a translated GPU plan is easiest to get subtly wrong, because the
dataframe libraries' defaults disagree with Arrow/SQL on five points that never show up in
a smoke test and each produce a *wrong answer* rather than an error:

* a **null group key** is a group. `groupby` drops it by default, so a query whose key column
  had nulls silently lost rows from its result;
* a **float group key of `-0.0` and `0.0` is one group**. The libraries group on a hash of the
  bits and make it two, splitting a sum between them;
* the **sum of an all-null group is null**, not `0.0`. `groupby.sum()` returns `0.0`, which
  reads as a real measurement;
* **`all` and `any` over an all-null group are null**, not `True` and `False`. The libraries
  skip the nulls and return the fold's identity, so a `.all()` over a group whose values were
  every one of them null reads as "every one of them was true";
* **variance and standard deviation are the sample** forms (`ddof=1`), which is the libraries'
  default but not the one a "population" reading would pick.

A sixth disagreement is *declined* rather than corrected: over a `NaN` the four order
statistics cannot be reconciled, and `_NAN_SAFE` records which reductions can. Everything else
runs on the device with a `NaN` present.

Each aggregate is computed off one shared `GroupBy`, so the grouping is built once on the
device and every reduction reuses it. The columns those reductions read are materialized
*before* the grouping is taken, which is what lets a per-column check see them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.core.gpu_plan.backend import Unsupported
from batcher.core.gpu_plan.exprs import eval_expr

if TYPE_CHECKING:
    from batcher.core.gpu_plan.backend import DfBackend

__all__ = ["aggregate", "supported_aggregate"]

# Reductions that are a method of the same name on both backends' `GroupBy`, taking no
# argument and already agreeing with the engine on nulls (nulls are skipped; an all-null
# group yields null, except `count`, which counts non-nulls and so yields 0).
_PLAIN = {
    "count": "count",
    "min": "min",
    "max": "max",
    "mean": "mean",
    "median": "median",
    "count_distinct": "nunique",
    # `any_value` is "the first non-null value", which is what both libraries' `first` returns.
    "any_value": "first",
}

# Reductions needing an empty group nulled so it yields null rather than the operator's
# identity element — the `sum` of nothing is not `0`. `product` needs the same and is handled
# on its own, because it also has to be retyped (see `_reduce`).
_MIN_COUNT = {"sum": "sum"}

# Boolean folds with the same problem `_MIN_COUNT` solves, but no `min_count` to solve it
# with. `all` and `any` skip nulls and then return their identity — `True` and `False` — for a
# group that had nothing to fold, where the engine (and SQL) return null. Left alone, a
# `.all()` over a group whose values were all null reads as "every one of them was true".
_BOOL_FOLD = {"bool_and": "all", "bool_or": "any"}

# Sample-moment reductions (`ddof=1`): a one-row group has no sample variance, so both the
# engine and the libraries return null for it. `skew` is the same family — the adjusted
# Fisher-Pearson form both the engine and the libraries compute.
_SAMPLE_MOMENT = {"var": "var", "stddev": "std", "skewness": "skew"}

_SUPPORTED = (
    frozenset(_PLAIN)
    | frozenset(_MIN_COUNT)
    | frozenset(_BOOL_FOLD)
    | frozenset(_SAMPLE_MOMENT)
    | {
        "count_star",
        "product",
        "quantile",
    }
)


def supported_aggregate(ir: dict) -> bool:
    """Whether one `aggregate` RelOp node is translatable to the dataframe backends.

    Args:
        ir: The `aggregate` node's JSON IR.

    Returns:
        True when every group key and every reduction in the node is translatable.
    """
    return all(a.get("func") in _SUPPORTED for a in ir["aggregates"])


def _key_columns(df, ir: dict, be: DfBackend) -> tuple[list[str], list[str], dict[str, str]]:
    """Materialize the group keys into `df`, returning their names, output aliases, and the
    source column of every key that had to be normalized.

    A key that is already a plain column is used in place; a computed key (``group_by(x2=col
    ("x") + 1)``) is evaluated into a private column first. Supporting the computed form here
    is what keeps a `GROUP BY <expression>` on the device instead of dropping the whole chain
    to the CPU engine.

    The third return value is what lets the caller *label* a normalized group with the value
    the engine would show rather than with the normalized one — see `_normalized_key`.
    """
    names: list[str] = []
    aliases: list[str] = []
    sources: dict[str, str] = {}
    for i, gk in enumerate(ir["group_keys"]):
        expr = gk["expr"]
        if expr.get("e") == "col":
            name = expr["name"]
        else:
            name = f"__bt_gk{i}"
            df[name] = be.column(eval_expr(expr, df, be), df)
        key = _normalized_key(df, name, be, slot=i)
        if key != name:
            sources[key] = name
        names.append(key)
        aliases.append(gk["alias"])
    return names, aliases, sources


def _normalized_key(df, name: str, be: DfBackend, *, slot: int) -> str:
    """`name`, or a private copy of it with negative zero folded onto zero.

    IEEE says `-0.0 == 0.0`, and so do the engine and SQL, so the two belong in one group.
    Both dataframe libraries group by a *hash* of the value instead, and the two zeros have
    different bit patterns — so a float key carrying both silently returned two groups where
    the engine returns one, splitting a sum between them. This is the failure a distributed
    aggregate is most exposed to, since a shard that happened to see only one of the two
    zeros produces a partial nothing later folds together.

    Adding zero is the fold: `-0.0 + 0.0` is `+0.0`, and `x + 0.0` is `x` for every other
    value, including the infinities and `NaN`. Only float keys pay for it.

    The fold decides *identity* only, never the label. Emitting the normalized value would
    make a group of `{-0.0, 0.0}` come back as `0.0` where the engine returns the first one it
    saw, `-0.0` — the same group under a different name. That reads as harmless until a
    sharded fan-out concatenates a device shard's `0.0` with a CPU-recovered shard's `-0.0`
    and gets two groups where the engine has one, which is the float-key split this fold
    exists to prevent, reintroduced one level up. `aggregate` therefore re-labels the group
    from `sources` with the first value it actually contained.
    """
    if not be.is_float(df[name]):
        return name
    normalized = f"__bt_gz{slot}"
    df[normalized] = df[name] + 0.0
    return normalized


def _null_if_empty(series, reduced):
    """`reduced`, nulled for every group that had no non-null value to fold.

    The engine's rule, and SQL's: a `sum` over nothing is null, not `0`; a `product` is not `1`;
    an `all` is not `True`. Both dataframe libraries return the operator's identity instead,
    which reads as a real measurement.

    pandas spells the fix `min_count=1`, and that is what this used to pass — but **cuDF has no
    `min_count`**, and raises `NotImplementedError` for it on every reduction that takes one.
    So the one parameter that made `sum` correct was also the one that made `sum` impossible on
    a device, and since `sum` is in essentially every analytical query, the GPU backend declined
    essentially every analytical query and fell back to the host. It cost the whole path, and it
    was invisible from here because the verification backend is pandas, where `min_count` works.

    Counting and masking is the same answer through an operation both libraries have. It is
    what the boolean folds already did, for the same reason — they never had a `min_count` to
    reach for — so this is now one statement rather than two.
    """
    return reduced.where(_call(series, "count") > 0)


def _as_int64(reduced, be: DfBackend):
    """A counting reduction in the engine's `int64`, whatever width the library chose for it.

    **cuDF answers `nunique` in `int32` and pandas answers it in `int64`.** So
    `COUNT(DISTINCT ...)` came back a different *column* on a device than on the host with every
    value agreeing — the one difference this package's pandas-backed tests are structurally
    unable to see, and the third defect of exactly that shape here. Found by comparing the
    device against the CPU engine on ClickBench q08 and q09, where `COUNT(DISTINCT UserID)`
    returned `int32`.

    Applied to the counting reductions only. They are the ones whose result type is fixed by the
    engine rather than carried from their input, so restating it here cannot drift from what the
    input said; a `sum` or a `min` keeps its column's own type and is left alone.
    """
    import pyarrow as pa

    return reduced.astype(be.dtype(pa.int64()))


def _reduce(grouped, spec: dict, column: str | None, be: DfBackend):
    """One reduction over the shared `GroupBy`, as a Series indexed by the group key."""
    func = spec["func"]
    if func == "count_star":
        return _as_int64(grouped.size(), be)
    series = grouped[column]
    if func in _PLAIN:
        reduced = _call(series, _PLAIN[func])
        return _as_int64(reduced, be) if func in _COUNTING else reduced
    if func == "product":
        # The engine (and DuckDB) answer `product` in **double** whatever the input's type,
        # because the running product of a bigint column leaves its range almost immediately.
        # Both libraries' `prod` keeps the integer instead, so an int64 column's product came
        # back int64 — the right number until it overflows, and a column a CPU-recovered shard's
        # double cannot be concatenated with either way.
        import pyarrow as pa

        reduced = _null_if_empty(series, _call(series, "prod"))
        return reduced.astype(be.dtype(pa.float64()))
    if func in _MIN_COUNT:
        return _null_if_empty(series, _call(series, _MIN_COUNT[func]))
    if func in _BOOL_FOLD:
        return _null_if_empty(series, _call(series, _BOOL_FOLD[func]))
    if func in _SAMPLE_MOMENT:
        return _call(series, _SAMPLE_MOMENT[func])
    if func == "quantile":
        return _call(series, "quantile", float(spec["param"]))
    raise Unsupported(f"aggregate {func}")


#: Reductions whose answer over a `NaN`-bearing column already matches the engine, because both
#: propagate the `NaN` through the arithmetic (or count it as the value it is). `min` and `max`
#: are absent and are the reason this list exists: the engine orders `NaN` above every number,
#: so it wins a maximum and loses a minimum, while both libraries treat it as missing for those
#: two alone. `median` and `quantile` are absent for the same family of reason: over a group
#: whose values are all `NaN` they report missing where the engine reports `NaN`. Every entry
#: that is here was checked against the engine before being listed.
_NAN_SAFE = frozenset(
    {
        "sum",
        "mean",
        "count",
        "count_star",
        "count_distinct",
        "any_value",
        "product",
        "var",
        "stddev",
        "skewness",
        "bool_and",
        "bool_or",
    }
)


def _materialize_inputs(df, ir: dict, be: DfBackend) -> list[str | None]:
    """Each reduction's input column, added to `df` when it is computed rather than read.

    Runs *before* the `GroupBy` is built, so every column a reduction reads is already in the
    frame the grouping was taken over. `count_star` reads no column and gets `None`.

    Declines the four order statistics — `min`, `max`, `median`, `quantile` — over a
    `NaN`-bearing column, and only those. The whole aggregate used to fall back for *any*
    reduction over such a column, so a division by zero somewhere upstream cost the entire query
    its device even when every reduction in it handles `NaN` exactly as the engine does.
    """
    out: list[str | None] = []
    for slot, spec in enumerate(ir["aggregates"]):
        func = spec["func"]
        if func == "count_star":
            out.append(None)
            continue
        expr = spec["input"]
        if expr.get("e") == "col":
            name = expr["name"]
        else:
            name = f"__bt_ag{slot}"
            df[name] = be.column(eval_expr(expr, df, be), df)
        if func not in _NAN_SAFE and be.has_nan(df[name]):
            raise Unsupported(f"{func} over the NaN-bearing column {name!r}")
        out.append(name)
    return out


def _call(series, name: str, *args, **kwargs):
    """Invoke `name` on a grouped column, declining rather than guessing when it is absent.

    cuDF's `GroupBy` surface is a subset of pandas', so a reduction pandas offers may not
    exist on the device. An `AttributeError` here would escape as a crash; `Unsupported`
    routes the stage to the CPU engine, which is the contract every other case follows.
    """
    method = getattr(series, name, None)
    if method is None:
        raise Unsupported(f"grouped {name}")
    try:
        return method(*args, **kwargs)
    except (TypeError, NotImplementedError) as exc:
        raise Unsupported(f"grouped {name}: {exc}") from exc


def aggregate(df, ir: dict, be: DfBackend):
    """Apply one `aggregate` RelOp to `df`, returning the grouped result.

    Args:
        df: The dataframe to reduce.
        ir: The `aggregate` node's JSON IR.
        be: The dataframe backend to compute on.

    Returns:
        A dataframe of one row per group, carrying the key aliases then the aggregate aliases.

    Raises:
        Unsupported: For a reduction outside the translated subset.
    """
    if not ir["group_keys"]:
        return _global(df, ir, be)
    # Shallow: the private key and input columns below are added to this frame, and the caller's
    # frame must not grow them. Sharing every existing column's buffer makes that free.
    df = df.copy(deep=False)
    keys, aliases, sources = _key_columns(df, ir, be)
    inputs = _materialize_inputs(df, ir, be)
    # `dropna=False`: a null key is a group, exactly as it is in the engine and in SQL.
    # The libraries drop it by default, which silently deletes rows from the answer.
    grouped = df.groupby(keys, sort=False, dropna=False)
    columns = {}
    for spec, column in zip(ir["aggregates"], inputs, strict=True):
        columns[spec["alias"]] = _reduce(grouped, spec, column, be)
    # The un-normalized value each normalized group is labelled by. `sort=False` keeps the
    # groups in first-seen order, and `first()` picks each group's first row, so the label is
    # the one the engine's own first-seen representative would be.
    labels = {f"__bt_lbl{i}": grouped[src].first() for i, src in enumerate(sources.values())}
    out = be.lib.DataFrame(columns | labels)
    if not (columns or labels):
        out = be.lib.DataFrame(index=grouped.size().index)
    out = out.reset_index()
    for (key, _), label in zip(sources.items(), labels, strict=True):
        out[key] = out[label]
    # `reset_index` restores the key columns under their *source* names; rename to the
    # aliases the plan asked for, then order as the plan does (keys first).
    renames = {src: alias for src, alias in zip(keys, aliases, strict=True) if src != alias}
    if renames:
        out = out.rename(columns=renames)
    return out[[*aliases, *(a["alias"] for a in ir["aggregates"])]]


def _global(df, ir: dict, be: DfBackend):
    """A keyless aggregate — one row over the whole frame.

    Reached by `agg()` with no `group_by`, and by every distributed *combine* step, so it
    cannot be left to the CPU engine without giving up the whole multi-GPU reduce path.
    Modeled as a single constant group so one code path serves both.

    A keyless aggregate always returns **one** row, including over no rows at all — that is
    what distinguishes it from a grouped one, which returns a row per group and so returns
    none. Grouping an empty frame produces no groups, so the empty case is finished by hand.
    """
    key = "__bt_all"
    # Shallow rather than deep: the only mutation is the constant key column added next, and a
    # deep copy here duplicated the whole frame on the device — for a keyless aggregate, which is
    # every distributed *combine* step, so the fan-out paid for a second copy of every partial.
    df = df.copy(deep=False)
    df[key] = 0
    inputs = _materialize_inputs(df, ir, be)
    grouped = df.groupby([key], sort=False, dropna=False)
    columns = {}
    for spec, column in zip(ir["aggregates"], inputs, strict=True):
        columns[spec["alias"]] = _reduce(grouped, spec, column, be)
    out = be.lib.DataFrame(columns).reset_index(drop=True)
    if not len(df):
        out = _empty_global_row(out, ir)
    return out[[a["alias"] for a in ir["aggregates"]]]


#: Reductions that count rather than measure, so their answer over no rows is `0` and not null.
_COUNTING = frozenset({"count", "count_star", "count_distinct"})


def _empty_global_row(out, ir: dict):
    """The one row a keyless aggregate over an empty frame returns.

    Built by reindexing the empty result rather than by constructing a row from scratch, which
    is what keeps each column's dtype: a `sum` over an empty float column must come back as a
    null *float*, not as a null of no type, or the shard contributes a column its neighbours
    cannot be concatenated with.
    """
    row = out.reindex(range(1))
    for spec in ir["aggregates"]:
        if spec["func"] in _COUNTING:
            row[spec["alias"]] = 0
    return row.reset_index(drop=True)
