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

from batcher.core.gpu_plan.backend import Unsupported, call_or_decline
from batcher.core.gpu_plan.exprs import eval_expr
from batcher.plan.ir_tags import COUNTING_AGGS

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
    # A non-linear `interpolation` is declined: the backends' `quantile` is only verified
    # against the engine's linear form, and no GPU run has recorded the others. `order_by`
    # (an ordered `array_agg`) is declined for the same reason and one more: the element
    # order is the engine's key-then-value rule, which no dataframe `groupby` reproduces, and
    # a list aggregate is not translated at all yet. Checked here rather than left to
    # `_SUPPORTED`, so translating `list_agg` later cannot silently drop the order.
    return all(
        a.get("func") in _SUPPORTED and a.get("interpolation") is None and not a.get("order_by")
        for a in ir["aggregates"]
    )


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


def _null_if_empty(counts, reduced):
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

    Takes the per-group counts rather than the grouped column, so the caller decides how they
    were obtained — from the one fused `agg` pass alongside every other reduction, or from a
    `count()` of its own. Sourcing them here would have made the mask a second full pass over
    the data for every reduction that needs one.
    """
    return reduced.where(counts > 0)


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


#: The `GroupBy` methods each reduction family needs. A reduction that needs an empty group
#: nulled needs `count` beside its own fold, which is why several map to two.
#:
#: This table is what makes the aggregate **one pass**. Every reduction used to be issued
#: separately against the shared `GroupBy` — `grouped[c].sum()`, then `grouped[c].mean()`, and
#: so on — and each of those is a full hash group-by over the shard on the device. TPC-H q1 has
#: eight reductions plus two key labels, so a 10 M-row shard was grouped **ten times**: measured
#: on a T4, 0.25 s of a 0.40 s shard, against 0.12 s to read the shard off storage and 0.03 s to
#: filter it. Collected into one `agg` call they are one pass and cuDF fuses the reductions
#: itself.
_METHODS: dict[str, tuple[str, ...]] = {
    **{func: (method,) for func, method in _PLAIN.items()},
    **{func: (method, "count") for func, method in _MIN_COUNT.items()},
    **{func: (method, "count") for func, method in _BOOL_FOLD.items()},
    **{func: (method,) for func, method in _SAMPLE_MOMENT.items()},
    "product": ("prod", "count"),
}


def _fused(grouped, columns: dict[str, list[str]]):
    """Every reduction the node needs, in one `agg` pass, or `None` to issue them separately.

    `None` is not a failure: it means this backend would not take the fused form, and the
    caller then reduces column by column exactly as it always did. The two produce identical
    values by construction — same `GroupBy`, same method names — so this is a scheduling
    choice, and keeping the unfused path is what makes it safe to take.
    """
    if not columns:
        return None
    try:
        return call_or_decline(grouped, "agg", {c: list(m) for c, m in columns.items()})
    except Unsupported:
        return None


def _series_reader(grouped, fused):
    """`(column, method) -> Series`, from the fused frame where it is there and the group else.

    One statement of "where does this reduction's raw series come from", so the semantics below
    are written once and work whether or not the fusion was taken.
    """

    def read(column: str, method: str):
        if fused is not None:
            try:
                return fused[(column, method)]
            except (KeyError, TypeError):
                pass
        return _call(grouped[column], method)

    return read


def _wanted_methods(ir: dict, inputs: list[str | None], sources: dict[str, str]):
    """`{column: [method, ...]}` — everything one fused `agg` has to compute for this node.

    Deduplicated per column, because `agg({c: ["sum", "sum"]})` is not a request for one
    reduction twice — it is a frame with a duplicated column, which the reader then cannot
    address unambiguously. Two aliases over the same `sum(x)` are one computation.

    Returns `{}` when any reduction is outside `_METHODS` (a `quantile`, which takes a
    parameter no method name carries), so the node falls back to the unfused path whole rather
    than half-fusing and grouping twice.
    """
    want: dict[str, list[str]] = {}
    for spec, column in zip(ir["aggregates"], inputs, strict=True):
        func = spec["func"]
        if func == "count_star":
            continue
        methods = _METHODS.get(func)
        if methods is None or column is None:
            return {}
        for method in methods:
            if method not in want.setdefault(column, []):
                want[column].append(method)
    # The label of each normalized group key is another full pass otherwise, and it is needed on
    # exactly the queries a float key makes most expensive.
    for source in sources.values():
        if "first" not in want.setdefault(source, []):
            want[source].append("first")
    return want


def _reduce(grouped, spec: dict, column: str | None, be: DfBackend, read=None):
    """One reduction over the shared `GroupBy`, as a Series indexed by the group key.

    `read` sources the raw per-group series for a `(column, method)` pair — from the fused
    `agg` pass when there was one, and from the `GroupBy` directly otherwise. The semantics
    below (the null-on-empty rule, the retypings) are the same either way, which is the point
    of routing both through here.
    """
    func = spec["func"]
    if func == "count_star":
        return _as_int64(grouped.size(), be)
    read = read or _series_reader(grouped, None)
    if func in _PLAIN:
        reduced = read(column, _PLAIN[func])
        return _as_int64(reduced, be) if func in _COUNTING else reduced
    if func == "product":
        # The engine (and DuckDB) answer `product` in **double** whatever the input's type,
        # because the running product of a bigint column leaves its range almost immediately.
        # Both libraries' `prod` keeps the integer instead, so an int64 column's product came
        # back int64 — the right number until it overflows, and a column a CPU-recovered shard's
        # double cannot be concatenated with either way.
        import pyarrow as pa

        reduced = _null_if_empty(read(column, "count"), read(column, "prod"))
        return reduced.astype(be.dtype(pa.float64()))
    if func in _MIN_COUNT:
        return _null_if_empty(read(column, "count"), read(column, _MIN_COUNT[func]))
    if func in _BOOL_FOLD:
        return _null_if_empty(read(column, "count"), read(column, _BOOL_FOLD[func]))
    if func in _SAMPLE_MOMENT:
        return read(column, _SAMPLE_MOMENT[func])
    if func == "quantile":
        # The one reduction the fused pass cannot carry: `agg` takes method *names*, and the
        # quantile to take is a parameter. It is why `_wanted_methods` declines the whole node
        # rather than half-fusing it, and why this is the only branch that still reaches for
        # the grouped column itself.
        return _call(grouped[column], "quantile", float(spec["param"]))
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
    # Every reduction, and every group label, computed in **one** pass where the backend takes
    # the fused form. Issuing them one at a time is a full hash group-by per reduction: TPC-H
    # q1's eight reductions and two labels grouped a 10 M-row shard ten times, which measured
    # 0.25 s of a 0.40 s shard on a T4 against 0.12 s to read that shard off storage.
    fused = _fused(grouped, _wanted_methods(ir, inputs, sources))
    read = _series_reader(grouped, fused)
    columns = {}
    for spec, column in zip(ir["aggregates"], inputs, strict=True):
        columns[spec["alias"]] = _reduce(grouped, spec, column, be, read)
    # The un-normalized value each normalized group is labelled by. `sort=False` keeps the
    # groups in first-seen order, and `first()` picks each group's first row, so the label is
    # the one the engine's own first-seen representative would be.
    labels = {f"__bt_lbl{i}": read(src, "first") for i, src in enumerate(sources.values())}
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
#: Imported, not restated: Kyber's statistics need the same set and cannot import this module.
_COUNTING = COUNTING_AGGS


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
