"""List and vector expressions, built from the two primitives both dataframe libraries have.

Neither backend exposes a list *reduction*. cuDF has some on its list accessor and pandas has
none at all, so translating those directly would ship a path only the device can run and only
the device could ever be wrong about — which is why this family used to be declined whole, and
why `.list.sum()`, `.list.dot()` and every vector distance sent an otherwise perfect chain to
the host. On a GPU cluster that is the wrong way round: an embedding column is the single most
common thing anyone has a device for.

The construction here is `explode` + `groupby`, which both libraries implement as one kernel
each and which the `unnest` operator already relies on. One row per element, grouped by the row
it came from, reduced, and put back — every step on the device, and identical arithmetic on
both backends because it is the same two calls.

Three rules the engine follows and the naive form does not:

* a **null element is skipped**, and a list of nothing but nulls reduces to null, exactly as a
  column of nulls does. Grouping gives that for free; the identity element a library returns
  for an empty group does not, so the count is checked;
* an **empty list reduces to null** — but an empty *pair* of lists has a dot product of `0.0`,
  because a sum over no terms is zero where a measurement over no values is unknown. The two
  families are separated here rather than reconciled;
* a **pairwise operation masks by both sides**. `cosine_similarity([1, null], [2, 3])` is `1.0`
  in the engine, which only holds if the second position is dropped from *both* norms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.core.gpu_plan.backend import Unsupported, call_or_decline

if TYPE_CHECKING:
    from batcher.core.gpu_plan.backend import DfBackend

__all__ = [
    "LIST_BINARY_FNS",
    "LIST_REDUCTIONS",
    "eval_list_binary",
    "eval_list_contains",
    "eval_list_fn",
    "eval_list_get",
    "eval_list_position",
    "supported_list_binary",
    "supported_list_fn",
]

#: Private columns the element view carries: the value being reduced, the position of the row
#: it came from, and the right-hand value when two lists are walked together.
_VAL = "__bt_lval"
_ROW = "__bt_lrow"
_RHS = "__bt_lrhs"
_LEN = "__bt_llen"
_EXT = "__bt_lext"

#: Reductions that are one grouped call on the exploded elements, as `(method, retype)`.
#: `retype` is `None` where the engine keeps the element's own type — `sum`, `min` and `max`
#: of a bigint list are bigints — and the Arrow type name where it does not. `product` is a
#: double in the engine whatever it reduces, for the same reason the grouped `product` is.
LIST_REDUCTIONS = {
    "sum": ("sum", None),
    "min": ("min", None),
    "max": ("max", None),
    "mean": ("mean", "float64"),
    "median": ("median", "float64"),
    "std": ("std", "float64"),
    "var": ("var", "float64"),
    "product": ("prod", "float64"),
}

#: Reductions over a *transformed* element, as `(elementwise transform, method, retype)`. Each
#: is the norm's own definition rather than an approximation of it.
_TRANSFORMED = {
    "l1_norm": ("abs", "sum", "float64"),
    "max_abs": ("abs", "max", "float64"),
    "l2_norm": ("square", "sum", "float64"),
}

#: Pairwise reductions over two lists walked together, as the term each sums.
LIST_BINARY_FNS = frozenset({"dot", "cosine_similarity", "l1_distance", "l2_distance", "hamming"})

#: The list functions this module translates. `n_unique` is separate from `LIST_REDUCTIONS`
#: because its answer over an *empty* list is `0` rather than null: it counts, and a count over
#: nothing is zero, where a measurement over nothing is unknown.
#: The two positional reductions: the index of the extreme element rather than its value.
_ARG_FNS = {"arg_max": "max", "arg_min": "min"}

_SCALAR_FNS = (
    frozenset(LIST_REDUCTIONS) | frozenset(_TRANSFORMED) | frozenset(_ARG_FNS) | {"len", "n_unique"}
)


def supported_list_fn(fn: str) -> bool:
    """Whether a one-list function is translatable.

    Args:
        fn: The `list` node's ``fn`` discriminator.

    Returns:
        True when this module evaluates it.
    """
    return fn in _SCALAR_FNS


def supported_list_binary(fn: str) -> bool:
    """Whether a two-list function is translatable.

    Args:
        fn: The `list_binary` node's ``fn`` discriminator.

    Returns:
        True when this module evaluates it.
    """
    return fn in LIST_BINARY_FNS


def _arrow(name: str):
    import pyarrow as pa

    return getattr(pa, name)()


def _positions(be: DfBackend, n: int):
    """`0..n-1` as a column, materialized in the library's own layer rather than in Python."""
    return be.series(range(n)).astype(be.dtype(_arrow("int64")))


def _elements(x, be: DfBackend):
    """One row per element of each list in `x`, tagged with the position of its source row.

    The frame is built on a fresh index so the row tag is a *position* rather than whatever
    index the column arrived with — the caller's frame may have been filtered, and a label-based
    tag would then disagree with the order the result has to be put back in.
    """
    frame = be.lib.DataFrame({_VAL: x.reset_index(drop=True)})
    frame[_ROW] = _positions(be, len(frame))
    return frame.explode(_VAL)


def _grouped(elements, column: str = _VAL):
    """The exploded elements grouped by their source row, in row order.

    `sort=True` is what makes the result directly assignable back: every row contributes at
    least one exploded row (an empty or null list contributes one carrying null), so the groups
    are exactly `0..n-1` in order.
    """
    return elements.groupby(_ROW, sort=True)[column]


def _restore(reduced, x, be: DfBackend, *, retype: str | None, empty_is_null: bool = True):
    """A per-row reduction back on `x`'s own index, with the engine's empty-list rule applied.

    Args:
        reduced: The grouped reduction, indexed by source-row position.
        x: The list column it was taken over.
        be: The dataframe backend.
        retype: An Arrow type name to cast to, or `None` to keep what the reduction produced.
        empty_is_null: Whether a list with no non-null element reduces to null (a measurement)
            rather than to the reduction's own answer over nothing (a count).

    Returns:
        The reduction as a column aligned to `x`.
    """
    if retype is not None:
        reduced = reduced.astype(be.dtype(_arrow(retype)))
    if empty_is_null:
        reduced = reduced.where(reduced.notna(), None)
    # A null *list* is null whatever the elements did, and is not the same as an empty one: an
    # empty list has a length and a distinct count, a missing list has neither.
    missing = x.isna().reset_index(drop=True)
    reduced = reduced.where(~missing, None)
    reduced.index = x.index
    return reduced


def eval_list_fn(fn: str, x, be: DfBackend):
    """Evaluate a one-list function over the list column `x`.

    Args:
        fn: The `list` node's ``fn`` discriminator.
        x: The list column.
        be: The dataframe backend to compute on.

    Returns:
        The reduction as a column of `x`'s length.

    Raises:
        Unsupported: For a function outside the translated subset.
    """
    if fn == "len":
        return call_or_decline(x.list, "len").astype(be.dtype(_arrow("int64")))
    elements = _elements(x, be)
    if fn == "n_unique":
        # A count, not a measurement: an empty list has zero distinct elements, where a `sum`
        # over an empty list is unknown. Only a *missing* list is null.
        counted = call_or_decline(_grouped(elements), "nunique")
        return _restore(counted, x, be, retype="int64", empty_is_null=False)
    if fn in _ARG_FNS:
        return _arg_extreme(fn, x, elements, be)
    if fn in _TRANSFORMED:
        transform, method, retype = _TRANSFORMED[fn]
        elements = elements.copy(deep=False)
        elements[_VAL] = _elementwise(elements[_VAL], transform)
        reduced = call_or_decline(_grouped(elements), method)
        if fn == "l2_norm":
            reduced = _sqrt(reduced, be)
        # `_present` rather than the reduction's own nullness: a sum of squares over an empty
        # group is `0`, which is a real number and would survive the null check.
        return _restore(_blank_empty(reduced, elements), x, be, retype=retype)
    if fn not in LIST_REDUCTIONS:
        raise Unsupported(f"list fn {fn}")
    method, retype = LIST_REDUCTIONS[fn]
    reduced = call_or_decline(_grouped(elements), method)
    return _restore(_blank_empty(reduced, elements), x, be, retype=retype)


def _arg_extreme(fn: str, x, elements, be: DfBackend):
    """`arg_max`/`arg_min` — the **0-based position** of the extreme element, else null.

    Found by marking the positions whose value equals the group's extreme and taking the
    smallest of them, rather than by an index lookup: `idxmax` says it in one call, and it
    reports an index *label*, which the two libraries do not agree about after an explode. The
    marking form also settles ties the way the engine does — the first occurrence wins.

    Declined over a `NaN`, for the reason the grouped order statistics are: the engine orders
    `NaN` above every number and both libraries treat it as missing, so the extreme itself
    would be a different element.
    """
    values = elements[_VAL]
    if bool((values != values).fillna(False).any()):
        raise Unsupported(f"list {fn} over a NaN-bearing list")
    extreme = call_or_decline(_grouped(elements), _ARG_FNS[fn])
    frame = elements.copy(deep=False)
    frame[_POS] = call_or_decline(_grouped(elements), "cumcount")
    marks = be.lib.DataFrame({_ROW: _positions(be, len(x)), _EXT: extreme.reset_index(drop=True)})
    frame = frame.merge(marks, on=_ROW, how="left")
    frame[_POS] = frame[_POS].where((frame[_VAL] == frame[_EXT]).fillna(False), None)
    found = frame.groupby(_ROW, sort=True)[_POS].min()
    return _restore(found, x, be, retype="int64", empty_is_null=False)


def _blank_empty(reduced, elements):
    """`reduced`, nulled for every row whose list held no non-null element.

    The libraries answer such a group with the reduction's identity — `0` for a sum, `1` for a
    product — which reads as a measurement. The engine returns null, exactly as it does for a
    column of nulls.
    """
    return reduced.where(_grouped(elements).count() > 0, None)


def _elementwise(values, transform: str):
    """One elementwise transform, spelled with operators both libraries implement on Arrow."""
    if transform == "abs":
        return values.abs()
    return values * values  # `square`, as a multiplication rather than a ufunc


def _sqrt(values, be: DfBackend):
    from batcher.core.gpu_plan.scalar_fns import apply_ufunc

    return apply_ufunc("sqrt", values.astype(be.dtype(_arrow("float64"))), be)


def _paired(left, right, be: DfBackend):
    """The two lists' elements walked together, one row per aligned pair.

    Positional alignment after two independent explodes is exact **only** when every row's two
    lists are the same length, which the engine requires as well — it raises on a mismatch. So
    the lengths are checked first and a mismatch is declined, which sends the chain to the CPU
    engine, where the same query raises the engine's own error rather than a translated one.
    """
    # Compared on *filled* lengths rather than by OR-ing in a both-null case. A missing list has
    # no length, so the naive comparison is null there, and reconciling that with `|` assumes
    # Kleene logic — which cuDF's `|` does not implement: `null | true` came back null on the
    # device, the fill turned it into `false`, and every vector distance in the fleet declined
    # while the host backend accepted them. `-1` is not a length any list has.
    llen = call_or_decline(left.list, "len").fillna(-1)
    rlen = call_or_decline(right.list, "len").fillna(-1)
    if not bool((llen == rlen).fillna(False).all()):
        raise Unsupported("list_binary over lists of unequal length")
    frame = be.lib.DataFrame(
        {_VAL: left.reset_index(drop=True), _RHS: right.reset_index(drop=True)}
    )
    frame[_ROW] = _positions(be, len(frame))
    # Exploded one column at a time, because cuDF's `explode` takes a single column. The two
    # results line up row for row precisely because the lengths were just proved equal.
    lhs = frame[[_ROW, _VAL]].explode(_VAL).reset_index(drop=True)
    rhs = frame[[_ROW, _RHS]].explode(_RHS).reset_index(drop=True)
    lhs[_RHS] = rhs[_RHS]
    return lhs


def eval_list_binary(fn: str, left, right, be: DfBackend):
    """Evaluate a pairwise function over two list columns.

    Args:
        fn: The `list_binary` node's ``fn`` discriminator.
        left: The left list column.
        right: The right list column.
        be: The dataframe backend to compute on.

    Returns:
        The reduction as a column of `left`'s length.

    Raises:
        Unsupported: For a function outside the translated subset, or for lists that do not
            align.
    """
    if fn not in LIST_BINARY_FNS:
        raise Unsupported(f"list_binary fn {fn}")
    pairs = _paired(left, right, be)
    lv, rv = pairs[_VAL], pairs[_RHS]
    # A pair counts only where *both* sides have a value. That is the engine's rule and it is
    # not the obvious one: `cosine_similarity([1, null], [2, 3])` is `1.0`, which holds only if
    # the second position leaves both norms as well as the dot product.
    both = (lv.notna() & rv.notna()).fillna(False)
    if fn == "cosine_similarity":
        return _cosine(pairs, lv, rv, both, left, be)
    term = _pair_term(fn, lv, rv, be)
    pairs = pairs.copy(deep=False)
    pairs[_VAL] = term.where(both, None)
    reduced = _summed(pairs)
    if fn == "l2_distance":
        reduced = _sqrt(reduced, be)
    # A sum over no terms is `0.0` here, unlike a reduction over one list: the engine's dot
    # product of two empty lists is zero, not unknown. Only a missing list is null.
    return _restore(reduced, left, be, retype="float64", empty_is_null=False)


def _summed(pairs):
    """The pairwise terms summed per row, with a sum over **no terms** pinned to zero.

    The two libraries disagree about that sum: pandas returns the additive identity and cuDF
    returns null. The engine returns zero — a dot product of two empty vectors is `0.0`, not
    unknown — so it is stated here rather than inherited, and every vector distance came back
    null from a GPU for an empty or all-null pair while returning zero in CI.
    """
    return call_or_decline(_grouped(pairs), "sum").fillna(0.0)


def _pair_term(fn: str, lv, rv, be: DfBackend):
    """The term each pairwise reduction sums, as its own definition."""
    if fn == "dot":
        return lv * rv
    if fn == "l1_distance":
        return (lv - rv).abs()
    if fn == "l2_distance":
        difference = lv - rv
        return difference * difference
    # `hamming` counts the positions that differ, which the engine reports as a double.
    return (lv != rv).astype(be.dtype(_arrow("float64")))


def _cosine(pairs, lv, rv, both, left, be: DfBackend):
    """`dot / (|left| * |right|)` over the aligned, both-present positions.

    Null where either norm is zero, which covers both the zero vector and the pair that had no
    aligned position at all — a similarity needs a direction, and neither has one.
    """
    frame = pairs.copy(deep=False)
    frame[_VAL] = (lv * rv).where(both, None)
    dot = _summed(frame)
    frame[_VAL] = (lv * lv).where(both, None)
    left_norm = _sqrt(_summed(frame), be)
    frame[_VAL] = (rv * rv).where(both, None)
    right_norm = _sqrt(_summed(frame), be)
    scale = left_norm * right_norm
    similarity = (dot / scale).where((scale != 0).fillna(False), None)
    return _restore(similarity, left, be, retype="float64", empty_is_null=False)


#: The position of an exploded element within its own list, from the front and from the back.
_POS = "__bt_lpos"


def eval_list_get(x, index: int, be: DfBackend):
    """`list[index]` — 0-based, negative counting from the end, null when out of range.

    Selected by *filtering* the exploded elements to the wanted position rather than by masking
    their values, because the two differ on exactly one case: a list whose element at that
    position **is** null. Masking and reducing would skip it and hand back a later element;
    filtering keeps the row, and the row's value is null, which is the engine's answer.

    A row whose list is too short contributes no matching element at all, so the left join
    below leaves it null — which is again the engine's answer, where both libraries' own
    accessors raise instead.
    """
    elements = _elements(x, be)
    frame = elements.copy(deep=False)
    frame[_POS] = call_or_decline(_grouped(elements), "cumcount")
    if index >= 0:
        wanted = frame[_POS] == index
    else:
        # Counting from the end needs the row's own length on every one of its exploded rows.
        # `cumcount(ascending=False)` would say it in one call and **cuDF does not implement
        # that keyword** — it raises, on the device only, where the host backend accepts it. So
        # the lengths are joined on instead, which is two calls both libraries certainly have.
        lengths = be.lib.DataFrame({_ROW: _positions(be, len(x)), _LEN: _list_lengths(x)})
        frame = frame.merge(lengths, on=_ROW, how="left")
        wanted = frame[_POS] == frame[_LEN] + index
    picked = frame[wanted.fillna(False)]
    return _rejoin(picked[[_ROW, _VAL]], x, be)


def _list_lengths(x):
    """Each list's element count, as a column on a fresh index."""
    return call_or_decline(x.list, "len").reset_index(drop=True)


def _rejoin(picked, x, be: DfBackend):
    """`picked`'s one value per source row, back on `x`'s index, null where it selected none."""
    base = be.lib.DataFrame({_ROW: _positions(be, len(x))})
    merged = base.merge(picked, on=_ROW, how="left").sort_values(_ROW)
    out = merged[_VAL].reset_index(drop=True)
    out.index = x.index
    return out


def eval_list_contains(x, value, be: DfBackend):
    """`list_contains(list, value)` — true when any element equals `value`, false when none does.

    Folded through an integer maximum rather than a boolean one: `max` over booleans is the
    only reduction of the three backends' `GroupBy` surfaces that is not offered identically,
    and `0`/`1` is the same fold through arithmetic both certainly have.
    """
    elements = _elements(x, be)
    frame = elements.copy(deep=False)
    frame[_VAL] = (elements[_VAL] == value).fillna(False).astype(be.dtype(_arrow("int64")))
    found = call_or_decline(_grouped(frame), "max") > 0
    found = found.astype(be.dtype(_arrow("bool_")))
    # An empty list contains nothing, which is `false` and not unknown; only a missing list is
    # null.
    return _restore(found, x, be, retype=None, empty_is_null=False)


def eval_list_position(x, value, be: DfBackend):
    """`list_position(list, value)` — the **1-based** position of the first match, else null.

    One-based and null-for-absent is SQL's convention and the engine's; a 0-based `index` with
    `-1` for absent, which is what a dataframe library's own accessor returns, would be wrong
    in both places at once.
    """
    elements = _elements(x, be)
    frame = elements.copy(deep=False)
    positions = call_or_decline(_grouped(elements), "cumcount") + 1
    frame[_POS] = positions.where((elements[_VAL] == value).fillna(False), None)
    first = call_or_decline(_grouped(frame, _POS), "min")
    return _restore(first, x, be, retype="int64", empty_is_null=False)
