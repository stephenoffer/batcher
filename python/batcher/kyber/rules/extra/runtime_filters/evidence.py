"""The proofs the runtime-filter rules stand on — and nothing else may stand on.

Every rule in this family deletes rows, so each one needs a *proof* that the rows it deletes
could not have appeared in the answer. Those proofs live here, once, so the rules cannot
disagree about them (a second, drifted copy of "is this bloom probe sound?" is exactly how a
family like this starts returning wrong answers).

Four kinds of evidence:

* **Null-key evidence** (`_may_hold_null`) — an equi-join never matches a NULL key
  (`bc_runtime::join`: "a row with any null key never matches — NULL ≠ NULL").
* **Membership evidence** (`_value_set`) — an `IN`/`=` conjunct is null-rejecting under 3VL, so
  a surviving row's column value is *exactly* one of the literals. The set is therefore a sound
  **upper bound** on the column's values, never an under-approximation.
* **Bloom evidence** (`_bloom_refutes`) — a bloom has no false negatives, so `contains() ->
  False` is a proof of absence, and absence survives row-shrinking. It is domain-guarded:
  probing an Int64 index with the string `"5"` reports a definitive absence for a value that
  *is* present, and that result deletes rows.
* **Bound evidence** (`_out_of_range`) — min/max are valid *bounds* at any provenance (a
  row-shrinking operator can only narrow the true range), so a value outside them is provably
  absent. Only an *always-true* claim would need `Provenance.EXACT`; this is always-false.

Importing `rules.joins` here is deliberate as well as necessary: it owns `_FILTERABLE_SIDES`
(the law for which side each join type may reduce) and `runtime_join_filter`, and importing it
before this package's rule decorators run is what puts that rule *first* in the ENFORCE phase —
so `sink_runtime_filter_to_source` sees the filter it inserts.
"""

from __future__ import annotations

import dataclasses
import math

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.rule import Phase, RuleCategory
from batcher.kyber.rules.joins import _FILTERABLE_SIDES
from batcher.kyber.rules.zonemap_pruning import _predicate_status, _same_bloom_domain
from batcher.plan.bloom_index import BloomIndex
from batcher.plan.expr_ir import Binary, Col, Expr, InList, Lit, referenced_columns, remap_columns
from batcher.plan.expr_rewrite import (
    combine_conjuncts,
    combine_disjuncts,
    expr_key,
    split_conjuncts,
    split_disjuncts,
)
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    Project,
    Sample,
    Scan,
    Sort,
    Union,
    is_cartesian_key_pair,
)
from batcher.plan.stats import ColumnStat, Provenance, RelStats, ambiguous_float_bound

__all__ = ["FILTERABLE_SIDES", "SIP"]

#: Re-exported so the rules read from the one table `rules.joins` owns, never a restatement of
#: it: `inner`/`semi` → both sides reducible, `anti`/`left` → right only, `right` → left only,
#: `full` → nothing. Get this wrong and rows are deleted, not merely mis-costed.
FILTERABLE_SIDES = _FILTERABLE_SIDES

#: The phase every rule in this family registers under. **PUSHDOWN, not ENFORCE** — and that is a
#: correctness requirement, not a preference.
#:
#: A runtime filter changes the join's *subtree*, and `kyber.signature.plan_signature` is
#: structural, so inserting one changes the signature of the join and of every operator above it.
#: The learning loop is keyed on exactly that signature: SELECTION reads the learned join-strategy
#: arm and cardinality correction for `plan_signature(join)` at SELECTION time, while
#: `annotate_ops` stamps the *final* plan's signature for Core to report its measurements against.
#: Insert a filter after SELECTION and those two keys differ — so what Core measures is filed under
#: a name Kyber will never look up again, and the Core-measures/Kyber-decides loop silently stops
#: closing for every join it fires on. (`rules.joins.runtime_join_filter` has the same flaw, but it
#: fires only when a key range genuinely narrows; these rules fire on almost every join, which
#: would turn a rare latent bug into a universal one.)
#:
#: Running in PUSHDOWN puts every rewrite *before* the first keyed decision, so the plan Kyber
#: costs is the plan Core measures. It also means the cost model can see the runtime filters when
#: it picks a join order and a build side — strictly better information. The price is that PUSHDOWN
#: iterates to a fixpoint, so every rule here MUST be idempotent: the inserting rules guard on
#: `_lacks` (which traces the whole spine, not just the adjacent filter chain — see `_conjuncts_on`)
#: and the pruning rules are monotone (they only ever remove a conjunct, a disjunct, or a member).
SIP = {"phase": Phase.PUSHDOWN, "category": RuleCategory.REWRITE}

# Operators a column passes through unchanged in name and value — they only shrink or reorder
# rows, so a constraint proven below one still holds above it.
_TRANSPARENT = (Sort, Limit, Sample, Distinct)

# Joins whose result is *empty* when no key pair can be equal. An outer/anti join keeps its
# preserved side's unmatched rows, so "nothing matches" is not emptiness for it — that case is
# `no_match_join_to_preserved_side`'s.
_MATCH_REQUIRED = ("inner", "semi")

# "This expression pins the column to no literal at all" — distinct from pinning it to `None`,
# which `_eq_value` also refuses (`col = NULL` is never TRUE, so it is not a member).
_MISSING = object()


# --- filter-chain plumbing ---------------------------------------------------


def _filter_chain(side: LogicalPlan) -> list[Filter]:
    """The `Filter`s stacked directly on `side`, outermost first (possibly empty)."""
    out: list[Filter] = []
    cur = side
    while isinstance(cur, Filter):
        out.append(cur)
        cur = cur.input
    return out


def _add_conjuncts(side: LogicalPlan, preds: list[tuple[str, Expr]]) -> LogicalPlan:
    """`side` under the `(column, predicate)` pairs it does not already carry.

    Unchanged (the *same object*, which is how the driver detects its fixpoint in O(1)) when every
    predicate is already enforced somewhere on the spine — see `_lacks` — or when the side is
    already provably empty.

    Skipping an empty side is not just an optimization. Filtering a relation that is already known
    to hold no rows buys nothing, and it actively harms: the added `Filter` sits *above* the
    `Limit(_, 0)` marker, which breaks the `Filter`-to-`Scan` adjacency that
    `infer_join_predicates` relies on to recognise a constraint it has already mirrored. It then
    mirrors it again, this rule filters again, and the plan grows a layer per fixpoint iteration.
    """
    if _already_empty(side):
        return side
    fresh = [pred for col, pred in preds if _lacks(side, col, pred)]
    return Filter(side, combine_conjuncts(fresh)) if fresh else side


def _rebuild(
    node: Join, left: LogicalPlan, right: LogicalPlan, ctx: OptimizerContext, note: str
) -> LogicalPlan | None:
    """The join over its (possibly filtered) inputs, or None when neither side changed."""
    if left is node.left and right is node.right:
        return None
    ctx.notes.setdefault("runtime_join_filters", []).append(f"{node.join_type}:{note}")
    return dataclasses.replace(node, left=left, right=right)


# --- key-shape evidence ------------------------------------------------------


def _real_key_pairs(node: Join) -> list[tuple[str, str]]:
    """The join's genuine equi-key pairs — the constant `__cross_key` pseudo-edge excluded.

    A cartesian pseudo-key is the same non-null constant on both sides: it matches
    unconditionally, so it carries no sideways information and a filter on it is pure cost.
    """
    return [
        (lk, rk)
        for lk, rk in zip(node.left_keys, node.right_keys, strict=True)
        if not is_cartesian_key_pair(node.left, lk, node.right, rk)
    ]


def _no_match_candidate(node: Join) -> bool:
    """Whether a "no key pair can be equal" proof would empty this join, and has not yet fired."""
    if node.join_type not in _MATCH_REQUIRED or not node.left_keys:
        return False
    return not _already_empty(node.left)


def _already_empty(plan: LogicalPlan) -> bool:
    """Whether an empty marker sits anywhere on `plan`'s unary spine — the idempotence guard.

    Checking only the *immediate* child is not enough, and the difference is a non-terminating
    fixpoint rather than a missed rewrite: once a rule marks a join's left input empty, PUSHDOWN
    keeps running, and the next rule to sink a predicate into that side wraps the marker in a
    `Filter`. The marker is then no longer the direct child, so a rule that looked only there
    would fire again, add a second marker, and grow the plan on every iteration until the phase
    hits its cap. Walking the spine sees the marker wherever the other rules have parked it.
    """
    while isinstance(plan, (Filter, Sort, Sample, Distinct, Project, Limit)):
        if isinstance(plan, Limit) and plan.n == 0:
            return True
        plan = plan.input
    return False


def _empty_marker(node: Join) -> Join:
    """The join with its left input marked empty — the canonical zero-row form.

    Mirrors `join_disjoint_keys_to_empty`: the IR cannot express a zero-row relation with a
    join's two-sided output schema, so the *input* is emptied instead and the estimator reports
    the join EXACT-empty from there.
    """
    return dataclasses.replace(node, left=Limit(node.left, 0))


# --- null evidence -----------------------------------------------------------


def _may_hold_null(stat: ColumnStat) -> bool:
    """Whether a null is still possible in the column — i.e. an `IS NOT NULL` would prune.

    A *known-zero* null count (at any provenance — a filter sets it to unknown rather than
    claiming zero) means the filter could never drop a row, so adding it is pure per-row cost.

    Ask this **where the predicate will land**, not where the rule stands — see
    `_provably_true_at_source`, which is the guard that keeps this evidence confluent.
    """
    return stat.null_count != 0


def _provably_true_at_source(
    side: LogicalPlan, col: str, pred: Expr, ctx: OptimizerContext
) -> bool:
    """Whether `pred` is *already always true* at the scan `side` is rooted in.

    `_may_hold_null` is read at the join, above the side's filters — and a `Filter` sets
    `null_count` to *unknown* (`stats/columns.py`). So a column the scan proves is null-free
    reads as "may hold null" up there, and `push_is_not_null_from_join_key` adds an
    `IS NOT NULL` that is a tautology. Pushdown then sinks that filter to the scan, where
    `drop_filter_conjunct_implied_by_zonemap` reads the *scan's* stats, proves the very same
    predicate always true, and deletes it — whereupon the adder, finding nothing on the spine,
    adds it back. The two rules ping-pong and PUSHDOWN never reaches a fixpoint: measured at 16
    iterations on TPC-H q3, 24 on q5, 25 on q7, every one of them re-walking the whole plan and
    every expression in it.

    The disagreement is *positional*, not logical: both rules consult `_predicate_status`, just
    at different depths. Asking it at the same place — the scan the predicate would sink to — is
    what makes the pair confluent. Column renames are followed down through projections, so the
    question is asked about the column the scan actually holds.

    Declining to add is always semantically safe: an equi-join drops null keys itself, so this
    predicate is a pure optimization and never load-bearing.
    """
    node: LogicalPlan = side
    name = col
    while True:
        if isinstance(node, Filter | Limit):
            node = node.input
        elif isinstance(node, Project):
            src = next(
                (i.expr.name for i in node.items if i.alias == name and isinstance(i.expr, Col)),
                None,
            )
            if src is None:  # computed, or not produced here — not provably the same column
                return False
            name = src
            node = node.input
        else:
            break
    if not isinstance(node, Scan):
        return False
    return _predicate_status(remap_columns(pred, {col: name}), ctx.estimator.estimate(node)) is True


def _all_null_key(side: LogicalPlan, keys: tuple[str, ...], ctx: OptimizerContext) -> bool:
    """Whether some key of `side` is *proven* to hold nothing but nulls.

    EXACT-gated on both counts: an *estimated* all-null column that holds one real value would
    delete the entire answer.
    """
    stats = ctx.estimator.estimate(side)
    if not stats.rows_exact or stats.rows <= 0:
        return False
    for key in keys:
        col = stats.column(key)
        if col.provenance is not Provenance.EXACT or col.null_count is None:
            continue
        if col.null_count >= stats.rows:
            return True
    return False


# --- membership evidence -----------------------------------------------------


def _eq_value(expr: Expr, col: str) -> object:
    """The literal a bare `col = lit` pins `col` to, else `_MISSING`.

    `NULL`/`NaN` literals are refused: `col = NULL` and `col = NaN` are never TRUE, so they pin
    the column to *nothing* — treating them as a member would invent a value the column cannot
    hold and make a "disjoint" or "refuted" conclusion out of a predicate that keeps no rows.
    """
    if not (isinstance(expr, Binary) and expr.op == "eq"):
        return _MISSING
    for side, other in ((expr.left, expr.right), (expr.right, expr.left)):
        if isinstance(side, Col) and side.name == col and isinstance(other, Lit):
            value = other.value
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return _MISSING
            return value
    return _MISSING


def _membership_values(conj: Expr, col: str) -> tuple | None:
    """The literal values `conj` pins `col` to, in order, else None.

    Three spellings of the same constraint, all null-rejecting under 3VL: the folded
    `col IN (…)` node, a bare `col = v`, and the **OR-chain of equalities** `is_in` actually
    lowers to (`algebraic._fold_disjunction` only folds a chain of ≥ 5 into an `InList`, so the
    chain is the common form and a rule that matched only `InList` would almost never fire).
    """
    if isinstance(conj, InList) and isinstance(conj.input, Col) and conj.input.name == col:
        return conj.values
    values: list = []
    for disjunct in split_disjuncts(conj):
        value = _eq_value(disjunct, col)
        if value is _MISSING:
            return None
        values.append(value)
    return tuple(values) if values else None


def _rebuild_membership(original: Expr, col: str, survivors: tuple) -> Expr:
    """`original` narrowed to `survivors`, in the spelling it arrived in."""
    if isinstance(original, InList):
        return InList(Col(col), survivors)
    return combine_disjuncts([Binary("eq", Col(col), Lit(v)) for v in survivors])


def _conjuncts_on(side: LogicalPlan, col: str) -> list[Expr]:
    """Conjuncts constraining *only* `col`, traced down through the operators that carry it.

    A `Filter` contributes its single-column conjuncts (true of every row it emits); the
    row-shrinking operators pass them through; a `Project` that merely *renames* the column
    follows it to its source name. Anything else (a computed projection, a join, an aggregate, a
    union) stops the trace — the column above is then not provably the column below.
    """
    if isinstance(side, Filter):
        here = [c for c in split_conjuncts(side.predicate) if referenced_columns(c) == {col}]
        return here + _conjuncts_on(side.input, col)
    if isinstance(side, _TRANSPARENT):
        return _conjuncts_on(side.input, col)
    if isinstance(side, Project):
        for item in side.items:
            if item.alias == col and isinstance(item.expr, Col):
                src = item.expr.name
                return [remap_columns(c, {src: col}) for c in _conjuncts_on(side.input, src)]
        return []
    if isinstance(side, Aggregate):
        # A bare-`Col` group key takes its values from the input column, so a constraint proven
        # on that column holds for every group the aggregate emits. (The same implication
        # `push_filter_through_aggregate` relies on, read in the other direction.) Tracing this
        # is what keeps the inserting rules idempotent under PUSHDOWN's fixpoint: that rule sinks
        # a freshly-added filter below the aggregate, and a guard that could not see through one
        # would add it again on the next iteration, forever.
        for key in side.group_keys:
            if key.alias == col and isinstance(key.expr, Col):
                src = key.expr.name
                return [remap_columns(c, {src: col}) for c in _conjuncts_on(side.input, src)]
        return []
    if isinstance(side, Union):
        # A constraint holds on the union only if it holds on *every* branch — the intersection.
        # (`push_filter_into_union` copies a filter into each branch, so this is the guard that
        # recognises its own work.)
        per_branch = [
            {expr_key(c): c for c in _conjuncts_on(branch, col)} for branch in side.inputs
        ]
        if not per_branch:
            return []
        shared = set(per_branch[0])
        for keys in per_branch[1:]:
            shared &= set(keys)
        return [per_branch[0][k] for k in sorted(shared)]
    return []


def _lacks(side: LogicalPlan, col: str, pred: Expr) -> bool:
    """Whether `pred` is *not* already enforced anywhere on `side`'s spine — the idempotence guard.

    Checking only the filters directly on `side` is not enough under a fixpoint phase: the moment
    a rule inserts `Filter(side, pred)`, the shipped pushdown rewrites sink it below the
    projection / aggregate / union above the scan, and a guard that could not see it there would
    insert it again on the next iteration, and the next — the plan growing until the phase hits its
    iteration cap. `_conjuncts_on` follows the column down that whole spine, through renames, so
    the guard recognises the predicate wherever pushdown has parked it.
    """
    return expr_key(pred) not in {expr_key(c) for c in _conjuncts_on(side, col)}


def _value_set(side: LogicalPlan, col: str) -> frozenset | None:
    """The finite set of values `col` can hold on `side`'s output, or None if unbounded.

    The intersection of every membership constraint found on the way down — a sound *upper*
    bound on the column's actual values, so a row outside it cannot exist.
    """
    best: set | None = None
    for conj in _conjuncts_on(side, col):
        values = _membership_values(conj, col)
        if values is None:
            continue
        try:
            found = set(values)
        except TypeError:
            continue  # an unhashable literal — not a value set we can reason about
        best = found if best is None else (best & found)
    return None if best is None else frozenset(best)


def _candidate_key_values(side: LogicalPlan, key: str, stats: RelStats) -> frozenset | None:
    """Every value `key` can take on `side` — from its membership constraints or an EXACT
    single-valued range — or None when the key is unbounded."""
    values = _value_set(side, key)
    if values is not None:
        return values
    stat = stats.column(key)
    if stat.provenance is Provenance.EXACT and stat.min is not None and stat.min == stat.max:
        return frozenset({stat.min})
    return None


# --- absence evidence (bloom + zone map) -------------------------------------


def _bloom_refutes(stat: ColumnStat, value: object) -> bool:
    """Whether `value` is **provably absent** from the column (a bloom has no false negatives).

    Domain-guarded (see the module docstring): a cross-domain probe reports a definitive absence
    for a value that is present, and that deletes rows. A domain we cannot establish declines to
    prune — refusing to prune costs a scan; probing wrongly costs the answer.
    """
    if stat.bloom is None or not _same_bloom_domain(stat, value):
        return False
    index = BloomIndex.from_bytes(stat.bloom)
    return index is not None and not index.contains(value)


def _out_of_range(stat: ColumnStat, value: object) -> bool:
    """Whether `value` lies outside the column's `[min, max]` — so no row can hold it.

    A NaN or zero float bound refutes nothing (`ambiguous_float_bound`): the engine orders
    floats on a total order where `-0.0 < 0.0`, and its key paths canonicalize them back
    together, so a value this comparison calls out-of-range may be one the engine matches.
    Deleting an `IN`-list member on that reasoning deletes rows.
    """
    if stat.min is None or stat.max is None:
        return False
    if ambiguous_float_bound(stat.min) or ambiguous_float_bound(stat.max):
        return False
    try:
        return value < stat.min or value > stat.max
    except TypeError:
        return False  # incomparable types → undecidable, so keep the value


def _all_refuted(values: frozenset | None, other: ColumnStat) -> bool:
    """Whether the other side's bloom proves *every* candidate value absent."""
    if not values or other.bloom is None:
        return False
    return all(_bloom_refutes(other, v) for v in values)
