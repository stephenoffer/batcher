"""Run a top-N from a bound: remembered from the last run, or proved by the source's footers.

Kyber decides whether a top-N is seedable and what bound to use (`kyber.learned_tuning.topn_bound`
for a remembered bound, `kyber.learned_tuning.topn_footer` for one derived from row-group
statistics). What happens *here* is the half a plan -> plan pass cannot express: fetch the
statistics the footer bound is computed from, run the seeded plan, look at how many rows came
back, and run the plan as written when the answer says the bound did not hold.

Split out of `run` so that module stays inside the size budget; `run` hands in the function that
executes one plan, which keeps the dependency one-way.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from batcher._internal.logging import note_suppressed
from batcher.api.source_stats import collect_source_stats

if TYPE_CHECKING:
    from batcher.core import ExecutionContext
    from batcher.io.source import Source
    from batcher.kyber.learned_tuning.topn_bound import TopNSeed
    from batcher.metadata.hub import MetadataHub
    from batcher.plan.logical import LogicalPlan

__all__ = ["footer_seed", "run_seeded_topn"]

# Below this many source rows the footer sweep is not attempted.
#
# The bound saves the rows a reader would decode, and on a relation this small that decode is
# a few milliseconds, which a footer read over an object store can exceed on its own. It is
# also the size under which the whole query is not what anyone is waiting on.
_MIN_FOOTER_SEED_ROWS = 1_000_000

# Per (source version, column) row-group statistics, so a repeated top-N reads its footers once.
_BOUNDS_CACHE: OrderedDict[tuple[str, str], Any] = OrderedDict()
_BOUNDS_CACHE_ENTRIES = 256

RunPlan = Callable[["LogicalPlan"], tuple[Any, Any]]


def run_seeded_topn(
    plan: LogicalPlan,
    sources: list[Source],
    ctx: ExecutionContext,
    run: RunPlan,
    *,
    materialize: bool,
) -> tuple[Any, Any]:
    """Run `plan`, first trying it seeded with a top-N bound when one is available.

    The check is a row count and nothing more, because that is all it has to be: a seeded plan
    removes only rows strictly beyond the bound, so any `k` survivors are the true top-k
    regardless of where the bound came from. A short result is the *only* way seeding can go
    wrong, and it is not a wrong answer, just a wasted scan.

    A remembered bound can only be trusted after that count, so it is tried only when the
    result is materialized. A footer bound is proved from the files being read, so it also
    seeds a streamed run, which has no count to check.

    Args:
        plan: The plan as written.
        sources: The plan's bound inputs.
        ctx: The execution context, whose hub holds remembered bounds.
        run: Executes one plan and returns ``(table, decisions)``.
        materialize: Whether the run produces a table that can be counted.

    Returns:
        What `run` returned for the plan that answered.
    """
    from batcher.kyber.learned_tuning.topn_bound import record_topn_bound, seed_topn_bound

    # The whole loop is keyed on the plan *as written*, never on a seeded rewrite, which is a
    # different shape and would learn a bound under a signature no later run asks for.
    seed = seed_topn_bound(plan, ctx.hub) if materialize else None
    if seed is None:
        seed = footer_seed(plan, sources, ctx.hub)
    if seed is not None:
        table, decisions = run(seed.plan)
        if not materialize or (table is not None and table.num_rows >= seed.k):
            if materialize:
                record_topn_bound(ctx.hub, plan, table)
            return table, decisions
        # The bound no longer separates `k` rows. Nothing about the seeded run is reusable --
        # a wider bound could admit rows it never looked at -- so redo as written.

    table, decisions = run(plan)
    if materialize:
        record_topn_bound(ctx.hub, plan, table)
    return table, decisions


def footer_seed(
    plan: LogicalPlan, sources: list[Source], hub: MetadataHub | None
) -> TopNSeed | None:
    """A top-N bound proved by the scanned source's row-group statistics, or `None`.

    Args:
        plan: The plan as written.
        sources: The plan's bound inputs.
        hub: The metadata hub source statistics are cached against, or `None`.

    Returns:
        The seeded plan, or `None` when the plan is not a seedable top-N over a Parquet
        scan, the source is too small to be worth a footer sweep, or the statistics prove
        nothing useful.
    """
    from batcher.kyber.learned_tuning.topn_footer import footer_topn_seed, topn_scan_key

    key = topn_scan_key(plan)
    if key is None or key.source_id >= len(sources):
        return None
    source = sources[key.source_id]
    read_bounds = getattr(source, "row_group_bounds", None)
    if read_bounds is None:
        return None
    try:
        stats = collect_source_stats([source], hub, need_columns=set())[0]
        if stats is None or (stats.row_count or 0) < _MIN_FOOTER_SEED_ROWS:
            return None
        if (stats.row_group_count or 0) < 2:
            return None  # one row group is read whole whatever the bound says
        bounds = _cached_bounds(source, key.column, read_bounds)
    except Exception as exc:
        note_suppressed("api", "read row-group statistics for a top-n bound", exc)
        return None
    return footer_topn_seed(plan, key, bounds) if bounds else None


def _cached_bounds(source: Source, column: str, read_bounds: Callable[[list[str]], Any]) -> Any:
    """`read_bounds([column])`, memoized on the source's content version when it has one."""
    from batcher.api.source_stats import _cache_key, _source_identity

    version = _cache_key(source, _source_identity(source))
    if version is None:
        return read_bounds([column])
    key = (version, column)
    if key in _BOUNDS_CACHE:
        _BOUNDS_CACHE.move_to_end(key)
        return _BOUNDS_CACHE[key]
    bounds = read_bounds([column])
    _BOUNDS_CACHE[key] = bounds
    if len(_BOUNDS_CACHE) > _BOUNDS_CACHE_ENTRIES:
        _BOUNDS_CACHE.popitem(last=False)
    return bounds
