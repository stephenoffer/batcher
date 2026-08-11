"""The prepared-execution cache: derive a small query's execution once, then dispatch it.

A repeated small query re-derives, on every single call, a set of answers that cannot have
changed. `collect()` on a 1,000-row filter costs ~360 us on the fast path, of which the
engine is ~145 us; the other ~215 us is the control plane recomputing the same optimized
plan, the same serialized IR, the same projection and predicate pushdowns, and the same
routing verdict it computed on the previous call. None of those depend on anything but the
plan, its sources and the config -- all of which are frozen.

So this module memoizes the whole derivation behind one dict lookup. On a hit, a terminal
op is: read each source's batches, call the engine, build the table. `latency_bench.py`
calls this "the prize a prepared-statement API would collect", and that is what it is --
except that nothing new is exposed. The user writes the same `ds.collect()`; the second one
is fast because the first one left an entry behind.

## Why a cached derivation is the same derivation

The key carries every input the derivation reads:

`plan.content_key()`
    The plan's IR fingerprint *plus* each node's `identity_suffix()`, which is where `Scan`
    contributes its schema. Two queries that differ in a literal, an operator, a column, or
    a source's column *types* get different keys.
`the source objects themselves`
    Held as weak references and checked with `is` on every hit, because `content_key` is
    not enough on its own: a `Scan`'s IR is only its `source_id`, so two datasets over
    different tables of the same schema share a plan key. Weak rather than strong so the
    cache never keeps a table alive, and identity-checked rather than id-checked so a
    recycled `id()` cannot alias one table onto another's plan.
`the resolved config object`
    Compared with `is`. `resolve_auto_config` deliberately returns the *same* object while
    the sensed memory envelope holds still, so identity is exactly the right test: any
    config change at all, sensed or explicit, builds a new object and misses.
`the terminal's own arguments`
    In the key, so `collect()` and `collect(distributed=True)` cannot share an entry.

## What it does not do

An entry is only ever created by `fast_path.run_fast`, so this cache inherits that path's
gate wholesale -- in-memory sources, a small plan, no UDF, no spill, no cache, CPU backend
-- and its trade: a prepared query does not write to the learned-stats loop. It is off
whenever `execution.fast_path` is off, which is the default. See `fast_path` for why each
skipped stage is safe under that gate, and for what is given up.

Two live inputs deserve naming, because a cached routing verdict would be wrong if either
could move it:

- **The cluster.** `_resolve_distributed` returns `False` for resident sources *at any
  size, whatever the topology* (routing.py:153), and the gate admits only `InMemorySource`.
  So Ray coming up mid-session cannot change this decision, and the cache cannot serve a
  stale one.
- **Memory pressure.** The gate caps resident input rows well inside any envelope, and the
  config-identity check above already misses as soon as the sensed envelope moves.
"""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    from batcher.config import Config
    from batcher.io.source import Source
    from batcher.plan.logical import LogicalPlan
    from batcher.plan.physical import PhysicalPlan

__all__ = ["Prepared", "lookup", "remember"]

#: Distinct prepared queries held before the least-recently-used one is evicted. Sized for a
#: dashboard or a request handler cycling through a working set of query shapes, not for a
#: generator of unique ones -- an entry is a plan and two short strings, so the cache is
#: kilobytes, and the LRU is what keeps a parameterized workload (a fresh literal every
#: call, hence a fresh key every call) from growing it without bound.
MAX_ENTRIES = 256

_CACHE: OrderedDict[tuple, Prepared] = OrderedDict()

#: Guards the LRU bookkeeping. The cache is a process singleton and several pipelines record
#: into it at once, and while each individual `OrderedDict` operation is atomic under the GIL,
#: the *sequences* here are not: `move_to_end(key)` raises `KeyError` if another thread's
#: eviction removed `key` in between. That would surface as a failed query rather than a slow
#: one. Uncontended acquisition is ~40 ns against a ~165 us query, and no engine work happens
#: inside the lock -- only the dict operations.
_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class Prepared:
    """One query's derivation, complete enough to execute without the control plane.

    Everything here is a pure function of the plan, the sources and the config, computed
    once by `fast_path.run_fast` and replayed by `execute`.
    """

    #: The optimized physical plan. Carries the pre-serialized IR (`to_json` is memoized on
    #: the instance) and the per-operator spill budgets the engine config folds in.
    physical: PhysicalPlan
    #: The optimized *logical* plan, for the bare-scan shortcut in `core.scan_only_result`.
    logical: LogicalPlan
    #: Schema for a zero-batch result, so an empty answer still has the promised columns.
    empty_schema: pa.Schema
    #: The config this entry was derived under, compared with `is` on every hit.
    config: Config
    #: Weak references to the source objects, checked with `is` on every hit.
    source_refs: tuple[weakref.ReferenceType, ...]

    def execute(self, sources: list[Source]) -> pa.Table:
        """Read the sources and run the engine -- the whole of a prepared query.

        Mirrors `stages.resolve_sources` followed by `run._execute_in_memory`, minus the
        derivation both of those would redo and minus the source-IO measurement the
        fast-path trade already gives up.

        Args:
            sources: The plan's bound sources, in scan order. Verified by `lookup` to be
                the same objects this entry was derived against.

        Returns:
            The result table -- the same rows, names and types the ordinary path returns.
        """
        from batcher import core
        from batcher.io.source import read_source

        phys = self.physical
        projections, predicates = phys.source_projections, phys.source_predicates
        batches = [
            read_source(src, projections.get(i), predicates.get(i)) for i, src in enumerate(sources)
        ]
        table = core.scan_only_result(self.logical, batches, predicates)
        if table is not None:
            return table
        out = core.execute_local(phys, batches, feedback=None)
        return pa.Table.from_batches(out, schema=out[0].schema if out else self.empty_schema)


def lookup(key: tuple, sources: list[Source], config: Config) -> Prepared | None:
    """The prepared query for `key`, or `None` when there is no usable entry.

    A hit additionally proves the entry was derived against *these* source objects under
    *this* config object; see the module docstring for why neither is implied by the key.

    Args:
        key: The entry key, from `entry_key`.
        sources: The plan's bound sources, in scan order.
        config: The resolved config this query is running under.

    Returns:
        The prepared query, or `None` to derive it the ordinary way.
    """
    entry = _CACHE.get(key)
    if entry is None:
        return None
    if entry.config is not config or len(entry.source_refs) != len(sources):
        return None
    for ref, src in zip(entry.source_refs, sources, strict=True):
        if ref() is not src:
            return None  # a dead or replaced source: this entry is not about these tables
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
    return entry


def remember(key: tuple, sources: list[Source], prepared: Prepared) -> None:
    """Store `prepared` under `key`, evicting the least recently used entry past the cap.

    A source that cannot be weakly referenced is simply not cached: the entry would have no
    way to prove on a later hit that it is about the same table, and a cache that cannot
    prove that is a wrong-answer machine rather than a slow one.

    Args:
        key: The entry key, from `entry_key`.
        sources: The plan's bound sources, in scan order.
        prepared: The derivation to remember.
    """
    try:
        refs = tuple(weakref.ref(src) for src in sources)
    except TypeError:
        return
    with _LOCK:
        _CACHE[key] = _replace_refs(prepared, refs)
        _CACHE.move_to_end(key)
        while len(_CACHE) > MAX_ENTRIES:
            _CACHE.popitem(last=False)


def entry_key(plan: LogicalPlan, sources: list[Source], args: tuple[Any, ...]) -> tuple:
    """The cache key for one terminal call.

    `id(src)` is in the key only to *separate* entries cheaply; it is never trusted to
    identify a source, which is what the weak-reference check in `lookup` is for.

    Args:
        plan: The logical plan, as written.
        sources: The plan's bound sources, in scan order.
        args: The terminal operation's own arguments, as a hashable tuple.

    Returns:
        A hashable key.
    """
    return (plan.content_key(), tuple(id(src) for src in sources), args)


def clear() -> None:
    """Drop every entry. For tests, and for a caller that has replaced a source in place."""
    with _LOCK:
        _CACHE.clear()


def _replace_refs(prepared: Prepared, refs: tuple[weakref.ReferenceType, ...]) -> Prepared:
    """`prepared` with its source references set -- the one field `run_fast` cannot fill."""
    return Prepared(
        physical=prepared.physical,
        logical=prepared.logical,
        empty_schema=prepared.empty_schema,
        config=prepared.config,
        source_refs=refs,
    )
