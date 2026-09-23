"""The fixpoint driver every iterative graph algorithm runs on.

PageRank, connected components, label propagation and breadth-first search are the same
program with a different step: hold a per-node state table, join it along the edges,
aggregate at the far end, and repeat until it stops changing. This module is that loop,
written once.

Two things it does that are easy to get wrong and expensive to get wrong.

**It materializes the state each round.** A `Dataset` is lazy, so `step(step(step(s)))`
is one plan three iterations deep, and executing it at the end re-runs every earlier
iteration from the source. Fifty rounds of PageRank would be fifty *nested* rounds, which
is quadratic in iterations and is why a naive relational PageRank appears to hang. Cutting
the plan once per round makes the cost linear. The materialized state is one row per node,
not per edge, so this is bounded by the smaller side.

**It checks convergence on the state, not on the round count.** Most graphs converge long
before the iteration cap, and the cap is a safety net rather than a schedule. The check
costs one aggregate per round, which is far less than the round it saves.

What a caller does when the cap is hit depends on what kind of answer it promises, and the
two helpers below are the only two policies. An *approximate* algorithm (PageRank, Katz,
eigenvector, HITS, label propagation) returns its last iterate and warns with
`ConvergenceWarning` through `settled`. An *exact* one (components, k-core, shortest paths,
topological order) has no "nearly right" answer: a component split in two is simply wrong.
So those run to their fixpoint by default, bounded by the node count through
`fixpoint_rounds`, and `require_fixpoint` raises `PlanError` if a cap the caller chose cuts
them short.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset

__all__ = ["ConvergenceWarning", "IterationResult", "checkpoint", "iterate"]


class ConvergenceWarning(UserWarning):
    """An approximate graph algorithm stopped at its iteration cap before it settled.

    The result is still returned, because the last iterate of PageRank or Katz is usually a
    usable ranking. It is not the fixed point the algorithm defines, though, and returning
    it silently is how a truncated run gets reported as a converged one. Raise
    `max_iterations` or loosen `tolerance`, or promote the warning to an error with
    `warnings.simplefilter("error", ConvergenceWarning)` when a truncated answer is not
    acceptable.

    Exact algorithms never warn: they run to their fixpoint, and raise `PlanError` when a
    `max_iterations` you set is too small to reach it.

    Examples:
        .. doctest::

            >>> import warnings
            >>> import batcher as bt
            >>> from batcher.graph import ConvergenceWarning, Graph, pagerank
            >>> chain = Graph.from_edges(bt.from_pydict({"src": [1, 2], "dst": [2, 3]}))
            >>> with warnings.catch_warnings(record=True) as caught:
            ...     warnings.simplefilter("always")
            ...     _ = pagerank(chain, max_iterations=1)
            >>> caught[0].category is ConvergenceWarning
            True
    """


def settled(result: IterationResult, algorithm: str, max_iterations: int) -> Dataset:
    """The state an approximate algorithm reached, warning if the cap cut it short.

    Args:
        result: What `iterate` returned.
        algorithm: The public function name, for the message.
        max_iterations: The cap the caller passed, for the message.

    Returns:
        The final state, whether or not it converged.
    """
    if not result.converged:
        warnings.warn(
            f"{algorithm}() stopped at max_iterations={max_iterations} before converging, "
            "so the result is the last iterate rather than the fixed point. Raise "
            "max_iterations or loosen tolerance.",
            ConvergenceWarning,
            stacklevel=3,
        )
    return result.state


def fixpoint_rounds(max_iterations: int | None, nodes: int) -> int:
    """The round budget for an exact algorithm: the caller's cap, or one the graph cannot exceed.

    Every exact algorithm here changes at least one node's state per round until it stops,
    or settles at least one node per round, so `nodes + 1` rounds always reach the fixpoint.
    That is what `None` means: run to completion.

    Args:
        max_iterations: The caller's cap, or `None` for no cap.
        nodes: The node count, which bounds the rounds any fixpoint here needs.

    Returns:
        The number of rounds to allow.

    Raises:
        PlanError: If `max_iterations` is given and not positive.
    """
    if max_iterations is None:
        return nodes + 1
    if max_iterations < 1:
        raise PlanError(f"max_iterations must be at least 1, got {max_iterations}")
    return max_iterations


def require_fixpoint(converged: bool, algorithm: str, rounds: int) -> None:
    """Refuse to return an exact algorithm's answer when the round cap stopped it early.

    Args:
        converged: Whether the loop reached its fixpoint.
        algorithm: The public function name, for the message.
        rounds: The cap that was hit.

    Raises:
        PlanError: If `converged` is false.
    """
    if not converged:
        raise PlanError(
            f"{algorithm}() did not reach its fixpoint within max_iterations={rounds} rounds. "
            "It computes an exact answer, and a truncated run would be a wrong one rather "
            "than an approximate one, so it is refused. Raise max_iterations, or pass "
            "max_iterations=None (the default) to run to completion."
        )


def check_iterations(max_iterations: int, tolerance: float) -> None:
    """Validate the convergence parameters every iterative graph algorithm takes.

    Shared rather than repeated: `centrality/rank.py` and `centrality/spectral.py` each
    carried a byte-identical private copy, and five call sites between them. Two copies of
    a validation rule is two places to update when a bound changes, and the failure mode is
    that one algorithm starts accepting an argument its neighbour rejects.

    Args:
        max_iterations: The iteration cap; must be at least 1.
        tolerance: The convergence threshold; must be non-negative.

    Raises:
        PlanError: If either argument is outside its valid range.
    """
    if max_iterations < 1:
        raise PlanError(f"max_iterations must be at least 1, got {max_iterations}")
    if tolerance < 0.0:
        raise PlanError(f"tolerance must be non-negative, got {tolerance}")


def checkpoint(state: Dataset) -> Dataset:
    """Cut the lazy plan, so the next round starts from data rather than from a plan.

    Not `Dataset.cache`, which is a different tool: caching memoizes *this* result under
    its plan key, and a further transform is a new uncached result — so the plan under the
    next round is still the whole history. Only collecting and re-wrapping truncates it.

    The cost is that the state travels through the client process once per round. It is
    one row per node, so this is bounded by the node count and not the edge count, which
    is the smaller side of every graph worth running an iterative algorithm on.

    Args:
        state: The per-node state to materialize.

    Returns:
        A dataset over the collected rows.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph._iterate import checkpoint
            >>> checkpoint(bt.from_pydict({"node": [1], "rank": [1.0]})).count()
            1
    """
    return bt.from_arrow(state.collect())


class IterationResult:
    """The state a fixpoint reached, plus how it got there.

    Args:
        state: The final per-node state.
        iterations: How many rounds ran.
        converged: Whether the loop stopped because the state settled rather than
            because it hit the cap.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph._iterate import IterationResult
            >>> r = IterationResult(bt.from_pydict({"node": [1]}), 3, True)
            >>> r.iterations, r.converged
            (3, True)
    """

    __slots__ = ("converged", "iterations", "state")

    def __init__(self, state: Dataset, iterations: int, converged: bool) -> None:
        self.state = state
        self.iterations = iterations
        self.converged = converged

    def __repr__(self) -> str:
        """Render the loop's outcome, not the state's contents."""
        how = "converged" if self.converged else "hit the iteration cap"
        return f"IterationResult({self.iterations} iterations, {how})"


def iterate(
    initial: Dataset,
    step: Callable[[Dataset], Dataset],
    *,
    max_iterations: int,
    delta: Callable[[Dataset, Dataset], float] | None = None,
    tolerance: float = 0.0,
) -> IterationResult:
    """Run `step` until the state stops changing or `max_iterations` rounds have passed.

    Args:
        initial: The starting per-node state.
        step: One round: takes the current state, returns the next.
        max_iterations: The hard cap on rounds. Reaching it is not an error, but the
            result reports `converged=False` so a caller can tell.
        delta: Measures how much the state changed between two rounds. `None` runs the
            full `max_iterations` without checking, which is right when the step is
            cheap enough that measuring costs more than it saves.
        tolerance: The loop stops once `delta` is at or below this.

    Returns:
        The final state, the round count, and whether it converged.

    Raises:
        PlanError: If `max_iterations` is not positive, or `tolerance` is negative.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph._iterate import iterate
            >>> start = bt.from_pydict({"node": [1, 2], "v": [1.0, 1.0]})
            >>> halve = lambda s: s.with_columns(v=bt.col("v") / 2.0)
            >>> out = iterate(start, halve, max_iterations=3)
            >>> out.iterations, out.state.to_pydict()["v"]
            (3, [0.125, 0.125])
    """
    if max_iterations < 1:
        raise PlanError(f"max_iterations must be at least 1, got {max_iterations}")
    if tolerance < 0.0:
        raise PlanError(f"tolerance must be non-negative, got {tolerance}")

    state = checkpoint(initial)
    for round_index in range(1, max_iterations + 1):
        nxt = checkpoint(step(state))
        if delta is not None and delta(state, nxt) <= tolerance:
            return IterationResult(nxt, round_index, True)
        state = nxt
    return IterationResult(state, max_iterations, delta is None)


def max_abs_change(key: str, value: str) -> Callable[[Dataset, Dataset], float]:
    """A `delta` that measures the largest per-node change in a numeric column.

    The right convergence test for PageRank and the other value-propagating algorithms:
    the sum of changes shrinks as the graph grows even when no individual node has
    settled, so a total would declare convergence on a large graph that is still moving.

    Args:
        key: The node-id column both states are keyed by.
        value: The numeric column to compare.

    Returns:
        A function of two states returning the largest absolute difference.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph._iterate import max_abs_change
            >>> a = bt.from_pydict({"node": [1, 2], "v": [1.0, 1.0]})
            >>> b = bt.from_pydict({"node": [1, 2], "v": [1.0, 1.5]})
            >>> max_abs_change("node", "v")(a, b)
            0.5
    """

    def measure(before: Dataset, after: Dataset) -> float:
        joined = before.select(**{key: bt.col(key), "_prev": bt.col(value)}).join(
            after.select(**{key: bt.col(key), "_next": bt.col(value)}), on=key, how="inner"
        )
        got = joined.agg(d=bt.max((bt.col("_next") - bt.col("_prev")).abs())).to_pydict()["d"]
        # An empty graph never changes, so it has converged.
        return float(got[0]) if got and got[0] is not None else 0.0

    return measure


def count_changed(key: str, value: str) -> Callable[[Dataset, Dataset], float]:
    """A `delta` that counts how many nodes changed a discrete label.

    The right test for the label-propagating algorithms (components, communities), where
    a value is an identity rather than a magnitude and "how far it moved" is meaningless.

    Args:
        key: The node-id column both states are keyed by.
        value: The label column to compare.

    Returns:
        A function of two states returning the number of nodes whose label changed.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph._iterate import count_changed
            >>> a = bt.from_pydict({"node": [1, 2], "c": [1, 2]})
            >>> b = bt.from_pydict({"node": [1, 2], "c": [1, 1]})
            >>> count_changed("node", "c")(a, b)
            1.0
    """

    def measure(before: Dataset, after: Dataset) -> float:
        joined = before.select(**{key: bt.col(key), "_prev": bt.col(value)}).join(
            after.select(**{key: bt.col(key), "_next": bt.col(value)}), on=key, how="inner"
        )
        return float(joined.filter(bt.col("_prev") != bt.col("_next")).count())

    return measure
