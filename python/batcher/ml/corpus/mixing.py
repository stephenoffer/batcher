"""Sampling several corpora into one training mixture, at declared weights.

A pretraining or fine-tuning run is almost never one dataset. It is code at 15%, web text at
50%, books at 20%, and a small high-quality set at 5% — and those ratios are a hyperparameter,
not an accident of how much of each you happened to have. Concatenating the sources gives you
the ratio of their *sizes*, which is the wrong number and is silently wrong: the run trains,
the loss falls, and the model is dominated by whichever corpus was biggest.

This samples each source to hit the weights you declared, tags every row with where it came
from, and reports what it actually managed — because a source too small to fill its share is
the failure that otherwise shows up as a mixture that does not match its own configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["MixtureReport", "mix_corpora"]


class MixtureReport:
    """What a mixture actually drew from each source, against what was asked for.

    A source with fewer rows than its weight calls for cannot be up-sampled without repeating
    rows, and repeating rows in a pretraining mixture is a decision (it changes the effective
    epoch count for that source), not a default. `mix_corpora` takes the rows that exist and
    records the shortfall here so it is visible rather than inferred from a loss curve.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import mix_corpora
            >>> web = bt.from_pydict({"text": ["w"] * 100})
            >>> code = bt.from_pydict({"text": ["c"] * 100})
            >>> _, report = mix_corpora({"web": web, "code": code}, {"web": 0.75, "code": 0.25})
            >>> report.shortfalls
            {}
    """

    def __init__(
        self,
        requested: dict[str, int],
        available: dict[str, int],
        taken: dict[str, int],
    ) -> None:
        """Record the per-source row counts a mixture asked for, had, and took.

        Args:
            requested: Rows each source's weight called for.
            available: Rows each source actually holds.
            taken: Rows the mixture drew from each source.
        """
        self.requested = dict(requested)
        self.available = dict(available)
        self.taken = dict(taken)

    @property
    def shortfalls(self) -> dict[str, int]:
        """Sources that could not fill their share, and by how many rows.

        Returns:
            A mapping of source name to the number of rows it fell short by, empty when
            every source met its weight.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import mix_corpora
                >>> big = bt.from_pydict({"text": ["b"] * 100})
                >>> tiny = bt.from_pydict({"text": ["t"] * 2})
                >>> _, report = mix_corpora(
                ...     {"big": big, "tiny": tiny}, {"big": 0.5, "tiny": 0.5}, total_rows=100
                ... )
                >>> report.shortfalls
                {'tiny': 48}
        """
        return {
            name: self.requested[name] - self.taken[name]
            for name in self.requested
            if self.taken[name] < self.requested[name]
        }

    @property
    def realized_weights(self) -> dict[str, float]:
        """The fraction of the output each source actually contributed.

        Returns:
            A mapping of source name to its share of the mixture, summing to 1 when the
            mixture is non-empty.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import mix_corpora
                >>> a = bt.from_pydict({"text": ["a"] * 100})
                >>> b = bt.from_pydict({"text": ["b"] * 100})
                >>> _, report = mix_corpora({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
                >>> report.realized_weights
                {'a': 0.5, 'b': 0.5}
        """
        total = sum(self.taken.values())
        if total == 0:
            return dict.fromkeys(self.taken, 0.0)
        return {name: count / total for name, count in self.taken.items()}

    def __repr__(self) -> str:
        """Show the realized weights and any shortfall, which is what a reader checks."""
        short = f", shortfalls={self.shortfalls}" if self.shortfalls else ""
        weights = {k: round(v, 4) for k, v in self.realized_weights.items()}
        return f"MixtureReport(realized_weights={weights}{short})"


def _validate(sources: dict[str, Dataset], weights: dict[str, float]) -> None:
    """Reject a mixture that cannot be built before any scan happens."""
    if not sources:
        raise PlanError("mix_corpora: at least one source is required")
    missing = sorted(set(sources) - set(weights))
    if missing:
        raise PlanError(f"mix_corpora: no weight given for source(s) {missing}")
    extra = sorted(set(weights) - set(sources))
    if extra:
        raise PlanError(f"mix_corpora: weight(s) for unknown source(s) {extra}")
    negative = sorted(n for n, w in weights.items() if w < 0)
    if negative:
        raise PlanError(f"mix_corpora: weight(s) must not be negative: {negative}")
    if sum(weights.values()) <= 0:
        raise PlanError("mix_corpora: the weights must not all be zero")


def mix_corpora(
    sources: dict[str, Dataset],
    weights: dict[str, float],
    *,
    total_rows: int | None = None,
    source_column: str | None = "source",
    seed: int | None = None,
) -> tuple[Dataset, MixtureReport]:
    """Sample several corpora into one dataset at declared weights.

    The mixture ratio of a training run is a hyperparameter. Concatenating the sources gives
    you the ratio of their sizes instead, which trains perfectly well and produces a model
    dominated by whichever corpus happened to be largest. This draws from each source in
    proportion to its weight, so the ratio in the output is the one you wrote down.

    Weights are normalized, so ``{"a": 3, "b": 1}`` and ``{"a": 0.75, "b": 0.25}`` mean the
    same thing. `total_rows` defaults to the largest mixture the sources can fill without
    repeating any row — a source is never up-sampled, because repeating rows changes the
    effective epoch count for that source and is a decision to make deliberately.

    Every source must share a schema, since the result is one dataset. `source_column` tags
    each row with where it came from, which is what lets you read loss or a quality metric per
    corpus afterwards; pass `None` to leave the schema untouched.

    Args:
        sources: The corpora to mix, keyed by name.
        weights: The target share for each source; normalized, so any positive scale works.
        total_rows: Rows in the output. Defaults to the largest mixture that needs no
            up-sampling.
        source_column: Name of the appended tag column, or `None` to add none.
        seed: Seed for the per-source sampling, for a reproducible mixture.

    Returns:
        The mixed dataset, and a `MixtureReport` of what each source actually contributed.

    Raises:
        PlanError: If a source has no weight, a weight names no source, a weight is
            negative, all weights are zero, or `total_rows` is not positive.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import mix_corpora
            >>> web = bt.from_pydict({"text": ["web"] * 800})
            >>> code = bt.from_pydict({"text": ["code"] * 800})
            >>> mixed, report = mix_corpora(
            ...     {"web": web, "code": code}, {"web": 3, "code": 1}, total_rows=400
            ... )
            >>> report.realized_weights
            {'web': 0.75, 'code': 0.25}
            >>> sorted(set(mixed.select("source").to_pydict()["source"]))
            ['code', 'web']
    """
    from batcher.api.session.combine import concat
    from batcher.plan.expr_ir.constructors import lit

    _validate(sources, weights)
    if total_rows is not None and total_rows <= 0:
        raise PlanError(f"mix_corpora: total_rows must be positive, got {total_rows}")

    scale = sum(weights[name] for name in sources)
    fractions = {name: weights[name] / scale for name in sources}
    available = {name: ds.count() for name, ds in sources.items()}

    if total_rows is None:
        # The largest mixture no source has to repeat itself to fill. A source with weight 0
        # imposes no ceiling, and one with no rows makes the whole mixture empty rather than
        # silently reweighting around it.
        ceilings = [
            int(available[name] / fractions[name]) for name in sources if fractions[name] > 0
        ]
        total_rows = min(ceilings) if ceilings else 0

    requested = {name: int(total_rows * fractions[name]) for name in sources}
    taken = {name: min(requested[name], available[name]) for name in sources}

    parts = []
    for index, (name, ds) in enumerate(sources.items()):
        if taken[name] <= 0:
            continue
        part = (
            ds
            if taken[name] >= available[name]
            else ds.sample(n=taken[name], seed=None if seed is None else seed + index)
        )
        if source_column is not None:
            part = part.with_columns(**{source_column: lit(name)})
        parts.append(part)

    if not parts:
        # Every source contributed nothing. Return an empty frame with the right shape rather
        # than raising: a mixture over empty sources is empty, not invalid.
        first = next(iter(sources.values()))
        empty = first.limit(0)
        if source_column is not None:
            empty = empty.with_columns(**{source_column: lit("")})
        return empty, MixtureReport(requested, available, taken)

    mixed = parts[0] if len(parts) == 1 else concat(*parts)
    return mixed, MixtureReport(requested, available, taken)
