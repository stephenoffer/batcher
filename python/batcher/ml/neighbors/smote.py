"""`smote` — grow a minority class by interpolating between its own neighbours.

`oversample` duplicates minority rows. A model trained on those sees the same points with
more weight, so it can still memorize them: the decision boundary tightens around the exact
minority examples rather than around the region they occupy.

SMOTE (Chawla et al., 2002) makes *new* points instead. For each minority row it picks one of
its k nearest minority neighbours and places a synthetic row somewhere on the segment between
them, so the class is filled in rather than repeated. That is what makes the boundary
generalize to minority points the training set never contained.

Everything here is expression-level. The minority reference set is bounded and folded in as
literals, and both random draws are content hashes of the row, so the same input produces the
same synthetic rows however the data is partitioned, on one node or a hundred. That
reproducibility is the property a `random.random()` per row would destroy, and it is the one
that lets an imbalanced experiment be repeated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.neighbors.reference import (
    MAX_REFERENCE_ROWS,
    balanced_sum,
    drop_staging,
    read_reference,
    stage_distances,
)
from batcher.plan.expr_ir import Expr, col, lit, nullif, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["smote"]

_RANK = "__bt_smote_rank"
_MIX = "__bt_smote_mix"


def _neighbour_distance(k: int, count: int) -> Expr:
    """The distance to this row's rank-th nearest neighbour, for a per-row random rank.

    `list.get` takes a literal index, so a *dynamic* rank cannot index the sorted distances
    directly. One branch per candidate rank does it instead: `k` is small by construction
    (five by default), so this is a short `CASE`, not a scan.

    Rank 0 is skipped deliberately. A minority row is in its own reference set, so its
    nearest neighbour at distance zero is itself, and interpolating towards itself would
    reproduce the row — which is `oversample`, not SMOTE.
    """
    from batcher.ml.neighbors.reference import DISTANCE_COLUMN

    sorted_distances = col(DISTANCE_COLUMN).list.sort()
    highest = min(k, max(count - 1, 1))
    chosen = sorted_distances.list.get(highest)
    for rank in range(1, highest):
        chosen = (
            when(col(_RANK) == lit(rank)).then(sorted_distances.list.get(rank)).otherwise(chosen)
        )
    return chosen


def smote(
    ds: Dataset,
    label: str,
    *,
    minority: Any,
    features: Sequence[str],
    k: int = 5,
    n_samples: int | None = None,
    seed: int = 0,
    max_reference: int = MAX_REFERENCE_ROWS,
) -> Dataset:
    """Append synthetic minority rows interpolated between real ones.

    Each synthetic row lies on the segment between a real minority row and one of its `k`
    nearest minority neighbours, at a random point along it. The result is `ds` with those
    rows appended, so the returned dataset is longer than the input and the minority class
    is denser.

    Scale the features first. Distance treats every column alike, so a column measured in
    millions decides every neighbour and one measured in fractions is ignored — and here
    that also decides *where the synthetic rows land*.

    Only `features` and `label` are filled on a synthetic row. Every other column is null,
    because there is no honest value to interpolate for an identifier or a free-text field,
    and inventing one would put fabricated data in a column nobody checked.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import smote
            >>> ds = bt.from_pydict(
            ...     {"x": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 5.3, 5.4],
            ...      "label": ["rare", "rare", "rare", "common", "common",
            ...                "common", "common", "common"]}
            ... )
            >>> grown = smote(ds, "label", minority="rare", features=["x"])
            >>> grown.count()
            10
            >>> sorted(set(grown.to_pydict()["label"]))
            ['common', 'rare']

    Args:
        ds: The imbalanced dataset.
        label: The class column.
        minority: The label value to grow.
        features: The numeric columns to interpolate, and to measure distance over.
        k: How many nearest neighbours a synthetic row may be drawn towards.
        n_samples: How many synthetic rows to make. Defaults to the gap between the minority
            class and the largest class, which balances them.
        seed: Seed for the neighbour and interpolation draws.
        max_reference: The ceiling on the minority reference set.

    Returns:
        A new lazy `Dataset`: `ds` with the synthetic rows appended.

    Raises:
        PlanError: If the minority class is empty, has one row, or `k` is not positive.
        ColumnNotFoundError: If a named column is missing.
    """
    from batcher.ml.sampling import class_counts

    names = list(features)
    if not names:
        raise PlanError("smote needs at least one feature column to interpolate")
    if k < 1:
        raise PlanError(f"smote: k must be at least 1, got {k}")

    rows = ds.filter(col(label) == lit(minority))
    points, _ = read_reference(rows, names, [], what="smote", limit=max_reference)
    if len(points) < 2:
        raise PlanError(
            f"smote: class {minority!r} has {len(points)} usable row(s). Interpolation needs "
            "at least two, because a synthetic row lies between a row and a *different* "
            "neighbour. Use oversample() to duplicate a single example instead."
        )

    counts = class_counts(ds, label)
    if n_samples is None:
        n_samples = max(max(counts.values()) - counts.get(minority, 0), 0)
    if n_samples <= 0:
        return ds

    import math

    # Each repeat gets its own seed, so a second pass over the same minority rows lands
    # somewhere else on its segments. Repeating one batch would merely oversample the
    # synthetic rows, which is the thing SMOTE exists to avoid.
    repeats = math.ceil(n_samples / len(points))
    batches = [
        _synthetic_batch(ds, rows, label, minority, names, points, k, seed + 7919 * repeat)
        for repeat in range(repeats)
    ]
    grown = batches[0]
    for batch in batches[1:]:
        grown = grown.union(batch)
    return ds.union(grown.limit(n_samples))


def _synthetic_batch(
    ds: Dataset,
    rows: Dataset,
    label: str,
    minority: Any,
    names: list[str],
    points: list[list[float]],
    k: int,
    seed: int,
) -> Dataset:
    """One synthetic row per minority row, interpolated towards a randomly chosen neighbour.

    The result carries *every* column of `ds` so it can be unioned with it: the features and
    the label are filled, and everything else is null. Inventing a value for an identifier or
    a free-text field would put fabricated data in a column nobody thought to check.
    """
    from batcher.api.dataset._build import split_key
    from batcher.ml.neighbors.reference import DISTANCE_COLUMN

    # Two independent uniforms per row: one picks the neighbour, one the point along the
    # segment. Different seeds keep them from moving together, which would make every
    # synthetic row sit at the same fraction of its own segment.
    highest = min(k, len(points) - 1)
    seeded = rows.with_columns(
        **{
            _RANK: (split_key(rows, names, seed) * lit(float(highest))).floor().cast("int64")
            + lit(1),
            _MIX: split_key(rows, names, seed + 1_000_003),
        }
    )
    staged = stage_distances(seeded, names, points, len(points), tie_break=True)
    target = _neighbour_distance(k, len(points))

    # The chosen neighbour's coordinates, recovered by matching its distance. `tie_break`
    # made the distances strictly ordered, so exactly one reference matches — averaging tied
    # neighbours would return the point between them, which is the base row itself.
    picks = [
        (col(DISTANCE_COLUMN).list.get(index) == target).cast("float64")
        for index in range(len(points))
    ]
    total = balanced_sum(picks)
    synthetic: dict[str, Expr] = {}
    for position, name in enumerate(names):
        neighbour = (
            balanced_sum(
                [
                    pick * lit(float(point[position]))
                    for pick, point in zip(picks, points, strict=True)
                ]
            )
            / total
        )
        base = col(name).cast("float64")
        synthetic[name] = base + col(_MIX) * (neighbour - base)

    made = drop_staging(staged.with_columns(**synthetic))
    projection: dict[str, Expr] = {}
    for name in ds.columns:
        if name in synthetic:
            projection[name] = col(name)
        elif name == label:
            projection[name] = lit(minority)
        else:
            # No honest value exists for a column SMOTE did not interpolate, and carrying the
            # source row's would assert an association that was never observed. `nullif(c, c)`
            # is always null and keeps the column's own type, which a null literal cannot.
            projection[name] = nullif(col(name), col(name))
    return made.select(**projection)
