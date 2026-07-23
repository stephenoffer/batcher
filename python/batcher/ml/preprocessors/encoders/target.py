"""Mean (likelihood) target encoding, plain and cross-fitted.

`fit` is one mergeable ``group_by(col).agg(count, sum)`` per column; `transform` is a
lazy CASE `Expr`. Plain encoding leaks the target into the training features, so `cv`
adds the cross-fitted variant: each row is encoded from the *other* folds only, which is
what makes the encoding safe to fit and apply on the same split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import (
    MAX_CATEGORIES,
    Preprocessor,
    check_cardinality,
    columns_arg,
    fit_aggregate,
)
from batcher.plan.expr_ir import Expr, coalesce, col, hash_rows, lit, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["TargetEncoder"]

_FOLD = "__te_fold"


def target_expr(column: str, mapping: dict[Any, float], prior: float) -> Expr:
    """A CASE expression mapping each category to its smoothed target mean, else `prior`."""
    builder = None
    for cat, value in mapping.items():
        cond = col(column) == cat
        builder = when(cond).then(value) if builder is None else builder.when(cond).then(value)
    if builder is None:
        return lit(prior)
    return builder.otherwise(prior)


class TargetEncoder(Preprocessor):
    """Replace each categorical column with the smoothed mean of a target column.

    Mean (a.k.a. likelihood) target encoding — the standard high-cardinality-categorical
    encoding for gradient-boosted and linear tabular models (scikit-learn ``TargetEncoder``,
    cuML, ``category_encoders``). Each category maps to an m-estimate shrinkage of its
    per-category target mean toward the global mean::

        encoding(cat) = (n·mean(cat) + m·prior) / (n + m)

    where ``n`` is the category's row count, ``prior`` the global target mean, and ``m`` the
    `smoothing` weight — so rare categories fall back to the prior and cannot overfit. `fit`
    is one mergeable ``group_by(col).agg(count, sum)`` per column; `transform` is a lazy CASE
    `Expr`. Unseen categories (and nulls) map to `prior`.

    Without `cv` this is plain encoding, and every row's own target contributes to its own
    feature: fit on the training split only, or the target leaks. With ``cv=k`` set,
    `fit_transform` returns the **cross-fitted** encoding, where each row is encoded from
    the other ``k-1`` folds only, which removes that leak. Folds are assigned by hashing
    the row, so they are deterministic and need no shuffle. `transform` (on a held-out
    split, where there is no leak to remove) always uses the full-data mapping.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import TargetEncoder
            >>> ds = bt.from_pydict({"c": ["a", "a", "b", "b"], "y": [1.0, 1.0, 0.0, 0.0]})
            >>> TargetEncoder(["c"], "y", smoothing=0.0).fit_transform(ds).to_pydict()["c"]
            [1.0, 1.0, 0.0, 0.0]

    Args:
        columns: the categorical columns to replace in place with their target encoding.
        target: the (numeric or 0/1) target column whose mean supplies the encoding.
        smoothing: the m-estimate weight pulling small categories toward the global mean.
        cv: the number of cross-fitting folds (>= 2), or None for plain encoding. Only
            `fit_transform` is cross-fitted; row order is not preserved when it is.
        max_categories: the ceiling on each column's fitted cardinality — one CASE arm each.
    """

    __slots__ = ("columns", "cv", "mapping_", "max_categories", "prior_", "smoothing", "target")

    def __init__(
        self,
        columns: str | Sequence[str],
        target: str,
        *,
        smoothing: float = 10.0,
        cv: int | None = None,
        max_categories: int = MAX_CATEGORIES,
    ) -> None:
        self.columns = columns_arg(columns, what="TargetEncoder")
        if not self.columns:
            raise PlanError("TargetEncoder requires at least one column")
        if cv is not None and cv < 2:
            raise PlanError(
                f"TargetEncoder cv must be >= 2 (or None for no cross-fitting), got {cv}"
            )
        self.target = target
        self.smoothing = smoothing
        self.cv = cv
        self.max_categories = max_categories
        self.prior_: float = 0.0
        self.mapping_: dict[str, dict[Any, float]] = {}

    def _encode(self, n: float, s: float) -> float:
        """The m-estimate encoding of a category with `n` rows summing to `s`."""
        denominator = n + self.smoothing
        if denominator <= 0:
            return self.prior_
        return (s + self.smoothing * self.prior_) / denominator

    def fit(self, ds: Dataset) -> TargetEncoder:
        """Learn each category's smoothed target mean and the global prior.

        Stored in `mapping_[col][category]` with the global mean in `prior_`; each is one
        mergeable ``group_by(col).agg(count, sum)`` pass over `ds`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import TargetEncoder
                >>> ds = bt.from_pydict({"c": ["a", "a", "b"], "y": [1.0, 0.0, 1.0]})
                >>> round(TargetEncoder(["c"], "y").fit(ds).prior_, 4)
                0.6667

        Args:
            ds: The (training) dataset supplying both the categories and the target.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If a column has more than `max_categories` distinct values.
        """
        prior = fit_aggregate(ds, {"_p": col(self.target).mean()})["_p"]
        self.prior_ = float(prior) if prior is not None else 0.0
        for c in self.columns:
            grp = ds.group_by(c).agg(_n=col(self.target).count(), _s=col(self.target).sum())
            rows = grp.limit(self.max_categories + 1).to_pydict()
            mapping: dict[Any, float] = {}
            for cat, n, s in zip(rows[c], rows["_n"], rows["_s"], strict=False):
                if cat is None or not n:
                    continue
                mapping[cat] = self._encode(n, s)
            check_cardinality("TargetEncoder", c, len(rows[c]), self.max_categories, exact=False)
            self.mapping_[c] = mapping
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Replace each fitted column with its smoothed target encoding.

        Categories unseen at fit time (and nulls) map to the global `prior_`. This is
        always the full-data mapping, including when `cv` is set: cross-fitting exists to
        de-bias the *training* rows, and a held-out split has no such bias.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import TargetEncoder
                >>> ds = bt.from_pydict({"c": ["a", "a", "b", "b"], "y": [1.0, 1.0, 0.0, 0.0]})
                >>> enc = TargetEncoder(["c"], "y", smoothing=0.0).fit(ds)
                >>> enc.transform(bt.from_pydict({"c": ["a", "z"]})).to_pydict()["c"]
                [1.0, 0.5]

        Args:
            ds: The dataset to encode.

        Returns:
            A new lazy `Dataset` with each fitted column replaced by its encoding.
        """
        self._require_fitted()
        new = {c: target_expr(c, self.mapping_[c], self.prior_) for c in self.columns}
        return ds.with_columns(**new)

    def fit_transform(self, ds: Dataset) -> Dataset:
        """`fit(ds)` then encode `ds` — cross-fitted when `cv` is set.

        With `cv` set each row is encoded from the folds it is *not* in, so the encoding
        of a row never sees that row's own target. Without `cv` this is `fit` followed by
        the plain `transform`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import TargetEncoder
                >>> ds = bt.from_pydict({"c": list("aabb"), "y": [1.0, 0.0, 1.0, 0.0]})
                >>> enc = TargetEncoder(["c"], "y", smoothing=0.0, cv=2).fit_transform(ds)
                >>> sorted(enc.to_pydict()["y"])
                [0.0, 0.0, 1.0, 1.0]

        Args:
            ds: The **training** split to fit on and encode.

        Returns:
            A new lazy `Dataset` with each fitted column replaced by its encoding.
        """
        self.fit(ds)
        if self.cv is None:
            return self.transform(ds)
        return self._cross_fitted(ds)

    def _fold_stats(
        self, folded: Dataset, column: str
    ) -> dict[tuple[Any, int], tuple[float, float]]:
        """Per ``(category, fold)`` ``(count, sum)`` of the target — one mergeable pass."""
        grp = folded.group_by(column, _FOLD).agg(
            _n=col(self.target).count(), _s=col(self.target).sum()
        )
        limit = self.max_categories * (self.cv or 1) + 1
        rows = grp.limit(limit).to_pydict()
        stats: dict[tuple[Any, int], tuple[float, float]] = {}
        for cat, fold, n, s in zip(rows[column], rows[_FOLD], rows["_n"], rows["_s"], strict=False):
            if cat is None:
                continue
            stats[(cat, fold)] = (float(n or 0), float(s or 0.0))
        return stats

    def _out_of_fold_table(
        self, column: str, stats: dict[tuple[Any, int], tuple[float, float]]
    ) -> Dataset:
        """The ``(category, fold) -> encoding`` lookup, each built from the other folds."""
        from batcher.api.session import from_pydict

        totals: dict[Any, tuple[float, float]] = {}
        for (cat, _fold), (n, s) in stats.items():
            have = totals.get(cat, (0.0, 0.0))
            totals[cat] = (have[0] + n, have[1] + s)
        cats: list[Any] = []
        folds: list[int] = []
        values: list[float] = []
        for (cat, fold), (n, s) in stats.items():
            total_n, total_s = totals[cat]
            cats.append(cat)
            folds.append(int(fold))
            values.append(self._encode(total_n - n, total_s - s))
        return from_pydict({column: cats, _FOLD: folds, f"__te_{column}": values})

    def _cross_fitted(self, ds: Dataset) -> Dataset:
        """Encode `ds` out-of-fold: join each row's ``(category, fold)`` to the other folds.

        The fold is a hash of the whole row, so it is deterministic, needs no shuffle, and
        puts identical rows in the same fold. The per-fold lookup is a small joined table
        rather than a ``cv * cardinality`` CASE chain.
        """
        keys = [col(c) for c in ds.columns]
        folded = ds.with_columns(**{_FOLD: hash_rows(*keys) % (self.cv or 1)})
        out = folded
        for c in self.columns:
            lookup = self._out_of_fold_table(c, self._fold_stats(folded, c))
            out = out.join(lookup, on=[c, _FOLD], how="left")
        replaced = {
            c: coalesce(col(f"__te_{c}"), lit(self.prior_)).cast("float64") for c in self.columns
        }
        return out.with_columns(**replaced).select(*ds.columns)
