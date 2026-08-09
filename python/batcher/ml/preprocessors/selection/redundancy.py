"""`DropCorrelated` — remove one of every pair of columns that say the same thing.

Two features with a correlation of 0.99 carry one feature's worth of information and two
features' worth of cost. Worse, for a linear model they are actively harmful: the fit has to
split one effect between two nearly identical columns, so the coefficients become large,
opposite in sign, and unstable under resampling, and any story you tell about them is
noise.

`batcher.ml.selection.correlated_columns` already finds the pairs. This wraps that in the
`Preprocessor` contract so the *same* columns are dropped from the validation split, which
is the part that has to be fitted state rather than recomputed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["DropCorrelated"]


class DropCorrelated(Preprocessor):
    """Drop one column from every pair correlated above `threshold`.

    Which one goes is decided by position in `columns`, so the choice is reproducible rather
    than dependent on dict ordering: of a correlated pair, the one appearing later is
    dropped. Pass `keep` to protect columns that must survive whatever they correlate with;
    their partners are dropped instead.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import DropCorrelated
            >>> ds = bt.from_pydict(
            ...     {"a": [1.0, 2.0, 3.0, 4.0], "a_copy": [2.0, 4.0, 6.0, 8.0],
            ...      "b": [1.0, 0.0, 1.0, 0.0]}
            ... )
            >>> DropCorrelated(threshold=0.95).fit(ds).dropped_
            ['a_copy']

    Args:
        columns: The numeric columns to consider; defaults to every numeric column.
        threshold: The absolute correlation above which a pair counts as redundant.
        keep: Columns never dropped, whatever they correlate with.
    """

    __slots__ = ("columns", "dropped_", "keep", "threshold")

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        *,
        threshold: float = 0.95,
        keep: Sequence[str] = (),
    ) -> None:
        if not 0 < threshold <= 1:
            raise PlanError(f"DropCorrelated: threshold must be in (0, 1], got {threshold!r}")
        self.columns = list(columns) if columns is not None else None
        self.threshold = threshold
        self.keep = sorted(keep)
        self.dropped_: list[str] = []

    def fit(self, ds: Dataset) -> DropCorrelated:
        """Find the redundant columns with one correlation pass and record them.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import DropCorrelated
                >>> ds = bt.from_pydict({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
                >>> DropCorrelated().fit(ds).dropped_
                ['b']

        Args:
            ds: The training split to measure the correlations on.

        Returns:
            ``self``, fitted.
        """
        from batcher.ml.selection import correlated_columns

        self.dropped_ = correlated_columns(
            ds, self.columns, threshold=self.threshold, keep=self.keep
        )
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Drop the redundant columns found by `fit`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import DropCorrelated
                >>> ds = bt.from_pydict({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
                >>> DropCorrelated().fit_transform(ds).columns
                ['a']

        Args:
            ds: The dataset to prune.

        Returns:
            A new lazy `Dataset` without the redundant columns.
        """
        self._require_fitted()
        gone = set(self.dropped_)
        return ds.select(*[c for c in ds.columns if c not in gone])
