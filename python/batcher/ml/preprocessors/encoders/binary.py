"""Binary encoding — a compact base-2 code for a categorical column.

One-hot encoding spends one column per category, which is wasteful and memory-hungry when a
column has hundreds of categories. Binary encoding instead assigns each category an integer and
writes that integer in base 2 across ``ceil(log2(n + 1))`` columns — so 100 categories cost 7
columns rather than 100. It keeps almost all of one-hot's separability while collapsing the width,
which is the right trade for a high-cardinality feature feeding a linear model or a tree.

`fit` learns the category-to-integer map in one bounded `distinct`; `transform` lowers to a
`when` chain for the integer and cheap bitwise shifts for the digits, so there is no per-row
Python.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import MAX_CATEGORIES, Preprocessor, distinct_values
from batcher.ml.preprocessors.encoders.ordinal import ordinal_expr

if TYPE_CHECKING:
    from typing import Any

    from batcher.api.dataset import Dataset

__all__ = ["BinaryEncoder"]


class BinaryEncoder(Preprocessor):
    """Encode a categorical column as its integer index in base 2, one column per bit.

    Learns a stable integer per category, then writes that integer as binary digits across
    ``{column}_0`` (the least significant bit) upward. An unseen category at serving time encodes
    as all-zero bits, distinct from every learned category. The result is far narrower than
    `OneHotEncoder` for a high-cardinality column while keeping the categories linearly separable.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import BinaryEncoder
            >>> ds = bt.from_pydict({"c": ["a", "b", "c", "a"]})
            >>> out = BinaryEncoder("c").fit_transform(ds).to_pydict()
            >>> out["c_0"], out["c_1"]
            ([1, 0, 1, 1], [0, 1, 1, 0])

    Args:
        column: The categorical column to encode.
        drop_original: Remove the source column after encoding it.
        max_categories: The ceiling on the learned category set.
    """

    __slots__ = ("categories_", "column", "drop_original", "max_categories", "n_bits_")

    def __init__(
        self, column: str, *, drop_original: bool = True, max_categories: int = MAX_CATEGORIES
    ) -> None:
        if not isinstance(column, str):
            raise PlanError(f"BinaryEncoder takes one column name, got {column!r}")
        self.column = column
        self.drop_original = drop_original
        self.max_categories = max_categories
        self.categories_: list[Any] = []
        self.n_bits_: int = 0

    def fit(self, ds: Dataset) -> BinaryEncoder:
        """Learn the category set and how many bits its integer codes need.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import BinaryEncoder
                >>> BinaryEncoder("c").fit(bt.from_pydict({"c": ["x", "y", "z"]})).n_bits_
                2

        Args:
            ds: The dataset to learn the category set from.

        Returns:
            ``self``, fitted.
        """
        self.categories_ = distinct_values(
            ds, self.column, what="BinaryEncoder", max_categories=self.max_categories
        )
        # Codes run 1..n (0 is reserved for an unseen category), so the largest code is n.
        self.n_bits_ = max(len(self.categories_), 1).bit_length()
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Append one 0/1 bit column per binary digit of each row's category code.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import BinaryEncoder
                >>> ds = bt.from_pydict({"c": ["a", "b"]})
                >>> BinaryEncoder("c", drop_original=False).fit(ds).transform(ds).columns
                ['c', 'c_0', 'c_1']

        Args:
            ds: The dataset to encode.

        Returns:
            A new lazy `Dataset` with one bit column per binary digit appended.
        """
        self._require_fitted()
        # Reserve 0 for an unseen value by coding learned categories from 1 upward.
        code = ordinal_expr(self.column, self.categories_, unknown_value=-1) + 1
        projections = {
            f"{self.column}_{bit}": code.bitwise_right_shift(bit).bitwise_and(1)
            for bit in range(self.n_bits_)
        }
        out = ds.with_columns(**projections)
        return out.drop(self.column) if self.drop_original else out
