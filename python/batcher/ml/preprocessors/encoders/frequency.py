"""Cardinality-tolerant categorical encoders — frequency, count, rare-bucketing, hashing.

`OneHotEncoder` and `OrdinalEncoder` both learn the category set, which caps them at the
cardinality the driver can hold and the plan can express. Real categorical columns break
that cap constantly: a URL path, a product SKU, a user agent, a postcode.

The three strategies here are what to do instead, in increasing order of how much
cardinality they tolerate:

`FrequencyEncoder`
    Replace each category with how often it occurs. One number per category, and it is
    often genuinely predictive — a rare value behaves differently from a common one.
`RareCategoryEncoder`
    Keep the categories worth keeping and collapse the tail into one bucket. This is the
    step that makes a one-hot encoding possible at all on a long-tailed column, and it also
    fixes the serving-time unknown-category problem, because the bucket already exists.
`HashingEncoder`
    Hash into a fixed number of buckets. Unbounded cardinality, no fitted state at all, and
    therefore no train/serve skew — at the cost of collisions, which a model tolerates
    better than most people expect.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import MAX_CATEGORIES, Preprocessor, columns_arg
from batcher.plan.expr_ir.constructors import col, lit, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset
    from batcher.plan.expr_ir import Expr

__all__ = ["FrequencyEncoder", "HashingEncoder", "RareCategoryEncoder"]


def _category_counts(ds: Dataset, column: str, limit: int) -> list[tuple[object, int]]:
    """The most frequent values of `column` with their counts, largest first, capped at `limit`."""
    counts = (
        ds.filter(col(column).is_not_null())
        .group_by(column)
        .agg(__bt_n=col(column).count())
        .sort("__bt_n", descending=True)
        .limit(limit)
        .collect()
    )
    values = counts.column(column).to_pylist()
    numbers = counts.column("__bt_n").to_pylist()
    return list(zip(values, numbers, strict=True))


class FrequencyEncoder(Preprocessor):
    """Replace each category with its frequency in the training data.

    Turns an unbounded categorical column into one numeric column that a tree model can
    split on directly. It also carries real signal: "how common is this value" separates a
    mainstream browser from a scraper, or a staple product from a long-tail one, without
    any of the cardinality cost of a one-hot.

    A category unseen at fit time encodes as `unknown_value` (0 by default), which is the
    correct answer — it was never seen, so its training frequency is zero.

    **A null takes the same route** and encodes as `unknown_value`, not as null: a missing
    value is not a category and has no frequency, so it joins the unknown bucket rather than
    inventing one. Impute first (`SimpleImputer`) if a missing value should carry its own
    signal, or add a `MissingIndicator` column before this step to keep it visible.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import FrequencyEncoder
            >>> ds = bt.from_pydict({"c": ["a", "a", "a", "b"]})
            >>> FrequencyEncoder("c").fit_transform(ds).to_pydict()
            {'c': [0.75, 0.75, 0.75, 0.25]}

    Args:
        columns: The categorical columns to encode (replaced in place).
        normalize: Encode the share of rows (default) rather than the raw count.
        max_categories: The ceiling on the learned category set; the tail encodes as
            `unknown_value`.
        unknown_value: The value an unseen or tail category encodes as.
    """

    __slots__ = ("columns", "frequencies_", "max_categories", "normalize", "unknown_value")

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        normalize: bool = True,
        max_categories: int = MAX_CATEGORIES,
        unknown_value: float = 0.0,
    ) -> None:
        self.columns = columns_arg(columns, what="FrequencyEncoder")
        self.normalize = normalize
        self.max_categories = max_categories
        self.unknown_value = unknown_value
        self.frequencies_: dict[str, dict[object, float]] = {}

    def fit(self, ds: Dataset) -> FrequencyEncoder:
        """Learn each column's category frequencies with one grouped count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import FrequencyEncoder
                >>> pre = FrequencyEncoder("c").fit(bt.from_pydict({"c": ["a", "b", "b"]}))
                >>> pre.frequencies_["c"]["b"]
                0.6666666666666666

        Args:
            ds: The dataset to learn the frequencies from.

        Returns:
            ``self``, fitted.
        """
        total = ds.count()
        for name in self.columns:
            pairs = _category_counts(ds, name, self.max_categories)
            divisor = float(total) if self.normalize and total else 1.0
            self.frequencies_[name] = {value: count / divisor for value, count in pairs}
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Replace each fitted column with its learned frequency.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import FrequencyEncoder
                >>> ds = bt.from_pydict({"c": ["a", "b", "b"]})
                >>> pre = FrequencyEncoder("c", normalize=False).fit(ds)
                >>> pre.transform(bt.from_pydict({"c": ["a", "z"]})).to_pydict()
                {'c': [1.0, 0.0]}

        Args:
            ds: The dataset to encode.

        Returns:
            A new lazy `Dataset` with the fitted columns replaced by their frequency.
        """
        self._require_fitted()
        return ds.with_columns(**{name: self._expr(name) for name in self.columns})

    def _expr(self, name: str) -> Expr:
        """A CASE ladder mapping each learned category to its frequency."""
        builder = None
        for value, frequency in self.frequencies_[name].items():
            branch = col(name) == lit(value)
            builder = (
                when(branch).then(lit(float(frequency)))
                if builder is None
                else builder.when(branch).then(lit(float(frequency)))
            )
        if builder is None:
            return lit(float(self.unknown_value))
        return builder.otherwise(lit(float(self.unknown_value)))


class RareCategoryEncoder(Preprocessor):
    """Collapse infrequent categories into a single bucket, keeping the rest unchanged.

    The step that makes every other categorical encoder work on a real column. A category
    seen three times in a million rows cannot be learned from, contributes an all-but-empty
    one-hot column, and is the reason a fitted encoder explodes at serving time on a value
    it has never seen. Bucketing the tail solves all three at once, and the bucket itself
    becomes the natural home for an unknown category.

    **A null lands in the bucket too**, taking `other_value` rather than staying null. That
    makes a missing value a *present* one from here on, which is what the downstream encoder
    needs but is worth knowing: add a `MissingIndicator` column before this step if the
    difference between "rare" and "absent" carries signal for your model.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import RareCategoryEncoder
            >>> ds = bt.from_pydict({"c": ["a", "a", "a", "b"]})
            >>> RareCategoryEncoder("c", min_frequency=0.5).fit_transform(ds).to_pydict()
            {'c': ['a', 'a', 'a', '__rare__']}

    Args:
        columns: The categorical columns to bucket (replaced in place).
        min_frequency: The minimum share of rows a category needs to survive, in ``(0, 1]``.
        max_categories: Keep at most this many categories, most frequent first.
        other_value: The label the collapsed tail takes.
    """

    __slots__ = ("columns", "kept_", "max_categories", "min_frequency", "other_value")

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        min_frequency: float = 0.01,
        max_categories: int = MAX_CATEGORIES,
        other_value: str = "__rare__",
    ) -> None:
        self.columns = columns_arg(columns, what="RareCategoryEncoder")
        if not 0.0 < min_frequency <= 1.0:
            raise PlanError(f"min_frequency must be in (0, 1], got {min_frequency}")
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.other_value = other_value
        self.kept_: dict[str, list[object]] = {}

    def fit(self, ds: Dataset) -> RareCategoryEncoder:
        """Learn which categories clear `min_frequency` (and fit within `max_categories`).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import RareCategoryEncoder
                >>> pre = RareCategoryEncoder("c", min_frequency=0.5).fit(
                ...     bt.from_pydict({"c": ["a", "a", "b"]})
                ... )
                >>> pre.kept_
                {'c': ['a']}

        Args:
            ds: The dataset to learn the surviving categories from.

        Returns:
            ``self``, fitted.
        """
        total = ds.count()
        import pyarrow.types as arrow_types

        schema = ds.schema
        for name in self.columns:
            index = schema.get_field_index(name)
            if index < 0:
                continue
            dtype = schema.field(index).type
            if arrow_types.is_floating(dtype) or arrow_types.is_integer(dtype):
                # This is the mirror image of every other check here: the replacement is a
                # *string* sentinel, so it is the numeric column that cannot work, not the
                # string one. Left alone, `transform` reached a `case` whose branches had
                # different types and the engine said "arguments need to have the same data
                # type", which names neither the column nor which two types it meant.
                raise PlanError(
                    f"RareCategoryEncoder replaces a rare category with the string "
                    f"{self.other_value!r}, so it cannot rewrite the numeric column "
                    f"{name!r}. Cast it to a string first, or bucket it with "
                    "KBinsDiscretizer."
                )
        threshold = self.min_frequency * total
        for name in self.columns:
            pairs = _category_counts(ds, name, self.max_categories)
            self.kept_[name] = [value for value, count in pairs if count >= threshold]
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Replace every category outside the learned set with `other_value`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import RareCategoryEncoder
                >>> ds = bt.from_pydict({"c": ["a", "a", "b"]})
                >>> pre = RareCategoryEncoder("c", min_frequency=0.5).fit(ds)
                >>> pre.transform(bt.from_pydict({"c": ["a", "z"]})).to_pydict()
                {'c': ['a', '__rare__']}

        Args:
            ds: The dataset to bucket.

        Returns:
            A new lazy `Dataset` with the rare categories collapsed.
        """
        self._require_fitted()
        projections = {}
        for name in self.columns:
            kept = self.kept_[name]
            if not kept:
                projections[name] = lit(self.other_value)
                continue
            keep = col(name) == lit(kept[0])
            for value in kept[1:]:
                keep = keep | (col(name) == lit(value))
            projections[name] = when(keep).then(col(name)).otherwise(lit(self.other_value))
        return ds.with_columns(**projections)


class HashingEncoder(Preprocessor):
    """Hash a categorical column into `n_buckets` integer buckets — the stateless encoder.

    The only categorical encoding with no fitted state, which is its whole point: there is
    nothing to persist, nothing to keep in sync between training and serving, and no
    unknown-category failure mode, because every possible value already has a bucket. The
    hashing trick is what makes an unbounded categorical (a URL, a device id) usable at all.

    The cost is collisions: two categories sharing a bucket are indistinguishable to the
    model. With `n_buckets` well above the effective cardinality the loss is small, and a
    tree model in particular degrades gracefully.

    Uses the engine's stable ``xxhash64``, so the same value hashes to the same bucket in
    every process, on every machine, and across restarts — which a Python ``hash()`` does
    not, and that difference is a silent train/serve skew.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import HashingEncoder
            >>> ds = bt.from_pydict({"c": ["a", "b", "a"]})
            >>> out = HashingEncoder("c", n_buckets=8).fit_transform(ds).to_pydict()["c"]
            >>> out[0] == out[2] and 0 <= out[1] < 8
            True

    Args:
        columns: The categorical columns to hash (replaced in place).
        n_buckets: How many buckets to hash into; larger means fewer collisions.
    """

    __slots__ = ("columns", "n_buckets")

    def __init__(self, columns: str | Sequence[str], *, n_buckets: int = 256) -> None:
        self.columns = columns_arg(columns, what="HashingEncoder")
        if n_buckets < 2:
            raise PlanError(f"n_buckets must be at least 2, got {n_buckets}")
        self.n_buckets = n_buckets

    def transform(self, ds: Dataset) -> Dataset:
        """Replace each column with its value's hash bucket.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import HashingEncoder
                >>> ds = bt.from_pydict({"c": ["a", "b"]})
                >>> all(0 <= v < 4 for v in HashingEncoder("c", n_buckets=4)
                ...     .transform(ds).to_pydict()["c"])
                True

        Args:
            ds: The dataset to encode.

        Returns:
            A new lazy `Dataset` with the fitted columns replaced by a bucket index.
        """
        return ds.with_columns(
            **{
                name: (col(name).cast("string").str.xxhash64() % lit(self.n_buckets)).abs()
                for name in self.columns
            }
        )
