"""Surface features from a text column — the numbers a model can use before an embedding.

An embedding is the powerful way to featurize text and also the expensive one: a GPU, a
model download, and a vector per row. A great many text signals need none of that. Whether a
review is long, whether a message is all-caps, how many digits a field has, how many words a
title runs to — these are cheap, interpretable, and often most of the signal, and they are
what a gradient-boosted model actually splits on.

`TextStatFeaturizer` computes them as pure string expressions, so a dozen text features over
a billion rows is one pass and no model. It is the first thing to reach for on a text column,
and frequently the last: reach for an embedding when these plateau, not before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import (
    Preprocessor,
    append_projections,
    columns_arg,
    require_column_kind,
)
from batcher.plan.expr_ir.constructors import col, lit, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset
    from batcher.plan.expr_ir import Expr

__all__ = ["TEXT_FEATURES", "TextStatFeaturizer"]

#: The surface features `TextStatFeaturizer` can compute, each a pure string expression.
TEXT_FEATURES = (
    "char_count",
    "word_count",
    "avg_word_length",
    "digit_ratio",
    "upper_ratio",
    "whitespace_ratio",
    "punctuation_count",
)


def _feature_expr(column: str, feature: str) -> Expr:
    """The expression computing one surface feature of a text `column`."""
    text = col(column)
    chars = text.str.len_chars().cast("float64")
    if feature == "char_count":
        return text.str.len_chars()
    if feature == "word_count":
        # Words are runs of non-space, approximated by (space runs + 1) for a non-empty
        # string and 0 for an empty one — cheap and stable, and what a length feature wants.
        spaces = text.str.count_matches(r"\s+")
        return when(chars > lit(0.0)).then(spaces + lit(1)).otherwise(lit(0))
    if feature == "avg_word_length":
        letters = text.str.count_matches(r"\S").cast("float64")
        spaces = text.str.count_matches(r"\s+").cast("float64")
        words = when(chars > lit(0.0)).then(spaces + lit(1.0)).otherwise(lit(1.0))
        return letters / words
    if feature == "digit_ratio":
        return text.str.count_matches(r"[0-9]").cast("float64") / _nonzero(chars)
    if feature == "upper_ratio":
        return text.str.count_matches(r"[A-Z]").cast("float64") / _nonzero(chars)
    if feature == "whitespace_ratio":
        return text.str.count_matches(r"\s").cast("float64") / _nonzero(chars)
    return text.str.count_matches(r"[^\w\s]")  # punctuation_count


def _nonzero(chars: Expr) -> Expr:
    """The character count, with 0 replaced by 1 so a ratio over an empty string is 0."""
    return when(chars > lit(0.0)).then(chars).otherwise(lit(1.0))


class TextStatFeaturizer(Preprocessor):
    """Append cheap, interpretable surface features of a text column — no model, one pass.

    The features it computes:

    ``char_count`` / ``word_count`` / ``avg_word_length``
        How long, in characters and words. Length alone separates a one-word tag from a
        paragraph, and it is often the strongest single text feature.
    ``digit_ratio`` / ``upper_ratio`` / ``whitespace_ratio``
        The character mix. A high digit ratio flags a code or an id masquerading as text; a
        high upper ratio flags shouting or a header row; whitespace ratio flags formatting.
    ``punctuation_count``
        A blunt proxy for structure — a URL, a list, an emoji-laden message.

    Stateless: nothing is learned, so the same expressions apply to training and serving.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import TextStatFeaturizer
            >>> ds = bt.from_pydict({"title": ["Hello World 42"]})
            >>> out = TextStatFeaturizer("title", features=["word_count", "char_count"])
            >>> got = out.fit_transform(ds).to_pydict()
            >>> got["title_word_count"], got["title_char_count"]
            ([3], [14])

    Args:
        columns: The text columns to featurize.
        features: Which features to compute; all of `TEXT_FEATURES` when omitted.
        drop_original: Remove the source text column after featurizing it.
    """

    __slots__ = ("columns", "drop_original", "features")

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        features: Sequence[str] = TEXT_FEATURES,
        drop_original: bool = False,
    ) -> None:
        self.columns = columns_arg(columns, what="TextStatFeaturizer")
        names = list(features)
        if not names:
            raise PlanError("TextStatFeaturizer needs at least one feature")
        for name in names:
            if name not in TEXT_FEATURES:
                from batcher._internal.errors import suggestion

                hint = suggestion(name, TEXT_FEATURES)
                tail = f" {hint}" if hint else ""
                raise PlanError(
                    f"unknown text feature {name!r}; expected one of {sorted(TEXT_FEATURES)}.{tail}"
                )
        self.features = names
        self.drop_original = drop_original

    def transform(self, ds: Dataset) -> Dataset:
        """Append one ``{column}_{feature}`` column per column and feature.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import TextStatFeaturizer
                >>> ds = bt.from_pydict({"t": ["ABC 12"]})
                >>> out = TextStatFeaturizer("t", features=["digit_ratio"]).transform(ds)
                >>> round(out.to_pydict()["t_digit_ratio"][0], 4)
                0.3333

        Args:
            ds: The dataset to featurize.

        Returns:
            A new lazy `Dataset` with the text-feature columns appended.
        """
        require_column_kind(ds, self.columns, what="TextStatFeaturizer", kind="string")
        projections = {
            f"{name}_{feature}": _feature_expr(name, feature)
            for name in self.columns
            for feature in self.features
        }
        return append_projections(ds, projections, self.columns, drop_original=self.drop_original)
