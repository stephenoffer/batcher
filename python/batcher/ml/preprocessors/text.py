"""Feature assembly and text tokenization.

`Concatenator` stacks several numeric columns into one list column natively (an
`array` expression — no per-row Python), the common "make a feature vector before
training" step. `Tokenizer` maps a text column through a user tokenizer; tokenization
is inherently per-string, so it runs as a `map_batches` UDF (the opaque path), but
stays whole-batch at the engine boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, columns_arg
from batcher.plan.expr_ir import array, col

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from batcher.api.dataset import Dataset

__all__ = ["Concatenator", "Tokenizer"]


def _set_column(batch: Any, name: str, values: Any) -> Any:
    """Replace `name` in `batch` if present, else append it."""
    if name in batch.schema.names:
        return batch.set_column(batch.schema.get_field_index(name), name, values)
    return batch.append_column(name, values)


class Concatenator(Preprocessor):
    """Stack numeric `columns` into a single list column ``output_column``.

    A stateless transform (``fit`` is a no-op): the output is ``array(col, ...)``, a
    native list column ready to become a tensor for training. The source columns are
    kept unless `drop` is set.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import Concatenator
            >>> ds = bt.from_pydict({"a": [1.0, 2.0], "b": [3.0, 4.0]})
            >>> Concatenator(["a", "b"]).fit_transform(ds).to_pydict()
            {'a': [1.0, 2.0], 'b': [3.0, 4.0], 'features': [[1.0, 3.0], [2.0, 4.0]]}

    Args:
        columns: the numeric columns to stack, in order.
        output_column: the name of the assembled list column.
        drop: drop the source columns from the output when True.
    """

    __slots__ = ("columns", "drop", "output_column")

    def __init__(
        self, columns: str | Sequence[str], *, output_column: str = "features", drop: bool = False
    ) -> None:
        self.columns = columns_arg(columns, what="Concatenator")
        self.output_column = output_column
        self.drop = drop

    def transform(self, ds: Dataset) -> Dataset:
        """Add `output_column`, a list column stacking the source columns per row.

        The source columns are dropped when `drop` is set.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import Concatenator
                >>> ds = bt.from_pydict({"a": [1.0, 2.0], "b": [3.0, 4.0]})
                >>> Concatenator(["a", "b"], drop=True).fit_transform(ds).to_pydict()
                {'features': [[1.0, 3.0], [2.0, 4.0]]}

        Args:
            ds: The dataset whose columns to assemble.

        Returns:
            A new lazy `Dataset` with the assembled list column added.
        """
        self._require_fitted()
        out = ds.with_columns(**{self.output_column: array(*(col(c) for c in self.columns))})
        if self.drop:
            keep = [c for c in out.columns if c not in set(self.columns)]
            out = out.select(*keep)
        return out


class Tokenizer(Preprocessor):
    """Tokenize a text column with a user-provided tokenizer (a `map_batches` UDF).

    `tokenizer` is either a HuggingFace-style tokenizer (a callable object that also has
    ``.encode``) or a plain ``str -> list`` callable. A HuggingFace tokenizer is driven
    the way it is meant to be: **one call per Arrow batch** over the whole list of texts,
    which is where its Rust fast path lives, rather than a Python call per row. That also
    unlocks `max_length`, `truncation`, `padding`, and the attention mask an LLM needs.

    A plain ``str -> list`` callable is inherently per string and is still applied per
    string, so pass a real batched tokenizer for anything large. A stateless transform
    (``fit`` is a no-op).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import Tokenizer
            >>> ds = bt.from_pydict({"t": ["a b", "c"]})
            >>> Tokenizer("t", str.split).fit_transform(ds).to_pydict()
            {'t': [['a', 'b'], ['c']]}

    Args:
        column: the text column to tokenize.
        tokenizer: a batched HuggingFace-style tokenizer, a ``str -> list`` callable, or
            an object with ``.encode``.
        output_column: where to put the token-id lists (defaults to `column`).
        max_length: the maximum token count per row, passed to a batched tokenizer.
        truncation: truncate rows longer than `max_length` (batched tokenizers only).
        padding: pad rows to a common width — ``True``, ``"max_length"``, or
            ``"longest"``, passed straight through (batched tokenizers only).
        attention_mask_column: emit the attention mask into this column as well.
    """

    __slots__ = (
        "_batched",
        "_encode",
        "attention_mask_column",
        "column",
        "max_length",
        "output_column",
        "padding",
        "truncation",
    )

    def __init__(
        self,
        column: str,
        tokenizer: Callable[[str], list[Any]] | Any,
        *,
        output_column: str | None = None,
        max_length: int | None = None,
        truncation: bool = False,
        padding: bool | str = False,
        attention_mask_column: str | None = None,
    ) -> None:
        self.column = column
        self.output_column = output_column or column
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        self.attention_mask_column = attention_mask_column
        # A HuggingFace tokenizer is callable *and* carries `.encode`; a bare function is
        # only callable. That is the discriminator for the batched fast path.
        self._batched = callable(tokenizer) and hasattr(tokenizer, "encode")
        encode = tokenizer if self._batched else getattr(tokenizer, "encode", tokenizer)
        if not callable(encode):
            raise PlanError("Tokenizer needs a callable, or an object with .encode")
        self._encode = encode
        if not self._batched and (
            max_length is not None or truncation or padding or attention_mask_column
        ):
            raise PlanError(
                "Tokenizer: max_length / truncation / padding / attention_mask_column need a "
                "batched tokenizer (a callable object with .encode, such as a HuggingFace "
                "tokenizer); a plain str -> list callable supports none of them"
            )

    def _batch_kwargs(self) -> dict[str, Any]:
        """The keyword arguments to forward to a batched tokenizer call."""
        kwargs: dict[str, Any] = {}
        if self.max_length is not None:
            kwargs["max_length"] = self.max_length
        if self.truncation:
            kwargs["truncation"] = True
        if self.padding:
            kwargs["padding"] = self.padding
        return kwargs

    def _encode_batch(self, texts: list[Any]) -> tuple[list[Any], list[Any]]:
        """Encode a whole batch, returning ``(token_ids, attention_mask)`` with nulls kept.

        Null inputs never reach the tokenizer (it would reject them) and stay null in both
        outputs, so the row count is preserved.
        """
        present = [i for i, t in enumerate(texts) if t is not None]
        ids: list[Any] = [None] * len(texts)
        mask: list[Any] = [None] * len(texts)
        if not present:
            return ids, mask
        encoded = self._encode([texts[i] for i in present], **self._batch_kwargs())
        # A HuggingFace call returns a mapping (`BatchEncoding`); a simpler batched
        # callable may just return the list of token lists.
        mapping = encoded if hasattr(encoded, "keys") else None
        got_ids = mapping["input_ids"] if mapping is not None else encoded
        got_mask = (
            mapping["attention_mask"]
            if mapping is not None and "attention_mask" in mapping
            else None
        )
        for slot, i in enumerate(present):
            ids[i] = got_ids[slot]
            if got_mask is not None:
                mask[i] = got_mask[slot]
        return ids, mask

    def transform(self, ds: Dataset) -> Dataset:
        """Apply the tokenizer to the text column, writing token ids to `output_column`.

        Runs as a whole-batch `map_batches` UDF — one tokenizer call per Arrow batch on
        the batched path. Null inputs stay null, in the ids and in the mask.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import Tokenizer
                >>> ds = bt.from_pydict({"t": ["a b", "c"]})
                >>> Tokenizer("t", str.split, output_column="toks").fit_transform(ds).to_pydict()
                {'t': ['a b', 'c'], 'toks': [['a', 'b'], ['c']]}

        Args:
            ds: The dataset whose text column to tokenize.

        Returns:
            A new lazy `Dataset` with the token-list column (and mask) added or replaced.
        """
        self._require_fitted()
        column, output, mask_column = self.column, self.output_column, self.attention_mask_column
        batched, encode, encode_batch = self._batched, self._encode, self._encode_batch

        def _udf(batch: Any) -> Any:
            import pyarrow as pa

            texts = batch.column(column).to_pylist()
            if batched:
                ids, mask = encode_batch(texts)
            else:
                ids = [encode(t) if t is not None else None for t in texts]
                mask = []
            out = _set_column(batch, output, pa.array(ids))
            if mask_column is not None:
                out = _set_column(out, mask_column, pa.array(mask))
            return out

        keep_cols = list(ds.columns)
        out_cols = [*keep_cols]
        for extra in (output, mask_column):
            if extra is not None and extra not in out_cols:
                out_cols.append(extra)
        return ds.map_batches(_udf, output_columns=out_cols)
