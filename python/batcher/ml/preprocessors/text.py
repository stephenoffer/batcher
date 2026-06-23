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
from batcher.ml.preprocessors.base import Preprocessor
from batcher.plan.expr_ir import array, col

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from batcher.api.dataset import Dataset

__all__ = ["Concatenator", "Tokenizer"]


class Concatenator(Preprocessor):
    """Stack numeric `columns` into a single list column ``output_column``.

    A stateless transform (``fit`` is a no-op): the output is ``array(col, ...)``, a
    native list column ready to become a tensor for training. The source columns are
    kept unless `drop` is set.

    Args:
        columns: the numeric columns to stack, in order.
        output_column: the name of the assembled list column.
        drop: drop the source columns from the output when True.
    """

    __slots__ = ("columns", "drop", "output_column")

    def __init__(
        self, columns: Sequence[str], *, output_column: str = "features", drop: bool = False
    ) -> None:
        self.columns = list(columns)
        if len(self.columns) < 1:
            raise PlanError("Concatenator requires at least one column")
        self.output_column = output_column
        self.drop = drop

    def transform(self, ds: Dataset) -> Dataset:
        self._require_fitted()
        out = ds.with_columns(**{self.output_column: array(*(col(c) for c in self.columns))})
        if self.drop:
            keep = [c for c in out.columns if c not in set(self.columns)]
            out = out.select(*keep)
        return out


class Tokenizer(Preprocessor):
    """Tokenize a text column with a user-provided tokenizer (a `map_batches` UDF).

    `tokenizer` is a callable ``str -> list`` or any object exposing ``.encode(str)``
    (e.g. a HuggingFace tokenizer). A stateless transform (``fit`` is a no-op).

    Args:
        column: the text column to tokenize.
        tokenizer: a ``str -> list`` callable or an object with ``.encode``.
        output_column: where to put the token lists (defaults to `column`).
    """

    __slots__ = ("_encode", "column", "output_column")

    def __init__(
        self,
        column: str,
        tokenizer: Callable[[str], list[Any]] | Any,
        *,
        output_column: str | None = None,
    ) -> None:
        self.column = column
        self.output_column = output_column or column
        encode = getattr(tokenizer, "encode", tokenizer)
        if not callable(encode):
            raise PlanError("Tokenizer needs a callable, or an object with .encode")
        self._encode = encode

    def transform(self, ds: Dataset) -> Dataset:
        self._require_fitted()
        column, output, encode = self.column, self.output_column, self._encode

        def _udf(batch: Any) -> Any:
            import pyarrow as pa

            texts = batch.column(column).to_pylist()
            tokens = pa.array([encode(t) if t is not None else None for t in texts])
            if output in batch.schema.names:
                idx = batch.schema.get_field_index(output)
                return batch.set_column(idx, output, tokens)
            return batch.append_column(output, tokens)

        keep_cols = list(ds.columns)
        out_cols = keep_cols if output in keep_cols else [*keep_cols, output]
        return ds.map_batches(_udf, output_columns=out_cols)
