"""Removing evaluation data from a training corpus.

A benchmark score is only evidence if the model has not seen the answers. Test sets leak into
web crawls constantly — questions get quoted in blog posts, whole datasets get mirrored to
GitHub, and a paper's appendix reproduces its own examples — so a corpus assembled from the web
contains some of every public benchmark. The resulting score is not a lie anyone told; it is a
measurement of memorization reported as a measurement of ability.

Decontamination is the standard defence: drop the training documents that share a long enough
verbatim span with the evaluation set. `n` is the whole judgement. Too short and it deletes
ordinary English; at 13 tokens, the length the published pipelines converged on, an accidental
match is vanishingly unlikely and a quotation is caught.

This runs as a join rather than a scan-per-document. The eval set's n-grams become one table,
the training corpus's become another, and the overlap is an anti-join — so it scales the way
every other join here does instead of being quadratic in the corpus size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.stats._shared import require_columns

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["contamination_rate", "decontaminate"]

_GRAM = "__bt_gram"
_ROW = "__bt_row"


def _validate(n: int) -> None:
    """Reject a span short enough to match ordinary language."""
    if n < 1:
        raise PlanError(f"decontaminate: n must be at least 1, got {n}")


def _grams(ds: Dataset, column: str, n: int, *, with_row_id: bool) -> Dataset:
    """One row per (document, n-gram), optionally carrying the document's row id."""
    from batcher.plan.expr_ir.constructors import col
    from batcher.plan.functions.metrics.text._text import token_ngrams

    framed = ds.with_row_index(_ROW) if with_row_id else ds
    # The package-wide SQuAD normalization, so a requoted question with different casing or
    # punctuation is still recognized as the same span. Reused rather than re-spelled: a
    # decontaminator that tokenized differently from the grounding metrics would report
    # contamination the rest of the eval stack could not see.
    exploded = framed.select(
        *([_ROW] if with_row_id else []),
        **{_GRAM: token_ngrams(col(column), n)},
    ).explode(_GRAM)
    return exploded.drop_nulls(subset=[_GRAM])


def _contaminated_rows(ds: Dataset, column: str, against: Dataset, eval_column: str, n: int):
    """The row ids of documents sharing an n-gram with the evaluation set."""
    eval_grams = _grams(against, eval_column, n, with_row_id=False).distinct()
    train_grams = _grams(ds, column, n, with_row_id=True)
    return train_grams.join(eval_grams, on=_GRAM, how="semi").select(_ROW).distinct()


def decontaminate(
    ds: Dataset,
    column: str,
    against: Dataset,
    *,
    eval_column: str | None = None,
    n: int = 13,
) -> Dataset:
    """Drop the training documents that share a verbatim `n`-token span with an eval set.

    The defence against reporting memorization as ability. A corpus assembled from the web
    contains some of every public benchmark, and a model that has read the test set answers it
    correctly for the wrong reason.

    `n` is the judgement being made. At 13 tokens — where the published pipelines settled — an
    accidental match between unrelated documents is vanishingly unlikely while a quoted question
    is caught. Shorter spans start deleting ordinary English, which costs you real training data
    to remove contamination that was not there.

    Matching is on the same normalization the text metrics use (lowercased, punctuation and
    articles dropped), so a reformatted quotation still matches. It is verbatim overlap, so a
    paraphrase of a test question does not — this removes copies, not leakage in general.

    Args:
        ds: The training corpus.
        column: The training corpus's text column.
        against: The evaluation set to decontaminate against.
        eval_column: The eval set's text column. Defaults to `column`.
        n: The span length, in tokens, that counts as contamination.

    Returns:
        A new dataset holding the training documents with no shared span.

    Raises:
        PlanError: If `n` is less than 1.
        ColumnNotFoundError: If either text column is absent.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import decontaminate
            >>> train = bt.from_pydict(
            ...     {"text": ["what is the capital of france", "an unrelated document"]}
            ... )
            >>> evals = bt.from_pydict({"text": ["what is the capital of france"]})
            >>> clean = decontaminate(train, "text", evals, n=4)
            >>> clean.to_pydict()["text"]
            ['an unrelated document']
    """
    _validate(n)
    eval_column = eval_column or column
    _require_column(ds, column)
    _require_column(against, eval_column)
    hits = _contaminated_rows(ds, column, against, eval_column, n)
    return ds.with_row_index(_ROW).join(hits, on=_ROW, how="anti").drop(_ROW)


def contamination_rate(
    ds: Dataset,
    column: str,
    against: Dataset,
    *,
    eval_column: str | None = None,
    n: int = 13,
) -> float:
    """The fraction of training documents that share a verbatim span with the eval set.

    Measure before removing. The number tells you two different things depending on its size: a
    small rate is contamination to drop, and a large one usually means `n` is too short for
    your corpus and the filter is matching ordinary language. Check both ends before running
    `decontaminate` over a corpus you cannot rebuild.

    Args:
        ds: The training corpus.
        column: The training corpus's text column.
        against: The evaluation set to check against.
        eval_column: The eval set's text column. Defaults to `column`.
        n: The span length, in tokens, that counts as contamination.

    Returns:
        The fraction of training documents that are contaminated, in ``[0, 1]``.

    Raises:
        PlanError: If `n` is less than 1.
        ColumnNotFoundError: If either text column is absent.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import contamination_rate
            >>> train = bt.from_pydict(
            ...     {"text": ["what is the capital of france", "an unrelated document"]}
            ... )
            >>> evals = bt.from_pydict({"text": ["what is the capital of france"]})
            >>> contamination_rate(train, "text", evals, n=4)
            0.5
    """
    _validate(n)
    eval_column = eval_column or column
    _require_column(ds, column)
    _require_column(against, eval_column)
    total = ds.count()
    if total == 0:
        return 0.0
    return _contaminated_rows(ds, column, against, eval_column, n).count() / total


def _require_column(ds: Dataset, column: str) -> None:
    """Fail at the API edge, naming the columns that do exist."""
    require_columns(ds, column, hint="Pass an existing text column.")
