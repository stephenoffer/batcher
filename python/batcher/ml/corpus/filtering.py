"""Heuristic quality filtering for a web-scale text corpus.

The published pretraining pipelines all begin the same way: throw away the documents that are
obviously not prose. Navigation bars, link farms, tables of numbers, truncated boilerplate, and
pages that are mostly punctuation make up a large fraction of scraped text, and none of it
teaches a model anything about language. The filters below are the well-known heuristics from
that line of work — a length floor, a word-shape check, character-ratio caps on punctuation,
digits and non-ASCII, and a terminal-punctuation requirement.

They are **heuristics on surface statistics**, not a quality model. Each one throws away good
documents along with bad ones, and the right thresholds depend on the corpus: a code dataset is
mostly symbols, a financial one mostly digits, and applying prose thresholds to either deletes
it. Run `quality_report` on a sample first and look at what each rule would remove before
turning it on.

Everything here is an expression over existing string primitives, so filtering a hundred
million documents is one scan with no per-row Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.plan.expr_ir.core import Expr

__all__ = ["QualityThresholds", "quality_filter", "quality_report"]


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """The bounds a document must satisfy to survive `quality_filter`.

    The defaults are prose thresholds, in the range the published web-corpus pipelines use.
    They are wrong for code, for tabular text, and for languages that do not put spaces between
    words — read `quality_report` on a sample of your own corpus before adopting them.

    Examples:
        .. doctest::

            >>> from batcher.ml import QualityThresholds
            >>> QualityThresholds().min_words
            50
            >>> QualityThresholds(min_words=10).min_words
            10
    """

    #: Fewest words a document may have. Below this it is a snippet, not a document.
    min_words: int = 50
    #: Most words a document may have. `None` for no ceiling.
    max_words: int | None = None
    #: Mean word length must be at least this. A page of one- and two-letter tokens is
    #: navigation, markup residue, or a character-level artifact rather than prose.
    min_mean_word_length: float = 3.0
    #: Mean word length must be at most this. Above it the text is usually base64, a hash
    #: dump, or a language the whitespace split cannot tokenize.
    max_mean_word_length: float = 10.0
    #: Highest share of *characters* that may be punctuation. Ordinary prose sits near 0.04;
    #: a page of ``!!!`` or ``...`` boilerplate runs several times that.
    max_punctuation_ratio: float = 0.15
    #: Highest share of characters that may be digits. A price list or a table of figures is
    #: not prose, and teaches a language model very little.
    max_digit_ratio: float = 0.2
    #: Highest share of characters that may be non-ASCII. Useful as a crude language gate on
    #: a corpus meant to be English; raise it to 1.0 for anything multilingual, where the
    #: default would delete most of the corpus.
    max_non_ascii_ratio: float = 1.0
    #: Require the document to end in terminal punctuation. Navigation, headings, and
    #: truncated boilerplate usually do not; a real document usually does.
    require_terminal_punctuation: bool = True

    def __post_init__(self) -> None:
        """Reject bounds that cannot admit any document."""
        if self.min_words < 0:
            raise PlanError(
                f"QualityThresholds: min_words must not be negative, got {self.min_words}"
            )
        if self.max_words is not None and self.max_words < self.min_words:
            raise PlanError(
                f"QualityThresholds: max_words ({self.max_words}) must not be below "
                f"min_words ({self.min_words})"
            )
        if self.max_mean_word_length < self.min_mean_word_length:
            raise PlanError(
                f"QualityThresholds: max_mean_word_length ({self.max_mean_word_length}) must "
                f"not be below min_mean_word_length ({self.min_mean_word_length})"
            )
        for name in ("max_punctuation_ratio", "max_digit_ratio", "max_non_ascii_ratio"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise PlanError(f"QualityThresholds: {name} must be in [0, 1], got {value}")


def _rules(column: str, thresholds: QualityThresholds) -> dict[str, Expr]:
    """One boolean expression per rule, keyed by the name a report shows.

    Each is written as "the document **passes**", so a report reads as a keep rate and the
    filter is the conjunction. Every rule is null-safe: a null document fails everything, which
    is what a corpus filter should do with a row it cannot read.
    """
    from batcher.plan.expr_ir.constructors import col, lit

    text = col(column)
    words = text.str.word_count()
    rules: dict[str, Expr] = {
        "min_words": words >= lit(thresholds.min_words),
        "mean_word_length": (text.str.avg_word_length() >= lit(thresholds.min_mean_word_length))
        & (text.str.avg_word_length() <= lit(thresholds.max_mean_word_length)),
        "punctuation_ratio": text.str.punctuation_ratio() <= lit(thresholds.max_punctuation_ratio),
        "digit_ratio": text.str.digit_ratio() <= lit(thresholds.max_digit_ratio),
    }
    if thresholds.max_non_ascii_ratio < 1.0:
        rules["non_ascii_ratio"] = text.str.non_ascii_ratio() <= lit(thresholds.max_non_ascii_ratio)
    if thresholds.require_terminal_punctuation:
        rules["terminal_punctuation"] = text.str.ends_with_punctuation()
    if thresholds.max_words is not None:
        rules["max_words"] = words <= lit(thresholds.max_words)
    return rules


def _all_of(rules: dict[str, Expr]) -> Expr:
    """The conjunction of every rule, treating a null as a failure."""
    from batcher.plan.expr_ir.constructors import lit

    combined: Expr | None = None
    for rule in rules.values():
        guarded = rule.fill_null(lit(False))
        combined = guarded if combined is None else combined & guarded
    return combined if combined is not None else lit(True)


def quality_filter(
    ds: Dataset,
    column: str,
    thresholds: QualityThresholds | None = None,
) -> Dataset:
    """Keep the documents that pass every heuristic quality rule.

    The first stage of a web-corpus pipeline: drop the pages that are obviously not prose
    before spending tokenizer, dedup, or training time on them. A document must pass all the
    rules in `thresholds`; a null document fails, because a row the filter cannot read is not a
    row to train on.

    This deletes data, and heuristics delete good data too. Run `quality_report` on a sample
    first — the rule that removes the most is usually the one whose threshold is wrong for your
    corpus rather than the one finding the most junk.

    Args:
        ds: The dataset to filter.
        column: The text column to judge.
        thresholds: The bounds to apply. Defaults to prose thresholds.

    Returns:
        A new dataset holding only the documents that passed.

    Raises:
        ColumnNotFoundError: If `column` is not in the dataset.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import QualityThresholds, quality_filter
            >>> docs = bt.from_pydict(
            ...     {
            ...         "text": [
            ...             "A real sentence about something, written out properly.",
            ...             "buy now!!!",
            ...             "1234 5678 9012 3456",
            ...         ]
            ...     }
            ... )
            >>> kept = quality_filter(docs, "text", QualityThresholds(min_words=3))
            >>> kept.to_pydict()["text"]
            ['A real sentence about something, written out properly.']
    """
    _require_column(ds, column)
    thresholds = thresholds or QualityThresholds()
    return ds.filter(_all_of(_rules(column, thresholds)))


def quality_report(
    ds: Dataset,
    column: str,
    thresholds: QualityThresholds | None = None,
) -> dict[str, float]:
    """The fraction of documents each quality rule would keep, and the fraction all of them do.

    Run this before `quality_filter`, on a sample. It is the difference between a filter you
    chose and one you inherited: a rule keeping 3% of your corpus is not finding junk, it is
    mis-tuned for the kind of text you have, and the only way to see that is per rule.

    The per-rule numbers are independent keep rates, so they do not sum to the ``"all"`` entry —
    a document usually fails several rules at once.

    Args:
        ds: The dataset to profile.
        column: The text column to judge.
        thresholds: The bounds to apply. Defaults to prose thresholds.

    Returns:
        A mapping of rule name to the fraction of documents it keeps, plus ``"all"`` for the
        fraction passing every rule.

    Raises:
        ColumnNotFoundError: If `column` is not in the dataset.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import QualityThresholds, quality_report
            >>> docs = bt.from_pydict({"text": ["one two three four five six seven."] * 3 + ["hi"]})
            >>> report = quality_report(docs, "text", QualityThresholds(min_words=5))
            >>> report["min_words"]
            0.75
    """
    from batcher.plan.expr_ir.constructors import lit

    _require_column(ds, column)
    thresholds = thresholds or QualityThresholds()
    rules = _rules(column, thresholds)
    aggregates = {
        name: rule.fill_null(lit(False)).cast("float64").mean() for name, rule in rules.items()
    }
    aggregates["all"] = _all_of(rules).cast("float64").mean()
    row = ds.agg(**aggregates).to_pydict()
    return {name: (row[name][0] if row[name][0] is not None else 0.0) for name in aggregates}


def _require_column(ds: Dataset, column: str) -> None:
    """Fail at the API edge, naming the columns that do exist."""
    if column not in ds.columns:
        from batcher._internal.errors import ColumnNotFoundError, unknown_message

        raise ColumnNotFoundError(
            unknown_message("column", column, ds.columns, hint="Pass an existing text column.")
        )
