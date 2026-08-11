"""Output types for the per-document text-quality string functions.

Split from `infer` on the same seam `media` and `sequence` were: that module answers "what
type does this arithmetic produce" from its operands, while these answer it from the function
name alone. Thirteen names across two output shapes is a lookup, and a lookup inline in
`infer` is what pushes that file past its size limit without adding any inference to it.

The functions themselves are the Gopher/C4/RefinedWeb corpus filters, evaluated per row in
`bc-expr::eval::str::quality`.
"""

from __future__ import annotations

import pyarrow as pa

__all__ = ["QUALITY_FLOAT_FNS", "QUALITY_INT_FNS", "quality_type"]

#: The measures that are a fraction or a mean. Every ratio is in ``[0, 1]``; `mean_word_length`
#: and `char_entropy` are unbounded above.
QUALITY_FLOAT_FNS = frozenset(
    {
        "mean_word_length", "symbol_ratio", "alpha_word_ratio", "bullet_line_ratio",
        "ellipsis_line_ratio", "duplicate_line_ratio", "duplicate_paragraph_ratio",
        "top_ngram_ratio", "duplicate_ngram_ratio", "char_entropy",
    }
)  # fmt: skip

#: The measures that are a count.
QUALITY_INT_FNS = frozenset({"word_count", "stopword_count"})


def quality_type(fn: str) -> pa.DataType | None:
    """The Arrow type a text-quality function produces, or ``None`` if it is not one.

    Args:
        fn: The engine function name the `StrFunc` node carries.

    Returns:
        Float64 for a ratio or mean, Int64 for a count, and ``None`` for any other name —
        which lets the caller fall through to the rest of the string-function table rather
        than having to ask twice.
    """
    if fn in QUALITY_FLOAT_FNS:
        return pa.float64()
    if fn in QUALITY_INT_FNS:
        return pa.int64()
    return None
