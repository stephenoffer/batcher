"""Output types for the `.seq` genomics expressions.

Split from `infer` for the same reason `media` was: that module answers "what type does
this arithmetic produce" from the operand types, while this answers it from a lookup keyed
on the function name alone. Twenty-two functions across six output shapes is a table, and a
table inline in `infer` is what pushes that file past its size limit without adding any
inference logic to it.

The shapes here mirror the per-function documentation on `bc_expr::SeqFunc`. They are a
control-plane *estimate* — `None` always remains a sound answer, meaning "fall back to
executing a zero-row query" — but getting them right is what lets a genomics projection be
planned without probing the engine.
"""

from __future__ import annotations

import pyarrow as pa

__all__ = ["seqfunc_type"]

# The sequence-in, sequence-out transforms: the only family whose output is text.
_STR_FNS = frozenset(
    {"complement", "reverse_complement", "transcribe", "back_transcribe", "translate"}
)

# The measures. Every one is a ratio, a physical quantity, or an expectation, so every one
# is Float64 rather than the input's type.
_FLOAT_FNS = frozenset(
    {
        "gc_content",
        "gc_skew",
        "melting_temp",
        "molecular_weight",
        "gravy",
        "isoelectric_point",
        "mean_quality",
        "expected_errors",
    }
)

# The counts.
_INT_FNS = frozenset({"max_homopolymer", "count_motif"})

# The sketches, which are lists of k-mers.
_KMER_FNS = frozenset({"kmers", "canonical_kmers", "minimizers"})

#: `base_counts` emits one Int64 per base class. The children are non-nullable and the row's
#: null bit carries absence, matching how the Rust kernel builds the struct — an inference
#: that marked them nullable would disagree with the array the engine actually returns.
_BASE_COUNTS_TYPE = pa.struct(
    [pa.field(name, pa.int64(), nullable=False) for name in ("a", "c", "g", "t", "u", "n", "other")]
)


def seqfunc_type(fn: str) -> pa.DataType | None:
    """The Arrow type a `.seq` accessor function produces, or ``None`` if not certain.

    Args:
        fn: The engine function name the expression carries, from ``SEQ_FNS``.

    Returns:
        The Arrow type, or ``None`` for an unrecognized name — the sound fallback, so a
        function added to the engine before this table never mislabels a column.
    """
    if fn in _STR_FNS:
        return pa.string()
    if fn in _FLOAT_FNS:
        return pa.float64()
    if fn in _INT_FNS:
        return pa.int64()
    if fn in _KMER_FNS:
        return pa.list_(pa.string())
    if fn == "find_motif":
        return pa.list_(pa.int64())  # 1-based match positions
    if fn == "phred_quality":
        return pa.list_(pa.int32())  # one Phred score per base
    if fn == "is_valid":
        return pa.bool_()
    if fn == "base_counts":
        return _BASE_COUNTS_TYPE
    return None
