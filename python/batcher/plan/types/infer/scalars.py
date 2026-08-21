"""Output types for the `str` and `dt` accessor functions, keyed by function name alone.

Both families answer from the function name without consulting the operand, which is what
separates them from `arithmetic` (whose rules need the operand types) and makes this module
a pure lookup. A name absent from every table returns ``None`` -- the sound fallback -- so a
newly added accessor reports an honest "not certain" rather than a guess.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.plan.types.text_quality import quality_type

__all__ = ["datefunc_type", "strfunc_type"]

# `str` accessor functions whose output type is certain.
_STR_BOOL = frozenset(
    {"contains", "starts_with", "ends_with", "like", "ilike", "regexp_matches", "json_extract_bool"}
)
_STR_INT = frozenset(
    {
        "len",
        "position",
        "regexp_count",
        "levenshtein",
        "ascii",
        "bit_length",
        "octet_length",
        "crc32",
        "hamming",
        "hash64",
        "xxhash64",
        "json_extract_int",
        # `json_array_length` counts a JSON array's elements. It reads as a `.json`
        # accessor rather than a string function, which is why it was missed while its
        # `json_extract_int` sibling two lines up was not.
        "json_array_length",
    }
)
_STR_FLOAT = frozenset({"json_extract_float", "jaccard_similarity"})

_STR_STR = frozenset(
    {
        # `json_type` names the root's JSON type (`object`/`array`/`number`/...).
        "json_type",
        "squad_normalize",
        "strip_html",
        "upper",
        "lower",
        "trim",
        "l_trim",
        "r_trim",
        "lpad",
        "rpad",
        "substr",
        "repeat",
        "replace",
        "regexp_replace",
        "regexp_replace_all",
        "regexp_extract",
        "initcap",
        "hex",
        "base64",
        "from_base64",
        "soundex",
        "md5",
        "sha1",
        "sha256",
        "hmac_sha256",
        "aes_encrypt",
        "aes_decrypt",
        "mask",
        "right",
        "substring_index",
        "overlay",
        "split_part",
        "json_extract_string",
        "reverse",
        "translate",
        "unhex",
        "url_encode",
        "url_decode",
        "regexp_escape",
        "parse_filename",
        "parse_dirname",
        "parse_dirpath",
        "to_binary",
        "from_binary",
        # Each of these renders its argument *as text* whatever the argument's own type --
        # a code point, a byte count, a JSON document, a media type, a radix spelling -- so
        # the result is String regardless of the operand. They were absent, which left
        # `Dataset.schema` reporting `null` for a provably-String column.
        "chr",
        "format_bytes",
        "format_bytes_si",
        "json_pretty",
        "json_structure",
        "mime_type",
        "to_base",
    }
)

# `str` accessor functions that split one document into many pieces -> List<String>.
_STR_STR_LIST = frozenset(
    {
        "chunk",
        "token_ngrams",  # one joined n-gram per window
        "split",
        "regexp_extract_all",  # every match of the pattern
        "regexp_split",
        "parse_path",  # the path's components
        # The two `.json` accessors that return a list of *text*: an object's keys in
        # source order, and an array's elements each rendered the way a leaf extract
        # renders one. Both were declaring `null`.
        "json_object_keys",
        "json_array_values",
    }
)

# `dt` accessor (`DateFunc`) output types. Every field-extraction fn yields Int64;
# these four are the exceptions. `last_day` names a day, so it yields a **date** for
# either input type -- as it does in DuckDB, Spark and Polars.
_DATE_STR = frozenset({"dayname", "monthname"})
_DATE_BOOL = frozenset({"is_leap_year"})
_DATE_DATE = frozenset({"last_day"})


def strfunc_type(fn: str) -> pa.DataType | None:
    """The Arrow type a `str` accessor function produces, or ``None`` if not certain."""
    # The per-document quality measures answer from the function name alone; their table
    # lives beside `media` and `sequence` rather than inline here.
    if (quality := quality_type(fn)) is not None:
        return quality
    if fn == "minhash":
        return pa.list_(pa.int64())  # the signature: one value per permutation
    if fn in _STR_STR_LIST:
        return pa.list_(pa.string())
    if fn in _STR_BOOL:
        return pa.bool_()
    if fn in _STR_INT:
        return pa.int64()
    if fn in _STR_FLOAT:
        return pa.float64()
    if fn in _STR_STR:
        return pa.string()
    return None


def datefunc_type(fn: str) -> pa.DataType | None:
    """The Arrow type a `dt` accessor function produces, or ``None`` if not certain."""
    if fn in _DATE_STR:
        return pa.string()
    if fn in _DATE_BOOL:
        return pa.bool_()
    if fn in _DATE_DATE:
        return pa.date32()
    from batcher.plan.expr_ir.fn_names import DATE_FNS

    # Every remaining date field-extraction fn (year/month/day/hour/epoch/...) is Int64.
    if fn in DATE_FNS:
        return pa.int64()
    return None


#: The temporal constructors that build a calendar **date**; the rest of the family
#: builds a microsecond timestamp. Both readings come from `bc_expr::eval_make_temporal`,
#: which collects a `Date32Array` for these two and a `TimestampMicrosecondArray` for the
#: four epoch scalings.
_MAKE_TEMPORAL_DATE = frozenset({"make_date", "from_unix_date"})


def make_temporal_type(fn: str) -> pa.DataType | None:
    """The Arrow type a temporal *constructor* produces, or ``None`` if not certain.

    The family had no rule at all, so `Dataset.schema` reported ``null`` for every
    `make_date`, `make_timestamp` and `from_unix_*` column — a declared type that no
    execution ever produces, which is worse than an unknown one: `available_schema` is
    what the device tier is held against, what a `write` validates against, and what a
    user reads. The values were always right; only the declaration was wrong.

    Args:
        fn: The constructor's `MakeTemporal` function name.

    Returns:
        The Arrow type, or ``None`` when `fn` is not one of the constructors.
    """
    from batcher.plan.expr_ir.fn_names import MAKE_TEMPORAL_FNS

    if fn in _MAKE_TEMPORAL_DATE:
        return pa.date32()
    if fn in MAKE_TEMPORAL_FNS:
        return pa.timestamp("us")
    return None
