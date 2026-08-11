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
    }
)
_STR_FLOAT = frozenset({"json_extract_float", "jaccard_similarity"})

_STR_STR = frozenset(
    {
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
