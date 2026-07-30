"""The scalar-function vocabulary — the documented home for `fn` discriminators.

`ir_tags.py` centralizes the *node* tags (the ``"e"`` discriminator). This module
is its sibling for the next level down: the ``fn`` string a function node carries
(``StrFunc(fn="contains")``, ``MathExpr(fn="sqrt")``). These strings mirror the Rust
``match`` arms in ``bc-expr``; keeping them named in one place turns a typo into a
clear `PlanError` at plan-build time instead of an opaque engine error, and gives
tooling/docs a single enumerable source of what the engine supports.

Two shapes, chosen by how the family grows:

* **Closed families** (a handful of stable operations) are `enum.StrEnum`s —
  ``MapFn``, ``ListBinaryFn``, ``ListSetFn``, ``Math2Fn``. The members read as code
  and a typo is an ``AttributeError``.
* **Open families** (string/math/date/list functions, which grow toward hundreds)
  are `frozenset`s — ``STR_FNS``, ``MATH_FNS``, ``DATE_FNS``, ``LIST_FNS``. A
  thousand-member ``Enum`` class would itself be the sprawl this codebase avoids; a
  named set is the scalable vocabulary, and adding a function is one new entry.

Every set/enum here is the *complete* vocabulary for its family (validated by the
test suite): the node base validates a node's ``fn`` against it at construction, so
the sets must stay exhaustive. Add the function's name here in the same change that
adds the namespace method and the Rust ``match`` arm.

The window-function sets live in `ir_tags` (the relational `Window` operator owns
them); they are re-exported here so this module is the one-stop view of the callable
vocabulary.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from batcher.plan.ir_tags import WINDOW_AGGREGATES, WINDOW_FUNCS, WINDOW_RANKING, WINDOW_VALUE

__all__ = [
    "DATE_FNS",
    "KEYED_STR_FNS",
    "LIST_FNS",
    "MAKE_TEMPORAL_FNS",
    "MATH_FNS",
    "STR_FNS",
    "WINDOW_AGGREGATES",
    "WINDOW_FUNCS",
    "WINDOW_RANKING",
    "WINDOW_VALUE",
    "ListBinaryFn",
    "ListSetFn",
    "MapFn",
    "Math2Fn",
]


class MapFn(StrEnum):
    """Map-column accessors carried by `MapFunc` (the `.map` namespace)."""

    MAP_KEYS = "map_keys"
    MAP_VALUES = "map_values"
    ELEMENT_AT = "element_at"


class ListBinaryFn(StrEnum):
    """Pairwise reductions over two List columns carried by `ListBinary`."""

    DOT = "dot"
    COSINE_SIMILARITY = "cosine_similarity"
    L2_DISTANCE = "l2_distance"
    L1_DISTANCE = "l1_distance"
    HAMMING = "hamming"
    JACCARD = "jaccard"
    MULTISET_OVERLAP = "multiset_overlap"


class ListSetFn(StrEnum):
    """Set operations between two List columns carried by `ListSet`."""

    ARRAY_INTERSECT = "array_intersect"
    ARRAY_EXCEPT = "array_except"
    ARRAY_UNION = "array_union"
    # `array_concat` rides this family because its shape is the same (two lists in, one
    # list out), but it is not a set operation: it appends without deduplicating.
    ARRAY_CONCAT = "array_concat"


class ListZipFn(StrEnum):
    """Element-wise arithmetic between two List columns carried by `ListZip`."""

    LIST_ADD = "list_add"
    LIST_SUBTRACT = "list_subtract"
    LIST_MULTIPLY = "list_multiply"


class Math2Fn(StrEnum):
    """Two-argument math functions carried by `Math2Expr` (→ Float64)."""

    POW = "pow"
    ATAN2 = "atan2"
    HYPOT = "hypot"
    GCD = "gcd"
    LCM = "lcm"
    ROUND = "round"  # round(x, digits)
    NEXT_AFTER = "next_after"


# --- Open families: named, exhaustive vocabularies (one entry per function) ------

STR_FNS: Final[frozenset[str]] = frozenset(
    {
        "aes_decrypt", "aes_encrypt", "ascii", "base64", "bit_length", "chunk", "contains",
        "compress", "crc32", "decompress", "ends_with", "from_base64", "hash64", "hex",
        "hmac_sha256", "ilike",
        "initcap", "json_array_length", "json_array_values", "json_exists",
        "json_extract_bool", "json_extract_float", "json_extract_int",
        "json_object_keys", "json_type", "json_value", "json_contains", "json_pretty",
        "json_structure", "chr", "to_base", "format_bytes", "format_bytes_si",
        "damerau_levenshtein", "jaro_similarity", "jaro_winkler_similarity",
        "json_extract_string", "l_trim",
        "len", "levenshtein", "like", "lower",
        "lpad", "mask", "md5", "minhash", "octet_length", "overlay", "position", "r_trim",
        "regexp_count", "regexp_extract", "regexp_extract_all", "regexp_matches",
        "regexp_replace", "regexp_replace_all", "regexp_split", "repeat", "replace",
        "reverse",
        "right", "rpad", "sha1", "sha256", "soundex", "split", "split_part",
        "starts_with", "strip_html", "substr", "substring_index", "to_case",
        "token_ngrams", "translate",
        "trim", "unhex", "upper", "xxhash64",
        "from_binary", "hamming", "jaccard_similarity", "parse_dirname", "parse_dirpath",
        "parse_filename", "parse_path", "regexp_escape", "to_binary", "url_decode",
        "url_encode",
    }
)  # fmt: skip

KEYED_STR_FNS: Final[frozenset[str]] = frozenset({"aes_decrypt", "aes_encrypt", "hmac_sha256"})
"""The `STR_FNS` members whose ``pattern`` slot carries secret key material.

Named here — rather than inline in the two places that care — because both the
plan-build-time key validation (`plan.functions.security`) and the `StrFunc.__repr__`
redaction must agree on the set exactly. A function added to one and not the other
would leak a key into a traceback."""

DATE_FNS: Final[frozenset[str]] = frozenset(
    {
        "century", "day", "day_of_week", "day_of_year", "dayname", "days_in_month",
        "decade", "epoch", "hour", "is_leap_year", "iso_year", "isodow", "last_day",
        "millennium", "minute", "month", "monthname", "quarter", "second", "week",
        "year",
    }
)  # fmt: skip

MAKE_TEMPORAL_FNS: Final[frozenset[str]] = frozenset(
    {
        "make_date", "make_timestamp", "from_unix_seconds", "from_unix_millis",
        "from_unix_micros", "from_unix_nanos", "from_unix_date",
    }
)  # fmt: skip
"""Temporal *constructors* carried by `MakeTemporal` — the inverse of `DATE_FNS`.

The epoch conversions are a family rather than one `from_epoch` node because the unit
is not a value the engine can infer: an Int64 column of epoch counts carries no record
of whether it counts seconds or nanoseconds, so the plan has to say."""

LIST_FNS: Final[frozenset[str]] = frozenset(
    {
        "arg_max", "arg_min", "arg_sort", "cum_sum", "diff", "flatten", "l1_norm", "l2_norm", "len",
        "max", "max_abs", "mean", "median", "min", "n_unique", "normalize", "product",
        "reverse", "softmax", "sort", "std", "sum", "unique", "var",
    }
)  # fmt: skip

MATH_FNS: Final[frozenset[str]] = frozenset(
    {
        "abs", "acos", "asin", "atan", "bit_count", "cbrt", "ceil", "cos", "cosh",
        "cot", "degrees", "exp", "factorial", "floor", "ln", "log10", "log2",
        "radians", "round", "sign", "sin", "sinh", "sqrt", "tan", "tanh", "trunc",
        "csc", "even", "gamma", "lgamma", "rint", "sec",
    }
)  # fmt: skip
