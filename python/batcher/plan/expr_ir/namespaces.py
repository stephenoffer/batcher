"""Accessor namespaces (`.str`/`.dt`/`.list`/`.struct`/`.json`) and their nodes.

These namespace classes are returned by the corresponding `Expr` properties (via
deferred imports in `core.py`, to keep this module's top-level import of `Expr`
acyclic). They build the IR node classes defined here — none of which the `Expr`
base references directly. The parameterless families (string transforms, date
fields, list reductions) are **data, not code**: each is one row in a dispatch
table, with the accessor methods generated below.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from batcher.plan.expr_ir.core import Cast, Expr, _wrap
from batcher.plan.expr_ir.func_nodes import (
    DateFunc,
    DateOffset,
    DateTrunc,
    ListBinary,
    ListContains,
    ListFunc,
    ListGet,
    ListSlice,
    Strftime,
    StrFunc,
    Strptime,
    StructField,
)
from batcher.plan.expr_ir.nodes import ListJoin

# Offset-string units → (months, days, micros) contribution per unit count. `mo`
# must precede `m` in the regex so "mo" parses as months, not minutes.
_OFFSET_UNITS = {
    "y": (12, 0, 0),
    "mo": (1, 0, 0),
    "w": (0, 7, 0),
    "d": (0, 1, 0),
    "h": (0, 0, 3_600_000_000),
    "m": (0, 0, 60_000_000),
    "s": (0, 0, 1_000_000),
}
_OFFSET_RE = re.compile(r"(-?\d+)(mo|[ymwdhs])")


def parse_offset(by: str) -> tuple[int, int, int]:
    """Parse a Polars-style offset string (e.g. ``"1mo"``, ``"-3d"``, ``"1h30m"``)
    into ``(months, days, micros)``. Raises ``ValueError`` if `by` is malformed."""
    pos = 0
    months = days = micros = 0
    for match in _OFFSET_RE.finditer(by):
        if match.start() != pos:
            break
        pos = match.end()
        n = int(match.group(1))
        mo, d, us = _OFFSET_UNITS[match.group(2)]
        months += n * mo
        days += n * d
        micros += n * us
    if pos != len(by) or not by:
        raise ValueError(
            f"invalid offset {by!r}; use counts with units y/mo/w/d/h/m/s, e.g. '1mo15d'"
        )
    return months, days, micros


def _bind_accessors(
    ns: type,
    table: dict[str, str],
    build: Callable[[Expr, str], Expr],
    doc: Callable[[str], str],
) -> None:
    """Generate one no-argument accessor per `table` row and attach it to `ns`.

    Collapses the three parameterless accessor families (string transforms, date
    fields, list reductions) onto a single factory: each accessor wraps the
    namespace's expression (``self._e``) into ``build(self._e, tag)`` and carries
    the per-family ``doc(name)``. ``__name__``/``__qualname__``/``__doc__`` mirror
    the hand-written equivalents so introspection is unchanged.
    """
    for name, tag in table.items():

        def accessor(
            self: Any,
            _tag: str = tag,
            _build: Callable[[Expr, str], Expr] = build,
        ) -> Expr:
            return _build(self._e, _tag)

        accessor.__name__ = name
        accessor.__qualname__ = f"{ns.__name__}.{name}"
        accessor.__doc__ = doc(name)
        setattr(ns, name, accessor)


class _StrNamespace:
    """String functions: ``col("s").str.upper()``, ``.str.contains("x")``, …

    The parameterless string→string transforms are **data, not code**
    (``_STR_TRANSFORMS``: accessor name → ``bc-expr`` ``StrFunc`` tag) and are
    generated below — adding one is a single table entry. The functions that take
    arguments (search / slice / replace) and ``len`` (returns Int64) stay explicit.
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        self._e = e

    def len(self) -> StrFunc:
        """Number of characters (→ Int64)."""
        return StrFunc("len", self._e)

    def hash64(self) -> StrFunc:
        """Deterministic FNV-1a 64-bit hash of the string (→ Int64). Stable across
        partitions, runs, and machines — the building block for surrogate keys
        (``col(keys).str.hash64()``) and slowly-changing-dimension change detection.
        Null → null."""
        return StrFunc("hash64", self._e)

    def to_datetime(self, format: str) -> Strptime:
        """Parse the string into a Timestamp using a chrono/strftime `format`
        (e.g. ``"%Y-%m-%d %H:%M:%S"``). Values that do not match become NULL
        (DuckDB ``try_strptime``) — the safe-ingest spelling for dirty date
        columns. A date-only format parses at midnight. → Timestamp(us)."""
        return Strptime(self._e, format)

    def to_date(self, format: str = "%Y-%m-%d") -> Cast:
        """Parse the string into a Date using a chrono/strftime `format` (default
        ISO ``"%Y-%m-%d"``); unmatched values become NULL. → Date32."""
        return Cast(Strptime(self._e, format), "date", try_cast=True)

    def contains(self, pattern: str) -> StrFunc:
        return StrFunc("contains", self._e, pattern=pattern)

    def starts_with(self, pattern: str) -> StrFunc:
        return StrFunc("starts_with", self._e, pattern=pattern)

    def ends_with(self, pattern: str) -> StrFunc:
        return StrFunc("ends_with", self._e, pattern=pattern)

    def substr(self, start: int, length: int | None = None) -> StrFunc:
        return StrFunc("substr", self._e, start=start, length=length)

    def left(self, n: int) -> StrFunc:
        """The first ``n`` characters (SQL ``left``) — a 1-based ``substr``."""
        return StrFunc("substr", self._e, start=1, length=n)

    def repeat(self, n: int) -> StrFunc:
        """Repeat the string ``n`` times (``n`` ≤ 0 → empty)."""
        return StrFunc("repeat", self._e, start=n)

    def lpad(self, width: int, fill: str = " ") -> StrFunc:
        """Left-pad to ``width`` characters with ``fill`` (cycled); truncate if longer."""
        return StrFunc("lpad", self._e, start=width, pattern=fill)

    def rpad(self, width: int, fill: str = " ") -> StrFunc:
        """Right-pad to ``width`` characters with ``fill`` (cycled); truncate if longer."""
        return StrFunc("rpad", self._e, start=width, pattern=fill)

    def position(self, pattern: str) -> StrFunc:
        """1-based index of ``pattern`` in the string, or 0 if absent (→ Int64)."""
        return StrFunc("position", self._e, pattern=pattern)

    def right(self, n: int) -> StrFunc:
        """The last ``n`` characters (SQL ``right``)."""
        return StrFunc("right", self._e, start=n)

    def ascii(self) -> StrFunc:
        """Unicode codepoint of the first character, 0 if empty (→ Int64)."""
        return StrFunc("ascii", self._e)

    def split(self, delimiter: str) -> StrFunc:
        """Split on ``delimiter`` → a list of strings (chain with ``.list``)."""
        return StrFunc("split", self._e, pattern=delimiter)

    def regexp_matches(self, pattern: str) -> StrFunc:
        """True where the regex ``pattern`` matches anywhere (→ Bool)."""
        return StrFunc("regexp_matches", self._e, pattern=pattern)

    def like(self, pattern: str) -> StrFunc:
        """SQL ``LIKE``: anchored match where ``%`` matches any run of chars and
        ``_`` matches exactly one (→ Bool)."""
        return StrFunc("like", self._e, pattern=pattern)

    def ilike(self, pattern: str) -> StrFunc:
        """Case-insensitive SQL ``LIKE`` (→ Bool)."""
        return StrFunc("ilike", self._e, pattern=pattern)

    def regexp_replace(self, pattern: str, replacement: str) -> StrFunc:
        """Replace the first regex match with ``replacement`` (``$1`` backrefs)."""
        return StrFunc("regexp_replace", self._e, pattern=pattern, replacement=replacement)

    def regexp_extract(self, pattern: str, group: int = 0) -> StrFunc:
        """Extract capture ``group`` of the regex (0 = whole match; '' if none)."""
        return StrFunc("regexp_extract", self._e, pattern=pattern, start=group)

    def replace(self, pattern: str, replacement: str) -> StrFunc:
        return StrFunc("replace", self._e, pattern=pattern, replacement=replacement)

    def trim(self, chars: str | None = None) -> StrFunc:
        """Trim from both ends: any of `chars` if given, else whitespace
        (DuckDB ``trim``; Polars ``strip_chars``)."""
        return StrFunc("trim", self._e, pattern=chars)

    def normalize_whitespace(self) -> StrFunc:
        """Collapse every run of whitespace to a single space and trim the ends —
        the common text-cleanup step for messy free-text columns. Composes existing
        ops (``regexp_replace_all`` + ``trim``), no new IR."""
        return StrFunc("trim", self.regexp_replace_all(r"\s+", " "))

    def lstrip(self, chars: str | None = None) -> StrFunc:
        """Trim from the left: any of `chars` if given, else whitespace."""
        return StrFunc("l_trim", self._e, pattern=chars)

    def rstrip(self, chars: str | None = None) -> StrFunc:
        """Trim from the right: any of `chars` if given, else whitespace."""
        return StrFunc("r_trim", self._e, pattern=chars)

    def split_part(self, delimiter: str, n: int) -> StrFunc:
        """The ``n``-th field (1-based) after splitting on `delimiter`; ``''`` if
        ``n`` is out of range (DuckDB/Spark ``split_part``)."""
        return StrFunc("split_part", self._e, pattern=delimiter, start=n)

    def regexp_replace_all(self, pattern: str, replacement: str) -> StrFunc:
        """Replace every regex match of `pattern` with `replacement` (DuckDB
        ``regexp_replace(..., 'g')``; Polars ``replace_all``; ``$1`` backrefs)."""
        return StrFunc("regexp_replace_all", self._e, pattern=pattern, replacement=replacement)

    def initcap(self) -> StrFunc:
        """Capitalize the first letter of each word, lowercasing the rest."""
        return StrFunc("initcap", self._e)

    def octet_length(self) -> StrFunc:
        """Number of UTF-8 bytes in the string (→ Int64)."""
        return StrFunc("octet_length", self._e)

    def bit_length(self) -> StrFunc:
        """Number of bits in the string (bytes * 8) (→ Int64)."""
        return StrFunc("bit_length", self._e)

    def hex(self) -> StrFunc:
        """Uppercase hexadecimal of the UTF-8 bytes (→ Utf8)."""
        return StrFunc("hex", self._e)

    def base64(self) -> StrFunc:
        """Standard base64 encoding of the UTF-8 bytes (→ Utf8)."""
        return StrFunc("base64", self._e)

    def from_base64(self) -> StrFunc:
        """Decode standard base64 to a UTF-8 string; null if invalid (→ Utf8)."""
        return StrFunc("from_base64", self._e)

    def unhex(self) -> StrFunc:
        """Decode pairs of hex digits to a UTF-8 string; null if invalid (→ Utf8)."""
        return StrFunc("unhex", self._e)

    def translate(self, from_chars: str, to_chars: str) -> StrFunc:
        """Replace each char of ``from_chars`` with the char at the same index of
        ``to_chars``; chars in ``from_chars`` beyond ``to_chars``'s length are
        deleted; other chars pass through (SQL/DuckDB ``translate``)."""
        return StrFunc("translate", self._e, pattern=from_chars, replacement=to_chars)


# Parameterless string→string transforms: accessor name → engine `StrFunc` tag.
# (`trim`/`lstrip`/`rstrip` are explicit methods — they take an optional char set.)
_STR_TRANSFORMS = {
    "upper": "upper",
    "lower": "lower",
    "reverse": "reverse",
}


_bind_accessors(
    _StrNamespace,
    _STR_TRANSFORMS,
    lambda e, t: StrFunc(t, e),
    lambda n: f"Return the string with {n} applied.",
)


class _DtNamespace:
    """Date/time field extractions: ``col("d").dt.year()``, ``.dt.hour()``, …

    The available extractors are **data, not code**: each is one row in
    ``_DT_FIELDS`` (Python accessor name → ``bc-expr`` ``DateFunc`` wire tag) and
    the no-argument accessor is generated below. Adding a field extractor is a
    single table entry — the pattern that keeps the namespace maintainable as it
    grows to hundreds of functions.
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        self._e = e

    def truncate(self, unit: str) -> DateTrunc:
        """Truncate to the start of ``unit`` (year/month/day/hour/minute/second)."""
        return DateTrunc(self._e, unit)

    def is_leap_year(self) -> DateFunc:
        """True where the year is a leap year (→ Bool)."""
        return DateFunc("is_leap_year", self._e)

    def days_in_month(self) -> DateFunc:
        """Number of days in the month, 28 to 31 (→ Int64)."""
        return DateFunc("days_in_month", self._e)

    def iso_year(self) -> DateFunc:
        """ISO 8601 week-numbering year (may differ from the calendar year near
        January 1st) (→ Int64)."""
        return DateFunc("iso_year", self._e)

    def strftime(self, format: str) -> Strftime:
        """Format the date/time with a chrono/strftime `format` string, e.g.
        ``"%Y-%m-%d"`` → ``"2024-02-15"`` (DuckDB ``strftime``; Polars
        ``dt.strftime``). → Utf8."""
        return Strftime(self._e, format)

    def offset_by(self, by: str) -> DateOffset:
        """Shift each date/time by `by` (Polars syntax: ``"1y"``/``"2mo"``/``"3w"``/
        ``"4d"``/``"5h"``/``"6m"``/``"7s"``, combinable like ``"1mo15d"``, negatives
        allowed). Months/years are calendar-correct (end-of-month clamping); a
        sub-day offset on a date column raises. Type-preserving."""
        months, days, micros = parse_offset(by)
        return DateOffset(self._e, months, days, micros)


# Python accessor name → engine `DateFunc` wire tag (serde snake_case). Each maps
# to one Arrow `DatePart` and matches the same-named DuckDB function.
_DT_FIELDS = {
    "year": "year",
    "month": "month",
    "day": "day",
    "hour": "hour",
    "minute": "minute",
    "second": "second",
    "quarter": "quarter",
    "week": "week",  # ISO week 1–53
    "dayofweek": "day_of_week",  # Sunday = 0
    "dayofyear": "day_of_year",  # 1–366
    "epoch": "epoch",  # seconds since the Unix epoch (→ Int64)
    "dayname": "dayname",  # full weekday name e.g. "Monday" (→ Utf8)
    "monthname": "monthname",  # full month name e.g. "January" (→ Utf8)
    "isodow": "isodow",  # ISO day of week: Monday = 1 … Sunday = 7 (→ Int64)
    "century": "century",  # the century, e.g. 2021 → 21 (→ Int64)
    "decade": "decade",  # the decade, e.g. 2021 → 202 (→ Int64)
    "millennium": "millennium",  # the millennium, e.g. 2021 → 3 (→ Int64)
    "last_day": "last_day",  # last day of the month at 00:00:00 (→ Timestamp(us))
}


_bind_accessors(
    _DtNamespace,
    _DT_FIELDS,
    lambda e, t: DateFunc(t, e),
    lambda n: f"Extract the {n} field of a date/time column (→ Int64).",
)


class _StructNamespace:
    """Struct accessors: ``col("s").struct.field("x")``."""

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        self._e = e

    def field(self, name: str) -> StructField:
        """Extract the named field from the struct column."""
        return StructField(self._e, name)


class _JsonNamespace:
    """JSON accessors on a string column: ``col("j").json.extract_string("$.a.b")``."""

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        self._e = e

    def extract_string(self, path: str) -> StrFunc:
        """The string value at JSON ``path`` (e.g. ``$.a.b``); null if absent."""
        return StrFunc("json_extract_string", self._e, pattern=path)

    def extract_int(self, path: str) -> StrFunc:
        """The integer value at JSON ``path``; null if absent or non-integral. → Int64."""
        return StrFunc("json_extract_int", self._e, pattern=path)

    def extract_float(self, path: str) -> StrFunc:
        """The numeric value at JSON ``path`` as a float; null if absent or non-numeric.
        → Float64."""
        return StrFunc("json_extract_float", self._e, pattern=path)

    def extract_bool(self, path: str) -> StrFunc:
        """The boolean value at JSON ``path``; null if absent or non-boolean. → Boolean."""
        return StrFunc("json_extract_bool", self._e, pattern=path)


class _ListNamespace:
    """List/array reductions: ``col("a").list.len()``, ``.list.sum()``, …

    Generated from ``_LIST_FUNCS`` (accessor name → ``bc-expr`` ``ListFunc`` tag) —
    a single table entry adds a reduction. `get` carries an index, so it is explicit.
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        self._e = e

    def get(self, index: int) -> ListGet:
        """The element at ``index`` (null if out of range). Negative indices count
        from the end: ``get(-1)`` is the last element (Polars/Python indexing)."""
        return ListGet(self._e, index)

    def first(self) -> ListGet:
        """The first element of each list (null for an empty/null list)."""
        return ListGet(self._e, 0)

    def last(self) -> ListGet:
        """The last element of each list (null for an empty/null list)."""
        return ListGet(self._e, -1)

    def contains(self, value: int | float | bool | str) -> ListContains:
        """True where any element equals ``value`` (→ Bool)."""
        return ListContains(self._e, value)

    def slice(self, offset: int, length: int | None = None) -> ListSlice:
        """The 0-based sub-range ``[offset, offset+length)`` of each list."""
        return ListSlice(self._e, offset, length)

    def join(self, separator: str) -> ListJoin:
        """Concatenate the elements (cast to text, nulls skipped) with ``separator`` → text."""
        return ListJoin(self._e, separator)

    def flatten(self) -> ListFunc:
        """Concatenate a list-of-lists into one list per row, in order (DuckDB
        ``flatten``). Null inner lists are skipped; a null row stays null."""
        return ListFunc("flatten", self._e)

    def dot(self, other: Any) -> ListBinary:
        """Dot product with another vector column (or an ``array(...)`` literal),
        paired element-wise → Float64. The unnormalized similarity score."""
        return ListBinary("dot", self._e, _wrap(other))

    def cosine_similarity(self, other: Any) -> ListBinary:
        """Cosine similarity with another vector column, in ``[-1, 1]`` → Float64;
        null if either vector has zero magnitude. The standard embedding-similarity
        score for retrieval / RAG."""
        return ListBinary("cosine_similarity", self._e, _wrap(other))

    def l2_distance(self, other: Any) -> ListBinary:
        """Euclidean distance to another vector column → Float64 (vector search)."""
        return ListBinary("l2_distance", self._e, _wrap(other))


# Python accessor name → engine `ListFunc` wire tag.
_LIST_FUNCS = {
    "len": "len",
    "sum": "sum",
    "min": "min",
    "max": "max",
    "mean": "mean",
    "n_unique": "n_unique",
    "sort": "sort",  # → list
    "reverse": "reverse",  # → list
    "product": "product",
    "std": "std",
    "var": "var",
    "unique": "unique",  # → list
    "median": "median",
    "arg_min": "arg_min",  # index of min element (→ Int64)
    "arg_max": "arg_max",  # index of max element (→ Int64)
    "l2_norm": "l2_norm",  # Euclidean norm = sqrt(sum of squares) (-> Float64)
}


_bind_accessors(
    _ListNamespace,
    _LIST_FUNCS,
    lambda e, t: ListFunc(t, e),
    lambda n: f"Per-row {n} over each list value.",
)
