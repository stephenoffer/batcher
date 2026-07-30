"""The `.str` accessor namespace.

`col("s").str.upper()`, `.str.md5()`, `.str.regexp_extract_all(...)`, … — each
method is a thin builder over a `bc-expr` `StrFunc` node. The parameterless
string→string transforms are generated from `_STR_TRANSFORMS` (data, not code).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from batcher._internal.errors import PlanError, require_int
from batcher.plan.expr_ir.compat.guidance import STR_UNSUPPORTED, accessor_attribute_error
from batcher.plan.expr_ir.constructors import lit, nullif
from batcher.plan.expr_ir.core import AggExpr, Cast, Expr, Lit
from batcher.plan.expr_ir.func_nodes import StrFunc, Strptime
from batcher.plan.expr_ir.namespaces._bind import _bind_accessors
from batcher.plan.expr_ir.nodes import ListJoin

# Where `str.chunk` may end a chunk; mirrors `bc-expr`'s `chunk::Boundary`.
_CHUNK_BOUNDARIES = frozenset({"char", "word", "sentence", "line"})

# Byte-stream codecs `str.compress`/`str.decompress` accept; mirrors `case`'s sibling
# `compress::CODECS` in `bc-expr`.
_COMPRESSION_CODECS = frozenset({"gzip", "zlib", "deflate", "zstd", "brotli", "lz4"})


def _require_codec(func: str, codec: str) -> str:
    """Return `codec` if it names a supported codec, else raise a `PlanError`.

    Shared by `compress` and `decompress` so the two cannot come to accept different
    codec sets, which would make a round trip fail on one side only.
    """
    if codec not in _COMPRESSION_CODECS:
        raise PlanError(
            f"str.{func}(): codec must be one of {sorted(_COMPRESSION_CODECS)}, got {codec!r}"
        )
    return codec


# Identifier styles `str.to_case` renders; mirrors `bc-expr`'s `case::STYLES`.
_CASE_STYLES = frozenset(
    {
        "snake",
        "upper_snake",
        "camel",
        "pascal",
        "kebab",
        "upper_kebab",
        "title",
        "sentence",
        "dot",
        "train",
    }
)


class _StrNamespace:
    """String functions on a text column: ``col("s").str.upper()``, ``.str.contains("x")``.

    The parameterless string→string transforms are **data, not code**
    (``_STR_TRANSFORMS``: accessor name → ``bc-expr`` ``StrFunc`` tag) and are
    generated below — adding one is a single table entry. The functions that take
    arguments (search / slice / replace) and ``len`` (returns Int64) stay explicit.

    Every method returns a new lazy :class:`Expr`; null inputs propagate to null
    outputs throughout.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"s": ["Hello", "world"]})
            >>> ds.select(bt.col("s").str.upper().alias("r")).to_pydict()
            {'r': ['HELLO', 'WORLD']}
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        """Wrap the parent :class:`Expr` so its `.str` methods can build on it."""
        self._e = e

    def __repr__(self) -> str:
        """Show the accessor and its parent, e.g. ``<.str accessor of col('name')>``."""
        return f"<.str accessor of {self._e!r}>"

    def __getattr__(self, name: str) -> Any:
        """Point a pandas/Polars ``.str`` idiom at its Batcher spelling.

        Only reached when normal lookup fails, so it never shadows a real ``.str``
        method. ``.str.pad``, ``.str.extractall``, ``.str.find`` come back naming
        ``.str.lpad``, ``.str.extract_all``, ``.str.position`` — see
        `batcher.plan.expr_ir.compat.guidance`.

        Args:
            name: The attribute name that was not found.

        Raises:
            AttributeError: Always, with guidance for `name`.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        raise accessor_attribute_error(self, "'.str' accessor", name, STR_UNSUPPORTED)

    def len(self) -> StrFunc:
        """Count the characters in the string (→ Int64).

        Counts Unicode characters, not bytes (see :meth:`octet_length`). Null → null.

        Returns:
            A new Int64 expression: the character count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["héllo", "hi"]})
                >>> ds.select(bt.col("s").str.len().alias("r")).to_pydict()
                {'r': [5, 2]}
        """
        return StrFunc("len", self._e)

    def hash64(self) -> StrFunc:
        """Compute a deterministic FNV-1a 64-bit hash of the string (→ Int64).

        Stable across partitions, runs, and machines — the basis for surrogate keys
        and slowly-changing-dimension change detection. Null → null.

        Returns:
            A new Int64 expression: the FNV-1a hash.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abc"]})
                >>> ds.select(bt.col("s").str.hash64().alias("r")).to_pydict()
                {'r': [-1792535898324117685]}
        """
        return StrFunc("hash64", self._e)

    def md5(self) -> StrFunc:
        """Compute the MD5 digest as lowercase hex (DuckDB ``md5``).

        Returns Utf8; null → null.

        Returns:
            A new Utf8 expression: the lowercase hex digest.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abc"]})
                >>> ds.select(bt.col("s").str.md5().alias("r")).to_pydict()
                {'r': ['900150983cd24fb0d6963f7d28e17f72']}
        """
        return StrFunc("md5", self._e)

    def sha1(self) -> StrFunc:
        """Compute the SHA-1 digest as lowercase hex (DuckDB ``sha1``).

        Returns Utf8; null → null.

        Returns:
            A new Utf8 expression: the lowercase hex digest.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abc"]})
                >>> ds.select(bt.col("s").str.sha1().alias("r")).to_pydict()
                {'r': ['a9993e364706816aba3e25717850c26c9cd0d89d']}
        """
        return StrFunc("sha1", self._e)

    def sha256(self) -> StrFunc:
        """Compute the SHA-256 digest as lowercase hex (DuckDB ``sha256``).

        Returns Utf8; null → null.

        Returns:
            A new Utf8 expression: the lowercase hex digest.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abc"]})
                >>> ds.select(bt.col("s").str.sha256().alias("r")).to_pydict()
                {'r': ['ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad']}
        """
        return StrFunc("sha256", self._e)

    def crc32(self) -> StrFunc:
        """Compute the CRC-32 (IEEE) checksum of the UTF-8 bytes (Spark ``crc32``).

        Returns Int64 — an integrity check, not a sharding hash (use
        :meth:`xxhash64`).

        Returns:
            A new Int64 expression: the CRC-32 checksum.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abc"]})
                >>> ds.select(bt.col("s").str.crc32().alias("r")).to_pydict()
                {'r': [891568578]}
        """
        return StrFunc("crc32", self._e)

    def xxhash64(self) -> StrFunc:
        """Compute a fast non-cryptographic 64-bit xxHash of the bytes (→ Int64).

        The standard bucketing/sharding hash, deterministic across machines. Null → null.

        Returns:
            A new Int64 expression: the xxHash value.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abc"]})
                >>> ds.select(bt.col("s").str.xxhash64().alias("r")).to_pydict()
                {'r': [4952883123889572249]}
        """
        return StrFunc("xxhash64", self._e)

    def to_datetime(self, format: str) -> Strptime:
        """Parse the string into a Timestamp using a chrono/strftime format.

        Values that do not match the format become NULL (DuckDB ``try_strptime``)
        — the safe-ingest spelling for dirty date columns. A date-only format
        parses at midnight. Returns Timestamp(us).

        Args:
            format: A chrono/strftime pattern, e.g. ``"%Y-%m-%d %H:%M:%S"``.

        Returns:
            A new Timestamp expression; unmatched values are null.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> d = bt.from_pydict({"s": ["2024-01-15 10:30:00", "bad"]})
                >>> d.select(
                ...     bt.col("s").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("t")
                ... ).to_pydict()
                {'t': [datetime.datetime(2024, 1, 15, 10, 30), None]}
        """
        return Strptime(self._e, format)

    def to_date(self, format: str = "%Y-%m-%d") -> Cast:
        """Parse the string into a Date using a chrono/strftime format.

        Unmatched values become NULL. Returns Date32.

        Args:
            format: A chrono/strftime pattern; defaults to ISO ``"%Y-%m-%d"``.

        Returns:
            A new Date32 expression; unmatched values are null.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["2024-02-15"]})
                >>> ds.select(bt.col("s").str.to_date().alias("r")).to_pydict()
                {'r': [datetime.date(2024, 2, 15)]}
        """
        return Cast(Strptime(self._e, format), "date", try_cast=True)

    def contains(self, pattern: str) -> StrFunc:
        """Test whether the string contains ``pattern`` as a substring (→ Bool).

        A plain substring search, not a regex (see :meth:`regexp_matches`).

        Args:
            pattern: The literal substring to search for.

        Returns:
            A new Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello", "world"]})
                >>> ds.select(bt.col("s").str.contains("ell").alias("r")).to_pydict()
                {'r': [True, False]}
        """
        return StrFunc("contains", self._e, pattern=pattern)

    def starts_with(self, pattern: str) -> StrFunc:
        """Test whether the string begins with the literal ``pattern`` (→ Bool).

        Args:
            pattern: The literal prefix to test for.

        Returns:
            A new Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello", "world"]})
                >>> ds.select(bt.col("s").str.starts_with("he").alias("r")).to_pydict()
                {'r': [True, False]}
        """
        return StrFunc("starts_with", self._e, pattern=pattern)

    def ends_with(self, pattern: str) -> StrFunc:
        """Test whether the string ends with the literal ``pattern`` (→ Bool).

        Args:
            pattern: The literal suffix to test for.

        Returns:
            A new Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello", "world"]})
                >>> ds.select(bt.col("s").str.ends_with("ld").alias("r")).to_pydict()
                {'r': [False, True]}
        """
        return StrFunc("ends_with", self._e, pattern=pattern)

    def substr(self, start: int, length: int | None = None) -> StrFunc:
        """Extract a substring of ``length`` characters from 1-based ``start``.

        When ``length`` is omitted, returns everything from ``start`` to the end
        (SQL ``substring``).

        Args:
            start: 1-based index of the first character to keep.
            length: Number of characters to take; all remaining if omitted.

        Returns:
            A new Utf8 expression: the extracted substring.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello"]})
                >>> ds.select(bt.col("s").str.substr(2, 3).alias("r")).to_pydict()
                {'r': ['ell']}
        """
        start = require_int(start, func="str.substr", arg="start")
        if length is not None:
            length = require_int(length, func="str.substr", arg="length")
        return StrFunc("substr", self._e, start=start, length=length)

    def left(self, n: int) -> StrFunc:
        """Take the first ``n`` characters (SQL ``left``); negative ``n`` drops the last ``|n|``.

        Matches DuckDB: ``left('abcdef', -2) = 'abcd'``. ``substr(1, n)`` covers the
        non-negative case; for ``n < 0`` the leading-keep is the mirror of ``right``'s
        negative case, so we reverse, drop the (now-leading) ``|n|`` with ``right``, and
        reverse back — reusing the engine's DuckDB-correct negative ``right``.

        Args:
            n: Number of leading characters to keep; negative drops the trailing ``|n|``.

        Returns:
            A new Utf8 expression: the leading characters.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello"]})
                >>> ds.select(bt.col("s").str.left(3).alias("r")).to_pydict()
                {'r': ['hel']}

                >>> ds.select(bt.col("s").str.left(-2).alias("r")).to_pydict()
                {'r': ['hel']}
        """
        n = require_int(n, func="str.left", arg="n")
        if n < 0:
            reversed_e = StrFunc("reverse", self._e)
            dropped = StrFunc("right", reversed_e, start=n)
            return StrFunc("reverse", dropped)
        return StrFunc("substr", self._e, start=1, length=n)

    def repeat(self, n: int) -> StrFunc:
        """Concatenate ``n`` copies of the string.

        Args:
            n: Repeat count; ``n`` ≤ 0 yields an empty string.

        Returns:
            A new Utf8 expression: the repeated string.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ab"]})
                >>> ds.select(bt.col("s").str.repeat(3).alias("r")).to_pydict()
                {'r': ['ababab']}
        """
        n = require_int(n, func="str.repeat", arg="n")
        return StrFunc("repeat", self._e, start=n)

    def lpad(self, width: int, fill: str = " ") -> StrFunc:
        """Left-pad the string to ``width`` characters, truncating if longer.

        Args:
            width: Target character width.
            fill: Pad string, cycled as needed; defaults to a space.

        Returns:
            A new Utf8 expression: the left-padded string.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ab"]})
                >>> ds.select(bt.col("s").str.lpad(5, "*").alias("r")).to_pydict()
                {'r': ['***ab']}
        """
        width = require_int(width, func="str.lpad", arg="width")
        return StrFunc("lpad", self._e, start=width, pattern=fill)

    def rpad(self, width: int, fill: str = " ") -> StrFunc:
        """Right-pad the string to ``width`` characters, truncating if longer.

        Args:
            width: Target character width.
            fill: Pad string, cycled as needed; defaults to a space.

        Returns:
            A new Utf8 expression: the right-padded string.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ab"]})
                >>> ds.select(bt.col("s").str.rpad(5, "*").alias("r")).to_pydict()
                {'r': ['ab***']}
        """
        width = require_int(width, func="str.rpad", arg="width")
        return StrFunc("rpad", self._e, start=width, pattern=fill)

    def zfill(self, width: int) -> StrFunc:
        """Left-pad with ``'0'`` to ``width`` characters — the numeric-string spelling of ``lpad``.

        A thin specialization of :meth:`lpad` with a ``'0'`` fill, matching the name
        Python/pandas/Polars users reach for when zero-padding fixed-width codes or ids.
        A string already ``width`` or longer is returned unchanged.

        Args:
            width: Target character width.

        Returns:
            A new Utf8 expression: the zero-padded string.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["7", "42", "100"]})
                >>> ds.select(bt.col("s").str.zfill(4).alias("r")).to_pydict()
                {'r': ['0007', '0042', '0100']}
        """
        width = require_int(width, func="str.zfill", arg="width")
        return StrFunc("lpad", self._e, start=width, pattern="0")

    def contains_any(self, patterns: Iterable[str]) -> Expr:
        """True where the string contains *any* of the literal ``patterns`` (an OR of substrings).

        Desugars to ``contains(p0) | contains(p1) | …`` over existing nodes, so it adds
        no IR and follows the same three-valued logic — a null input yields null. Use it
        as a fast keyword filter without writing a regex alternation.

        Args:
            patterns: Literal substrings to test for; matches if any is present.

        Returns:
            A new Boolean expression, true where at least one pattern is a substring.

        Raises:
            PlanError: If `patterns` is empty (no predicate to build).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["cat", "dog", "bird"]})
                >>> ds.select(bt.col("s").str.contains_any(["ca", "ir"]).alias("r")).to_pydict()
                {'r': [True, False, True]}
        """
        terms = [StrFunc("contains", self._e, pattern=p) for p in patterns]
        if not terms:
            raise PlanError("contains_any() requires at least one pattern")
        result: Expr = terms[0]
        for term in terms[1:]:
            result = result | term
        return result

    # --- Polars/pandas-compatible spellings (delegate to the SQL-named methods) -----

    def to_lowercase(self) -> StrFunc:
        """Lowercase the string — the Polars ``to_lowercase`` spelling of :meth:`lower`.

        Returns:
            A new Utf8 expression with every letter lowercased.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["Hello"]})
                >>> ds.select(r=bt.col("s").str.to_lowercase()).to_pydict()
                {'r': ['hello']}
        """
        return self.lower()

    def to_uppercase(self) -> StrFunc:
        """Uppercase the string — the Polars ``to_uppercase`` spelling of :meth:`upper`.

        Returns:
            A new Utf8 expression with every letter uppercased.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["Hello"]})
                >>> ds.select(r=bt.col("s").str.to_uppercase()).to_pydict()
                {'r': ['HELLO']}
        """
        return self.upper()

    def to_titlecase(self) -> StrFunc:
        """Title-case the string — the Polars ``to_titlecase`` spelling of :meth:`initcap`.

        Returns:
            A new Utf8 expression with the first letter of each word uppercased.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello world"]})
                >>> ds.select(r=bt.col("s").str.to_titlecase()).to_pydict()
                {'r': ['Hello World']}
        """
        return self.initcap()

    def pad_start(self, width: int, fill: str = " ") -> StrFunc:
        """Left-pad to ``width`` — the Polars ``pad_start`` spelling of :meth:`lpad`.

        Args:
            width: Target character width.
            fill: Pad character, defaulting to a space.

        Returns:
            A new Utf8 expression: the left-padded string.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ab"]})
                >>> ds.select(r=bt.col("s").str.pad_start(5, "*")).to_pydict()
                {'r': ['***ab']}
        """
        return self.lpad(require_int(width, func="str.pad_start", arg="width"), fill)

    def pad_end(self, width: int, fill: str = " ") -> StrFunc:
        """Right-pad to ``width`` — the Polars ``pad_end`` spelling of :meth:`rpad`.

        Args:
            width: Target character width.
            fill: Pad character, defaulting to a space.

        Returns:
            A new Utf8 expression: the right-padded string.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ab"]})
                >>> ds.select(r=bt.col("s").str.pad_end(5, "*")).to_pydict()
                {'r': ['ab***']}
        """
        return self.rpad(require_int(width, func="str.pad_end", arg="width"), fill)

    def count_matches(self, pattern: str) -> StrFunc:
        """Count regex matches — the Polars ``count_matches`` spelling of :meth:`regexp_count`.

        Args:
            pattern: The regular expression to count.

        Returns:
            An Int64 expression of the number of matches per row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a1b2c3"]})
                >>> ds.select(r=bt.col("s").str.count_matches("[0-9]")).to_pydict()
                {'r': [3]}
        """
        return self.regexp_count(pattern)

    def extract(self, pattern: str, group: int = 1) -> StrFunc:
        """Extract a regex capture group — Polars' ``extract`` (see :meth:`regexp_extract`).

        Args:
            pattern: The regular expression with capture groups.
            group: The 1-based capture group to return (``0`` is the whole match).

        Returns:
            A Utf8 expression of the captured text, or null if no match.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a1"]})
                >>> ds.select(r=bt.col("s").str.extract(r"([a-z])([0-9])", 2)).to_pydict()
                {'r': ['1']}
        """
        return self.regexp_extract(pattern, group)

    def extract_all(self, pattern: str) -> StrFunc:
        """All regex matches as a list — Polars' ``extract_all`` (see :meth:`regexp_extract_all`).

        Args:
            pattern: The regular expression to find all matches of.

        Returns:
            A ``List<Utf8>`` expression of every match per row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a1b2"]})
                >>> ds.select(r=bt.col("s").str.extract_all("[0-9]")).to_pydict()
                {'r': [['1', '2']]}
        """
        return self.regexp_extract_all(pattern)

    def replace_all(self, pattern: str, value: str) -> StrFunc:
        """Replace every regex match — Polars' ``replace_all`` (see :meth:`regexp_replace_all`).

        Args:
            pattern: The regular expression to replace.
            value: The replacement text.

        Returns:
            A Utf8 expression with every match replaced.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a1b2"]})
                >>> ds.select(r=bt.col("s").str.replace_all("[0-9]", "#")).to_pydict()
                {'r': ['a#b#']}
        """
        return self.regexp_replace_all(pattern, value)

    def len_chars(self) -> StrFunc:
        """Character length — the Polars ``len_chars`` spelling of :meth:`len`.

        Returns:
            An Int64 expression of the number of characters per row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["café"]})
                >>> ds.select(r=bt.col("s").str.len_chars()).to_pydict()
                {'r': [4]}
        """
        return self.len()

    def len_bytes(self) -> StrFunc:
        """UTF-8 byte length — the Polars ``len_bytes`` spelling of :meth:`octet_length`.

        Returns:
            An Int64 expression of the number of UTF-8 bytes per row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["café"]})
                >>> ds.select(r=bt.col("s").str.len_bytes()).to_pydict()
                {'r': [5]}
        """
        return self.octet_length()

    def strip_chars(self, chars: str | None = None) -> StrFunc:
        """Trim from both ends — the Polars ``strip_chars`` spelling of :meth:`trim`.

        Note the divergence from Polars: with ``chars=None`` this strips the ASCII **space**
        only, following SQL ``TRIM`` (and DuckDB), not the whole whitespace class. Tabs and
        newlines survive. Pass them explicitly — ``strip_chars(" \t\n")`` — when the input
        may carry them, which scraped and CSV text usually does.

        Args:
            chars: The characters to strip; the ASCII space when ``None``.

        Returns:
            A Utf8 expression with the leading and trailing characters removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["  ab  "]})
                >>> ds.select(r=bt.col("s").str.strip_chars()).to_pydict()
                {'r': ['ab']}
        """
        return self.trim(chars)

    def strip_chars_start(self, chars: str | None = None) -> StrFunc:
        """Trim from the left — the Polars ``strip_chars_start`` spelling of :meth:`lstrip`.

        Args:
            chars: The characters to strip; the ASCII space when ``None``.

        Returns:
            A Utf8 expression with the leading characters removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["  ab  "]})
                >>> ds.select(r=bt.col("s").str.strip_chars_start()).to_pydict()
                {'r': ['ab  ']}
        """
        return self.lstrip(chars)

    def strip_chars_end(self, chars: str | None = None) -> StrFunc:
        """Trim from the right — the Polars ``strip_chars_end`` spelling of :meth:`rstrip`.

        Args:
            chars: The characters to strip; the ASCII space when ``None``.

        Returns:
            A Utf8 expression with the trailing characters removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["  ab  "]})
                >>> ds.select(r=bt.col("s").str.strip_chars_end()).to_pydict()
                {'r': ['  ab']}
        """
        return self.rstrip(chars)

    def head(self, n: int) -> StrFunc:
        """First ``n`` characters — the Polars ``str.head`` spelling of :meth:`left`.

        Args:
            n: How many leading characters to keep.

        Returns:
            A Utf8 expression of the first ``n`` characters.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello"]})
                >>> ds.select(r=bt.col("s").str.head(3)).to_pydict()
                {'r': ['hel']}
        """
        return self.left(require_int(n, func="str.head", arg="n"))

    def tail(self, n: int) -> StrFunc:
        """Last ``n`` characters — the Polars ``str.tail`` spelling of :meth:`right`.

        Args:
            n: How many trailing characters to keep.

        Returns:
            A Utf8 expression of the last ``n`` characters.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello"]})
                >>> ds.select(r=bt.col("s").str.tail(3)).to_pydict()
                {'r': ['llo']}
        """
        return self.right(require_int(n, func="str.tail", arg="n"))

    def slice(self, offset: int, length: int | None = None) -> StrFunc:
        """0-based substring — the Polars ``str.slice`` spelling over :meth:`substr` (1-based).

        Args:
            offset: 0-based start index.
            length: Number of characters; to the end when ``None``.

        Returns:
            A Utf8 expression of the selected substring.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello"]})
                >>> ds.select(r=bt.col("s").str.slice(1, 3)).to_pydict()
                {'r': ['ell']}
        """
        offset = require_int(offset, func="str.slice", arg="offset")
        return self.substr(offset + 1, length)

    def ljust(self, width: int, fill: str = " ") -> StrFunc:
        """Left-justify to ``width`` (pad right) — pandas' ``str.ljust`` (see :meth:`rpad`).

        Args:
            width: Target character width.
            fill: Pad character, defaulting to a space.

        Returns:
            A Utf8 expression: the right-padded string.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ab"]})
                >>> ds.select(r=bt.col("s").str.ljust(5, "*")).to_pydict()
                {'r': ['ab***']}
        """
        return self.rpad(require_int(width, func="str.ljust", arg="width"), fill)

    def rjust(self, width: int, fill: str = " ") -> StrFunc:
        """Right-justify to ``width`` (pad left) — pandas' ``str.rjust`` (see :meth:`lpad`).

        Args:
            width: Target character width.
            fill: Pad character, defaulting to a space.

        Returns:
            A Utf8 expression: the left-padded string.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ab"]})
                >>> ds.select(r=bt.col("s").str.rjust(5, "*")).to_pydict()
                {'r': ['***ab']}
        """
        return self.lpad(require_int(width, func="str.rjust", arg="width"), fill)

    # --- text features (the cheap signals a text model or data check consumes) ------

    def word_count(self) -> StrFunc:
        """Count whitespace-separated words (→ Int64); an all-blank string counts 0.

        Returns:
            An Int64 expression of the number of words per row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello big  world", "hi"]})
                >>> ds.select(r=bt.col("s").str.word_count()).to_pydict()
                {'r': [3, 1]}
        """
        return self.regexp_count(r"\S+")

    def digit_count(self) -> StrFunc:
        """Count the digit characters in the string (→ Int64).

        Returns:
            An Int64 expression of the number of digits per row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a1b23", "xyz"]})
                >>> ds.select(r=bt.col("s").str.digit_count()).to_pydict()
                {'r': [3, 0]}
        """
        return self.regexp_count("[0-9]")

    def is_alpha(self) -> Expr:
        """True where the string is non-empty and all letters (pandas ``str.isalpha``).

        Returns:
            A Boolean expression, true for all-alphabetic strings.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abc", "ab1"]})
                >>> ds.select(r=bt.col("s").str.is_alpha()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches("^[A-Za-z]+$")

    def is_numeric(self) -> Expr:
        """True where the string is non-empty and all digits (pandas ``str.isnumeric``).

        Returns:
            A Boolean expression, true for all-digit strings.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["123", "12a"]})
                >>> ds.select(r=bt.col("s").str.is_numeric()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches("^[0-9]+$")

    def is_alnum(self) -> Expr:
        """True where the string is non-empty and all letters or digits (pandas ``str.isalnum``).

        Returns:
            A Boolean expression, true for all-alphanumeric strings.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ab12", "ab 12"]})
                >>> ds.select(r=bt.col("s").str.is_alnum()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches("^[A-Za-z0-9]+$")

    def is_space(self) -> Expr:
        """True where the string is non-empty and all whitespace (pandas ``str.isspace``).

        Returns:
            A Boolean expression, true for all-whitespace strings.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["   ", " a "]})
                >>> ds.select(r=bt.col("s").str.is_space()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches(r"^\s+$")

    def is_upper(self) -> Expr:
        """True where the string equals its uppercase form (pandas ``str.isupper``).

        Returns:
            A Boolean expression, true for uppercase strings.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ABC", "Abc"]})
                >>> ds.select(r=bt.col("s").str.is_upper()).to_pydict()
                {'r': [True, False]}
        """
        return self._e == self.upper()

    def is_lower(self) -> Expr:
        """True where the string equals its lowercase form (pandas ``str.islower``).

        Returns:
            A Boolean expression, true for lowercase strings.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abc", "Abc"]})
                >>> ds.select(r=bt.col("s").str.is_lower()).to_pydict()
                {'r': [True, False]}
        """
        return self._e == self.lower()

    def capitalize(self) -> Expr:
        """Uppercase the first character and lowercase the rest (pandas ``str.capitalize``).

        Unlike :meth:`initcap`, which title-cases *every* word, this only touches the
        first character of the whole string.

        Returns:
            A Utf8 expression of the capitalized string.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hELLO wORLD"]})
                >>> ds.select(r=bt.col("s").str.capitalize()).to_pydict()
                {'r': ['Hello world']}
        """
        from batcher.plan.functions.string import concat

        return concat(self.left(1).str.upper(), self.substr(2).str.lower())

    def remove_punctuation(self) -> StrFunc:
        """Drop every character that is not a word character or whitespace.

        The usual first step of text normalization, before tokenizing or hashing.

        Returns:
            A Utf8 expression with punctuation removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a,b! c."]})
                >>> ds.select(r=bt.col("s").str.remove_punctuation()).to_pydict()
                {'r': ['ab c']}
        """
        return self.regexp_replace_all(r"[^\w\s]", "")

    def contains_all(self, patterns: Iterable[str]) -> Expr:
        """True where the string contains *every* one of the literal `patterns`.

        The conjunctive counterpart to :meth:`contains_any` — an AND of substring tests,
        for "must mention all of these terms" filters.

        Args:
            patterns: Literal substrings that must all be present.

        Returns:
            A Boolean expression, true only when every pattern is a substring.

        Raises:
            PlanError: If `patterns` is empty.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["cat dog", "cat"]})
                >>> ds.select(r=bt.col("s").str.contains_all(["cat", "dog"])).to_pydict()
                {'r': [True, False]}
        """
        terms = [StrFunc("contains", self._e, pattern=p) for p in patterns]
        if not terms:
            raise PlanError("contains_all() requires at least one pattern")
        result: Expr = terms[0]
        for term in terms[1:]:
            result = result & term
        return result

    def count_char(self, char: str) -> StrFunc:
        """Count occurrences of a literal substring (→ Int64).

        The literal is regex-escaped, so punctuation is matched exactly rather than
        interpreted — the difference between counting ``"."`` and counting every
        character.

        Args:
            char: The literal substring to count.

        Returns:
            An Int64 expression of the number of occurrences.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a.b.c", "abc"]})
                >>> ds.select(r=bt.col("s").str.count_char(".")).to_pydict()
                {'r': [2, 0]}
        """
        return self.regexp_count(re.escape(char))

    # --- LLM training-data quality heuristics ---------------------------------------
    # The character-class ratios and shape statistics that Gopher / C4 / RefinedWeb-style
    # filters threshold on to drop boilerplate, markup dumps, and machine-generated text
    # from a pretraining corpus. Each is one regex count over the row, so a whole corpus
    # is scored in a single vectorized pass rather than a Python loop.

    def _char_ratio(self, pattern: str) -> Expr:
        """Fraction of characters matching `pattern`; null for an empty string."""

        return self.regexp_count(pattern) / nullif(self.len(), lit(0))

    def alpha_ratio(self) -> Expr:
        """Fraction of characters that are ASCII letters — the core text-density signal.

        Pretraining filters drop rows below roughly 0.6-0.7, which removes tables, logs,
        and ID dumps. A ratio in ``[0, 1]``; an empty string yields null rather than
        dividing by zero.

        Returns:
            A Float64 expression of the ratio.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["Hello 123"]})
                >>> ds.select(r=bt.col("s").str.alpha_ratio().round(3)).to_pydict()
                {'r': [0.556]}
        """
        return self._char_ratio(r"[A-Za-z]")

    def digit_ratio(self) -> Expr:
        """Fraction of characters that are digits — high values mark tables and logs.

        A ratio in ``[0, 1]``; an empty string yields null rather than dividing by zero.

        Returns:
            A Float64 expression of the ratio.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["Hello 123"]})
                >>> ds.select(r=bt.col("s").str.digit_ratio().round(3)).to_pydict()
                {'r': [0.333]}
        """
        return self._char_ratio(r"[0-9]")

    def uppercase_ratio(self) -> Expr:
        """Fraction of characters that are uppercase letters — high values mark shouting or headers.

        A ratio in ``[0, 1]``; an empty string yields null rather than dividing by zero.

        Returns:
            A Float64 expression of the ratio.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ABc de"]})
                >>> ds.select(r=bt.col("s").str.uppercase_ratio().round(3)).to_pydict()
                {'r': [0.333]}
        """
        return self._char_ratio(r"[A-Z]")

    def lowercase_ratio(self) -> Expr:
        """Fraction of characters that are lowercase letters.

        A ratio in ``[0, 1]``; an empty string yields null rather than dividing by zero.

        Returns:
            A Float64 expression of the ratio.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ABc de"]})
                >>> ds.select(r=bt.col("s").str.lowercase_ratio().round(3)).to_pydict()
                {'r': [0.5]}
        """
        return self._char_ratio(r"[a-z]")

    def punctuation_ratio(self) -> Expr:
        """Fraction of characters that are punctuation or symbols.

        A ratio in ``[0, 1]``; an empty string yields null rather than dividing by zero.

        Returns:
            A Float64 expression of the ratio.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hi!! ok"]})
                >>> ds.select(r=bt.col("s").str.punctuation_ratio().round(3)).to_pydict()
                {'r': [0.286]}
        """
        return self._char_ratio(r"[^\w\s]")

    def whitespace_ratio(self) -> Expr:
        """Fraction of characters that are whitespace — high values mark broken layout.

        A ratio in ``[0, 1]``; an empty string yields null rather than dividing by zero.

        Returns:
            A Float64 expression of the ratio.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a b c"]})
                >>> ds.select(r=bt.col("s").str.whitespace_ratio().round(3)).to_pydict()
                {'r': [0.4]}
        """
        return self._char_ratio(r"\s")

    def non_ascii_ratio(self) -> Expr:
        """Fraction of characters outside ASCII — a language and mojibake signal.

        A ratio in ``[0, 1]``; an empty string yields null rather than dividing by zero.

        Returns:
            A Float64 expression of the ratio.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["café x"]})
                >>> ds.select(r=bt.col("s").str.non_ascii_ratio().round(3)).to_pydict()
                {'r': [0.167]}
        """
        return self._char_ratio(r"[^\x00-\x7F]")

    def alnum_ratio(self) -> Expr:
        """Fraction of characters that are letters or digits.

        A ratio in ``[0, 1]``; an empty string yields null rather than dividing by zero.

        Returns:
            A Float64 expression of the ratio.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ab!12"]})
                >>> ds.select(r=bt.col("s").str.alnum_ratio().round(3)).to_pydict()
                {'r': [0.8]}
        """
        return self._char_ratio(r"[A-Za-z0-9]")

    def non_ascii_count(self) -> StrFunc:
        """Count characters outside the ASCII range (→ Int64).

        Returns:
            An Int64 expression of the non-ASCII character count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["caf\u00e9 na\u00efve"]})
                >>> ds.select(r=bt.col("s").str.non_ascii_count()).to_pydict()
                {'r': [2]}
        """
        return self.regexp_count(r"[^\x00-\x7F]")

    def line_count(self) -> Expr:
        r"""Number of lines, counting newline separators plus one (→ Int64).

        Returns:
            An Int64 expression of the line count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a\nb\nc"]})
                >>> ds.select(r=bt.col("s").str.line_count()).to_pydict()
                {'r': [3]}
        """
        return self.regexp_count("\n") + 1

    def mean_line_length(self) -> Expr:
        r"""Average characters per line — short means mark navigation and link dumps.

        Returns:
            A Float64 expression of the mean line length.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ab\ncdef"]})
                >>> ds.select(r=bt.col("s").str.mean_line_length().round(2)).to_pydict()
                {'r': [3.5]}
        """
        return self.len() / self.line_count()

    def avg_word_length(self) -> Expr:
        """Average letters per whitespace-separated word — a tokenizer-free text-shape signal.

        Gopher-style filters drop rows outside roughly 3-10, which catches both
        character-spam and concatenated-identifier dumps.

        Returns:
            A Float64 expression of the mean word length.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["alpha beta"]})
                >>> ds.select(r=bt.col("s").str.avg_word_length().round(2)).to_pydict()
                {'r': [4.5]}
        """

        return self.regexp_count("[A-Za-z]") / nullif(self.word_count(), lit(0))

    def url_count(self) -> StrFunc:
        """Count HTTP(S) URLs in the string (→ Int64) — a boilerplate/link-dump signal.

        Returns:
            An Int64 expression of the URL count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["see http://a.com and https://b.io"]})
                >>> ds.select(r=bt.col("s").str.url_count()).to_pydict()
                {'r': [2]}
        """
        return self.regexp_count(r"https?://\S+")

    def email_count(self) -> StrFunc:
        """Count email addresses in the string (→ Int64) — a PII and scrape-noise signal.

        Returns:
            An Int64 expression of the email count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a@b.com and c@d.org"]})
                >>> ds.select(r=bt.col("s").str.email_count()).to_pydict()
                {'r': [2]}
        """
        return self.regexp_count(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

    # --- corpus cleaning and detection ----------------------------------------------

    _URL_RE = r"https?://\S*[^\s.,;:!?)\]}]"
    r"""A URL run, stopping before trailing sentence punctuation.

    A plain ``\S+`` swallowed the period in "read https://a.example." and returned a URL
    with a trailing dot, which does not resolve. Ending the match on a character that
    cannot close a URL keeps "https://a.example/x?q=1" whole while dropping the
    punctuation that belongs to the sentence."""
    _EMAIL_RE = r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
    _NON_ASCII_RE = r"[^\x00-\x7F]"

    def remove_urls(self) -> StrFunc:
        """Strip HTTP(S) URLs from the text — the first step of web-corpus cleaning.

        Returns:
            A Utf8 expression with URLs removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["see http://a.com now"]})
                >>> ds.select(r=bt.col("s").str.remove_urls()).to_pydict()
                {'r': ['see  now']}
        """
        return self.regexp_replace_all(self._URL_RE, "")

    def remove_emails(self) -> StrFunc:
        """Strip email addresses from the text — a cheap PII scrub before training.

        Returns:
            A Utf8 expression with email addresses removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["mail me a@b.com"]})
                >>> ds.select(r=bt.col("s").str.remove_emails()).to_pydict()
                {'r': ['mail me ']}
        """
        return self.regexp_replace_all(self._EMAIL_RE, "")

    def remove_non_ascii(self) -> StrFunc:
        """Drop every character outside ASCII — the blunt mojibake/emoji scrub.

        Returns:
            A Utf8 expression containing only ASCII characters.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["café x"]})
                >>> ds.select(r=bt.col("s").str.remove_non_ascii()).to_pydict()
                {'r': ['caf x']}
        """
        return self.regexp_replace_all(self._NON_ASCII_RE, "")

    def remove_digits(self) -> StrFunc:
        """Drop every digit character — used to normalize IDs out of near-duplicate keys.

        Returns:
            A Utf8 expression with digits removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abc 123"]})
                >>> ds.select(r=bt.col("s").str.remove_digits()).to_pydict()
                {'r': ['abc ']}
        """
        return self.regexp_replace_all("[0-9]", "")

    def truncate_chars(self, n: int) -> StrFunc:
        """Keep at most the first `n` characters — a hard context-window guard.

        Args:
            n: Maximum characters to keep.

        Returns:
            A Utf8 expression truncated to `n` characters.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abcdefgh"]})
                >>> ds.select(r=bt.col("s").str.truncate_chars(3)).to_pydict()
                {'r': ['abc']}
        """
        return self.left(require_int(n, func="str.truncate_chars", arg="n"))

    def truncate_words(self, n: int) -> StrFunc:
        """Keep at most the first `n` whitespace-separated words, without splitting one.

        Prompt and chunk builders need a budget that never cuts mid-token; this trims on
        a word boundary instead.

        Args:
            n: Maximum words to keep (must be >= 1).

        Returns:
            A Utf8 expression truncated to `n` words.

        Raises:
            PlanError: If `n` < 1.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["one two three four"]})
                >>> ds.select(r=bt.col("s").str.truncate_words(2)).to_pydict()
                {'r': ['one two']}
        """
        n = require_int(n, func="str.truncate_words", arg="n", minimum=1)
        return self.regexp_extract(r"^(?:\S+\s+){0," + str(n - 1) + r"}\S+", 0)

    def has_url(self) -> StrFunc:
        """True where the text contains an HTTP(S) URL.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["see http://a.com", "plain"]})
                >>> ds.select(r=bt.col("s").str.has_url()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches(self._URL_RE)

    def has_email(self) -> StrFunc:
        """True where the text contains an email address — a PII pre-filter.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a@b.com", "plain"]})
                >>> ds.select(r=bt.col("s").str.has_email()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches(self._EMAIL_RE)

    def has_non_ascii(self) -> StrFunc:
        """True where the text contains any character outside ASCII.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["café", "plain"]})
                >>> ds.select(r=bt.col("s").str.has_non_ascii()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches(self._NON_ASCII_RE)

    def has_digits(self) -> StrFunc:
        """True where the text contains at least one digit.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a1", "ab"]})
                >>> ds.select(r=bt.col("s").str.has_digits()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches("[0-9]")

    def is_blank(self) -> StrFunc:
        """True where the text is empty or only whitespace — the empty-document filter.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["   ", "a"]})
                >>> ds.select(r=bt.col("s").str.is_blank()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches(r"^\s*$")

    def looks_like_json(self) -> StrFunc:
        """True where the text is shaped like a JSON object or array (a cheap pre-check).

        Tests only the outer delimiters, so it is a fast filter to run before the real
        parse — the shape check an LLM structured-output pipeline needs before decoding.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ['{"a": 1}', "not json"]})
                >>> ds.select(r=bt.col("s").str.looks_like_json()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches(r"^\s*[\{\[].*[\}\]]\s*$")

    def estimate_tokens(self, chars_per_token: float = 4.0) -> Expr:
        """Approximate LLM token count as ``len / chars_per_token`` (→ Int64).

        The standard tokenizer-free estimate (~4 characters per token for English GPT-style
        vocabularies). Use it to budget context windows or batch by cost without paying to
        run a real tokenizer over the corpus.

        Args:
            chars_per_token: Characters per token to assume.

        Returns:
            An Int64 expression of the estimated token count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abcdefgh"]})
                >>> ds.select(r=bt.col("s").str.estimate_tokens()).to_pydict()
                {'r': [2]}
        """

        return (self.len() / Lit(chars_per_token)).cast("int64")

    def fits_token_budget(self, budget: int, chars_per_token: float = 4.0) -> Expr:
        """True where :meth:`estimate_tokens` is within `budget` — the context-window filter.

        Args:
            budget: The maximum estimated tokens allowed.
            chars_per_token: Characters per token to assume.

        Returns:
            A Boolean expression, true for rows that fit.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abcd", "abcdefghijkl"]})
                >>> ds.select(r=bt.col("s").str.fits_token_budget(2)).to_pydict()
                {'r': [True, False]}
        """

        budget = require_int(budget, func="str.fits_token_budget", arg="budget")
        return self.estimate_tokens(chars_per_token) <= Lit(budget)

    def sentence_count(self) -> StrFunc:
        """Count sentence-ending punctuation marks (→ Int64) — a document-shape signal.

        Returns:
            An Int64 expression of the sentence count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["One. Two! Three?"]})
                >>> ds.select(r=bt.col("s").str.sentence_count()).to_pydict()
                {'r': [3]}
        """
        return self.regexp_count(r"[.!?]")

    def has_html(self) -> StrFunc:
        """True where the text still contains HTML tags — the un-stripped-markup check.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["<p>hi</p>", "plain"]})
                >>> ds.select(r=bt.col("s").str.has_html()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches("<[^>]+>")

    def remove_html_tags(self) -> StrFunc:
        """Delete HTML tags, keeping their text content.

        The blunt tag-stripper; :meth:`strip_html` is the smarter one that also drops
        ``<script>``/``<style>`` bodies and decodes entities.

        Returns:
            A Utf8 expression with tags removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["<p>hi</p> there"]})
                >>> ds.select(r=bt.col("s").str.remove_html_tags()).to_pydict()
                {'r': ['hi there']}
        """
        return self.regexp_replace_all("<[^>]+>", "")

    def is_ascii_only(self) -> Expr:
        """True where every character is ASCII — the inverse of :meth:`has_non_ascii`.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["plain", "café"]})
                >>> ds.select(r=bt.col("s").str.is_ascii_only()).to_pydict()
                {'r': [True, False]}
        """
        return ~self.has_non_ascii()

    def starts_with_bullet(self) -> StrFunc:
        """True where the line opens with a list bullet (``-``, ``*``, or ``+``).

        Gopher-style filters drop documents whose lines are mostly bullets, which is how
        navigation menus and link lists are removed from a corpus.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["- item", "prose"]})
                >>> ds.select(r=bt.col("s").str.starts_with_bullet()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches(r"^\s*[-*+]\s")

    _PHONE_RE = r"\(?\d{3}\)?[-.\s]\s?\d{3}[-.\s]\d{4}"

    def has_phone(self) -> StrFunc:
        """True where the text contains a phone-number-shaped run of digits.

        A PII pre-filter, matching the common ``NNN-NNN-NNNN`` grouping.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["call 555-123-4567", "no digits"]})
                >>> ds.select(r=bt.col("s").str.has_phone()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches(self._PHONE_RE)

    def phone_count(self) -> StrFunc:
        """Count phone-number-shaped runs of digits (→ Int64).

        Returns:
            An Int64 expression of the match count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["555-123-4567 and 555.987.6543"]})
                >>> ds.select(r=bt.col("s").str.phone_count()).to_pydict()
                {'r': [2]}
        """
        return self.regexp_count(self._PHONE_RE)

    def remove_phones(self) -> StrFunc:
        """Strip phone-number-shaped digit runs — a PII scrub alongside `remove_emails`.

        Returns:
            A Utf8 expression with phone numbers removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["Call 555-123-4567 now"]})
                >>> ds.select(r=bt.col("s").str.remove_phones()).to_pydict()
                {'r': ['Call  now']}
        """
        return self.regexp_replace_all(self._PHONE_RE, "")

    def mask_emails(self, replacement: str = "[EMAIL]") -> StrFunc:
        """Replace email addresses with a placeholder token, keeping the sentence shape.

        Preferred over deletion for training data: the model still sees that an address
        was there, without memorizing it.

        Args:
            replacement: The token to substitute for each address.

        Returns:
            A Utf8 expression with addresses masked.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["mail a@b.com now"]})
                >>> ds.select(r=bt.col("s").str.mask_emails()).to_pydict()
                {'r': ['mail [EMAIL] now']}
        """
        return self.regexp_replace_all(self._EMAIL_RE, replacement)

    def mask_urls(self, replacement: str = "[URL]") -> StrFunc:
        """Replace HTTP(S) URLs with a placeholder token, keeping the sentence shape.

        Args:
            replacement: The token to substitute for each URL.

        Returns:
            A Utf8 expression with URLs masked.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["see http://a.com now"]})
                >>> ds.select(r=bt.col("s").str.mask_urls()).to_pydict()
                {'r': ['see [URL] now']}
        """
        return self.regexp_replace_all(self._URL_RE, replacement)

    def uppercase_word_count(self) -> StrFunc:
        """Count all-caps words of two or more letters (→ Int64) — a shouting/header signal.

        Returns:
            An Int64 expression of the all-caps word count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["HELLO WORLD ok"]})
                >>> ds.select(r=bt.col("s").str.uppercase_word_count()).to_pydict()
                {'r': [2]}
        """
        return self.regexp_count(r"\b[A-Z]{2,}\b")

    def long_word_count(self, min_length: int = 5) -> StrFunc:
        """Count words of at least `min_length` characters (→ Int64).

        Args:
            min_length: The minimum word length to count.

        Returns:
            An Int64 expression of the long-word count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["HELLO WORLD ok"]})
                >>> ds.select(r=bt.col("s").str.long_word_count(5)).to_pydict()
                {'r': [2]}
        """
        return self.regexp_count(r"\b\w{" + str(min_length) + r",}\b")

    def hashtag_count(self) -> StrFunc:
        """Count ``#hashtag`` tokens (→ Int64) — a social-media provenance signal.

        Returns:
            An Int64 expression of the hashtag count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["#a and #b"]})
                >>> ds.select(r=bt.col("s").str.hashtag_count()).to_pydict()
                {'r': [2]}
        """
        return self.regexp_count(r"#\w+")

    def mention_count(self) -> StrFunc:
        """Count ``@mention`` tokens (→ Int64) — a social-media provenance signal.

        Returns:
            An Int64 expression of the mention count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["@a and @b"]})
                >>> ds.select(r=bt.col("s").str.mention_count()).to_pydict()
                {'r': [2]}
        """
        return self.regexp_count(r"@\w+")

    def symbol_to_word_ratio(self) -> Expr:
        """Punctuation characters per word — high values mark markup and ASCII art.

        A Gopher-style filter threshold; an empty string yields null rather than dividing
        by zero.

        Returns:
            A Float64 expression of the symbol-to-word ratio.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hi there!!"]})
                >>> ds.select(r=bt.col("s").str.symbol_to_word_ratio()).to_pydict()
                {'r': [1.0]}
        """

        return self.regexp_count(r"[^\w\s]") / nullif(self.word_count(), lit(0))

    def paragraph_count(self) -> Expr:
        r"""Count paragraphs, separated by a blank line (→ Int64).

        Returns:
            An Int64 expression of the paragraph count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["one\n\ntwo"]})
                >>> ds.select(r=bt.col("s").str.paragraph_count()).to_pydict()
                {'r': [2]}
        """
        return self.regexp_count(r"\n\s*\n") + 1

    def code_fence_count(self) -> StrFunc:
        """Count Markdown code fences (→ Int64) — a code-content signal.

        Returns:
            An Int64 expression of the fence count (two per fenced block).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["text ```x``` end"]})
                >>> ds.select(r=bt.col("s").str.code_fence_count()).to_pydict()
                {'r': [2]}
        """
        return self.regexp_count("```")

    def looks_like_code(self) -> StrFunc:
        """True where the text shows source-code punctuation or keywords — a coarse filter.

        Matches braces, semicolons, or a ``def``/``if (`` opener. Useful to route code out
        of a prose corpus (or to keep only code for a code model).

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["if (x) { y; }", "plain prose"]})
                >>> ds.select(r=bt.col("s").str.looks_like_code()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches(r"[{};]|\bdef\b|\bif\s*\(")

    def has_repeated_punctuation(self) -> StrFunc:
        """True where three or more sentence marks run together, as in ``"Wow!!!"``.

        A low-quality/emphatic-text signal used when filtering scraped corpora.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["Wow!!! really", "calm."]})
                >>> ds.select(r=bt.col("s").str.has_repeated_punctuation()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches(r"[!?.]{3,}")

    def is_single_line(self) -> Expr:
        r"""True where the text contains no newline — a title/snippet check.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["one line", "two\nlines"]})
                >>> ds.select(r=bt.col("s").str.is_single_line()).to_pydict()
                {'r': [True, False]}
        """
        return ~self.regexp_matches("\n")

    def ends_with_punctuation(self) -> StrFunc:
        """True where the text ends in ``.``, ``!``, or ``?`` — a truncation check.

        A document that stops mid-sentence is usually a bad crawl or a cut-off chunk.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["complete.", "cut off mid"]})
                >>> ds.select(r=bt.col("s").str.ends_with_punctuation()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches(r"[.!?]\s*$")

    def quote_count(self) -> StrFunc:
        """Count double-quote characters (→ Int64) — a dialogue/citation signal.

        Returns:
            An Int64 expression of the quote count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ['say "hi" now']})
                >>> ds.select(r=bt.col("s").str.quote_count()).to_pydict()
                {'r': [2]}
        """
        return self.regexp_count('"')

    def paren_count(self) -> StrFunc:
        """Count parenthesis characters (→ Int64) — a citation/code density signal.

        Returns:
            An Int64 expression of the parenthesis count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a (b) c"]})
                >>> ds.select(r=bt.col("s").str.paren_count()).to_pydict()
                {'r': [2]}
        """
        return self.regexp_count(r"[()]")

    def digit_to_word_ratio(self) -> Expr:
        """Digit characters per word — high values mark tables, logs, and ID dumps.

        An empty string yields null rather than dividing by zero.

        Returns:
            A Float64 expression of the digit-to-word ratio.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a 1 2"]})
                >>> ds.select(r=bt.col("s").str.digit_to_word_ratio().round(3)).to_pydict()
                {'r': [0.667]}
        """

        return self.regexp_count("[0-9]") / nullif(self.word_count(), lit(0))

    def newline_count(self) -> StrFunc:
        r"""Count newline characters (→ Int64).

        Returns:
            An Int64 expression of the newline count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a\nb\nc"]})
                >>> ds.select(r=bt.col("s").str.newline_count()).to_pydict()
                {'r': [2]}
        """
        return self.regexp_count("\n")

    def tab_count(self) -> StrFunc:
        r"""Count tab characters (→ Int64) — a pasted-table signal.

        Returns:
            An Int64 expression of the tab count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a\tb"]})
                >>> ds.select(r=bt.col("s").str.tab_count()).to_pydict()
                {'r': [1]}
        """
        return self.regexp_count("\t")

    def space_count(self) -> StrFunc:
        """Count space characters (→ Int64).

        Returns:
            An Int64 expression of the space count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a b c"]})
                >>> ds.select(r=bt.col("s").str.space_count()).to_pydict()
                {'r': [2]}
        """
        return self.regexp_count(" ")

    def is_short(self, max_chars: int) -> Expr:
        """True where the text is at most `max_chars` long — the stub-document filter.

        Args:
            max_chars: The inclusive maximum length.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hi", "a longer document"]})
                >>> ds.select(r=bt.col("s").str.is_short(5)).to_pydict()
                {'r': [True, False]}
        """

        max_chars = require_int(max_chars, func="str.is_short", arg="max_chars")
        return self.len() <= Lit(max_chars)

    def is_long(self, min_chars: int) -> Expr:
        """True where the text is at least `min_chars` long.

        Args:
            min_chars: The inclusive minimum length.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hi", "a longer document"]})
                >>> ds.select(r=bt.col("s").str.is_long(5)).to_pydict()
                {'r': [False, True]}
        """

        min_chars = require_int(min_chars, func="str.is_long", arg="min_chars")
        return self.len() >= Lit(min_chars)

    def is_question(self) -> StrFunc:
        """True where the text ends in a question mark.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["why?", "because."]})
                >>> ds.select(r=bt.col("s").str.is_question()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches(r"\?\s*$")

    def is_exclamation(self) -> StrFunc:
        """True where the text ends in an exclamation mark.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["wow!", "ok."]})
                >>> ds.select(r=bt.col("s").str.is_exclamation()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches(r"!\s*$")

    def starts_with_capital(self) -> StrFunc:
        """True where the text begins with an uppercase letter — a prose-shape signal.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["Hello", "hello"]})
                >>> ds.select(r=bt.col("s").str.starts_with_capital()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches("^[A-Z]")

    def is_all_caps(self) -> Expr:
        """True where the text has letters and none of them are lowercase.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["LOUD TEXT", "Normal text"]})
                >>> ds.select(r=bt.col("s").str.is_all_caps()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches("[A-Z]") & ~self.regexp_matches("[a-z]")

    def word_char_ratio(self) -> Expr:
        """Word characters per total character — the inverse of markup/symbol density.

        An empty string yields null rather than dividing by zero.

        Returns:
            A Float64 expression of the ratio.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ab!!"]})
                >>> ds.select(r=bt.col("s").str.word_char_ratio()).to_pydict()
                {'r': [0.5]}
        """
        return self._char_ratio(r"\w")

    def extract_urls(self) -> StrFunc:
        """Every HTTP(S) URL in the text, as a ``List<Utf8>`` — link harvesting.

        Returns:
            A List expression of the URLs found.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a http://x.co b"]})
                >>> ds.select(r=bt.col("s").str.extract_urls()).to_pydict()
                {'r': [['http://x.co']]}
        """
        return self.regexp_extract_all(self._URL_RE)

    def extract_emails(self) -> StrFunc:
        """Every email address in the text, as a ``List<Utf8>`` — PII auditing.

        Returns:
            A List expression of the addresses found.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a@b.com and c@d.org"]})
                >>> ds.select(r=bt.col("s").str.extract_emails()).to_pydict()
                {'r': [['a@b.com', 'c@d.org']]}
        """
        return self.regexp_extract_all(self._EMAIL_RE)

    def extract_numbers(self) -> StrFunc:
        """Every run of digits in the text, as a ``List<Utf8>``.

        Returns:
            A List expression of the digit runs found.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a 42 b 7"]})
                >>> ds.select(r=bt.col("s").str.extract_numbers()).to_pydict()
                {'r': [['42', '7']]}
        """
        return self.regexp_extract_all("[0-9]+")

    def extract_hashtags(self) -> StrFunc:
        """Every ``#hashtag`` in the text, as a ``List<Utf8>``.

        Returns:
            A List expression of the hashtags found.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["#a and #b"]})
                >>> ds.select(r=bt.col("s").str.extract_hashtags()).to_pydict()
                {'r': [['#a', '#b']]}
        """
        return self.regexp_extract_all(r"#\w+")

    def extract_mentions(self) -> StrFunc:
        """Every ``@mention`` in the text, as a ``List<Utf8>``.

        Returns:
            A List expression of the mentions found.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["@a and @b"]})
                >>> ds.select(r=bt.col("s").str.extract_mentions()).to_pydict()
                {'r': [['@a', '@b']]}
        """
        return self.regexp_extract_all(r"@\w+")

    def first_sentence(self) -> StrFunc:
        """The text up to and including the first sentence mark — a snippet/summary field.

        Returns:
            A Utf8 expression of the first sentence, empty if there is no sentence mark.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["One. Two."]})
                >>> ds.select(r=bt.col("s").str.first_sentence()).to_pydict()
                {'r': ['One.']}
        """
        return self.regexp_extract(r"^[^.!?]*[.!?]", 0)

    def first_word(self) -> StrFunc:
        """The first whitespace-separated token.

        Returns:
            A Utf8 expression of the first word.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello big world"]})
                >>> ds.select(r=bt.col("s").str.first_word()).to_pydict()
                {'r': ['hello']}
        """
        return self.regexp_extract(r"^\S+", 0)

    def slugify(self) -> StrFunc:
        """Lowercase and hyphenate into a URL/identifier-safe slug.

        Runs of non-alphanumeric characters collapse to a single ``-`` and the ends are
        trimmed — the canonical form for a document id or a dedup key.

        Returns:
            A Utf8 expression of the slug.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["Hello, World!"]})
                >>> ds.select(r=bt.col("s").str.slugify()).to_pydict()
                {'r': ['hello-world']}
        """
        return self.lower().str.regexp_replace_all(r"[^a-z0-9]+", "-").str.trim("-")

    def remove_bullets(self) -> StrFunc:
        """Strip a leading list bullet (``-``, ``*``, or ``+``) and its spacing.

        Returns:
            A Utf8 expression without the leading bullet.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["- item one"]})
                >>> ds.select(r=bt.col("s").str.remove_bullets()).to_pydict()
                {'r': ['item one']}
        """
        return self.regexp_replace_all(r"^\s*[-*+]\s+", "")

    def remove_repeated_punctuation(self) -> StrFunc:
        """Collapse runs of sentence marks to one, turning ``"Wow!!!"`` into ``"Wow!"``.

        Returns:
            A Utf8 expression with punctuation runs collapsed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["Wow!!! ok"]})
                >>> ds.select(r=bt.col("s").str.remove_repeated_punctuation()).to_pydict()
                {'r': ['Wow! ok']}
        """
        return self.regexp_replace_all(r"([!?.])[!?.]+", r"\1")

    def remove_markdown_links(self) -> StrFunc:
        """Reduce ``[text](url)`` to just ``text``, dropping the target.

        Keeps the prose while removing the link noise that would otherwise inflate the
        symbol ratio of a Markdown corpus.

        Returns:
            A Utf8 expression with link targets removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["see [text](http://x) here"]})
                >>> ds.select(r=bt.col("s").str.remove_markdown_links()).to_pydict()
                {'r': ['see text here']}
        """
        return self.regexp_replace_all(r"\[([^\]]*)\]\([^)]*\)", r"\1")

    def remove_code_blocks(self) -> StrFunc:
        """Delete fenced ``` code blocks, keeping the surrounding prose.

        Returns:
            A Utf8 expression with fenced blocks removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["text ```py q``` end"]})
                >>> ds.select(r=bt.col("s").str.remove_code_blocks()).to_pydict()
                {'r': ['text  end']}
        """
        return self.regexp_replace_all(r"```[^`]*```", "")

    def remove_stopwords(self, words: Iterable[str]) -> StrFunc:
        """Delete whole-word occurrences of `words`, matching either case of the first letter.

        The engine's regex engine has no inline case-insensitive flag, so each word is
        expanded to its lowercase and capitalized forms.

        Args:
            words: The stopwords to remove.

        Returns:
            A Utf8 expression with the stopwords removed.

        Raises:
            PlanError: If `words` is empty.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["The cat is here"]})
                >>> ds.select(r=bt.col("s").str.remove_stopwords(["the", "is"])).to_pydict()
                {'r': [' cat  here']}
        """
        forms = [w for word in words for w in (re.escape(word), re.escape(word.capitalize()))]
        if not forms:
            raise PlanError("remove_stopwords() requires at least one word")
        return self.regexp_replace_all(r"\b(?:" + "|".join(forms) + r")\b", "")

    def truncate_sentences(self, n: int) -> StrFunc:
        """Keep at most the first `n` sentences, cutting on a sentence mark.

        The summary/snippet budget that never leaves a half sentence behind.

        Args:
            n: Maximum sentences to keep (must be >= 1).

        Returns:
            A Utf8 expression truncated to `n` sentences.

        Raises:
            PlanError: If `n` < 1.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["One. Two! Three?"]})
                >>> ds.select(r=bt.col("s").str.truncate_sentences(2)).to_pydict()
                {'r': ['One. Two!']}
        """
        n = require_int(n, func="str.truncate_sentences", arg="n", minimum=1)
        return self.regexp_extract(r"^(?:[^.!?]*[.!?]){1," + str(n) + r"}", 0)

    def avg_sentence_length(self) -> Expr:
        """Words per sentence — very long or very short values mark non-prose.

        An empty string yields null rather than dividing by zero.

        Returns:
            A Float64 expression of the mean sentence length in words.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["One two. Three four."]})
                >>> ds.select(r=bt.col("s").str.avg_sentence_length()).to_pydict()
                {'r': [2.0]}
        """

        return self.word_count() / nullif(self.sentence_count(), lit(0))

    def has_currency(self) -> StrFunc:
        """True where the text contains a currency symbol — a price/commerce signal.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["costs $5", "free"]})
                >>> ds.select(r=bt.col("s").str.has_currency()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches(r"[$€£¥]")

    def last_word(self) -> StrFunc:
        """The last whitespace-separated token.

        Returns:
            A Utf8 expression of the final word.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello big world"]})
                >>> ds.select(r=bt.col("s").str.last_word()).to_pydict()
                {'r': ['world']}
        """
        return self.regexp_extract(r"\S+$", 0)

    def is_url(self) -> StrFunc:
        """True where the whole string is a single HTTP(S) URL.

        Stricter than :meth:`has_url` — it rejects prose that merely mentions a link,
        which is what a link-only-row filter needs.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["http://a.co", "see http://a.co"]})
                >>> ds.select(r=bt.col("s").str.is_url()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches(r"^https?://\S+$")

    def is_email(self) -> StrFunc:
        """True where the whole string is a single email address.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a@b.com", "mail a@b.com"]})
                >>> ds.select(r=bt.col("s").str.is_email()).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches(r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$")

    # --- pandas-compatible string spellings -----------------------------------------

    def strip(self, chars: str | None = None) -> StrFunc:
        """Trim from both ends — the pandas ``str.strip`` spelling of :meth:`trim`.

        Unlike Python's ``str.strip()``, the no-argument form removes the ASCII **space**
        only, following SQL ``TRIM``. Pass ``strip(" \t\n")`` to also drop tabs and newlines.

        Args:
            chars: The characters to strip; the ASCII space when ``None``.

        Returns:
            A Utf8 expression with leading and trailing characters removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["  ab  "]})
                >>> ds.select(r=bt.col("s").str.strip()).to_pydict()
                {'r': ['ab']}
        """
        return self.trim(chars)

    def startswith(self, pattern: str) -> StrFunc:
        """True where the string starts with `pattern` — the pandas ``str.startswith``.

        Args:
            pattern: The literal prefix to test for.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abc", "xbc"]})
                >>> ds.select(r=bt.col("s").str.startswith("a")).to_pydict()
                {'r': [True, False]}
        """
        return self.starts_with(pattern)

    def endswith(self, pattern: str) -> StrFunc:
        """True where the string ends with `pattern` — the pandas ``str.endswith``.

        Args:
            pattern: The literal suffix to test for.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abc", "abx"]})
                >>> ds.select(r=bt.col("s").str.endswith("c")).to_pydict()
                {'r': [True, False]}
        """
        return self.ends_with(pattern)

    def match(self, pattern: str) -> StrFunc:
        """True where the regex `pattern` matches — the pandas ``str.match``.

        Args:
            pattern: The regular expression to test.

        Returns:
            A Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a1", "ab"]})
                >>> ds.select(r=bt.col("s").str.match("[a-z][0-9]")).to_pydict()
                {'r': [True, False]}
        """
        return self.regexp_matches(pattern)

    def title(self) -> StrFunc:
        """Title-case each word — the pandas ``str.title`` spelling of :meth:`initcap`.

        Returns:
            A Utf8 expression with each word's first letter uppercased.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello world"]})
                >>> ds.select(r=bt.col("s").str.title()).to_pydict()
                {'r': ['Hello World']}
        """
        return self.initcap()

    def removeprefix(self, prefix: str) -> StrFunc:
        """Drop `prefix` from the start if present, else leave the string unchanged.

        Mirrors Python's ``str.removeprefix``; the literal is regex-escaped, so it is
        matched exactly.

        Args:
            prefix: The literal prefix to remove.

        Returns:
            A Utf8 expression with the prefix removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["id_7", "7"]})
                >>> ds.select(r=bt.col("s").str.removeprefix("id_")).to_pydict()
                {'r': ['7', '7']}
        """
        return self.regexp_replace("^" + re.escape(prefix), "")

    def removesuffix(self, suffix: str) -> StrFunc:
        """Drop `suffix` from the end if present, else leave the string unchanged.

        Mirrors Python's ``str.removesuffix``; the literal is regex-escaped.

        Args:
            suffix: The literal suffix to remove.

        Returns:
            A Utf8 expression with the suffix removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["7_id", "7"]})
                >>> ds.select(r=bt.col("s").str.removesuffix("_id")).to_pydict()
                {'r': ['7', '7']}
        """
        return self.regexp_replace(re.escape(suffix) + "$", "")

    def position(self, pattern: str) -> StrFunc:
        """Find the 1-based index of ``pattern`` in the string, or 0 if absent.

        Returns Int64.

        Args:
            pattern: The literal substring to locate.

        Returns:
            A new Int64 expression: the 1-based index, or 0.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello"]})
                >>> ds.select(bt.col("s").str.position("lo").alias("r")).to_pydict()
                {'r': [4]}
        """
        return StrFunc("position", self._e, pattern=pattern)

    def substring_index(self, delimiter: str, count: int) -> StrFunc:
        """Return the substring before the ``count``-th occurrence of ``delimiter``.

        Spark ``substring_index``: ``count > 0`` counts delimiters from the left,
        ``count < 0`` from the right. Returns Utf8.

        Args:
            delimiter: The delimiter to count occurrences of.
            count: Which occurrence to cut at; sign selects the direction.

        Returns:
            A new Utf8 expression: the substring before the cut.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a.b.c.d"]})
                >>> ds.select(bt.col("s").str.substring_index(".", 2).alias("r")).to_pydict()
                {'r': ['a.b']}
        """
        count = require_int(count, func="str.substring_index", arg="count")
        return StrFunc("substring_index", self._e, pattern=delimiter, start=count)

    def overlay(self, replacement: str, pos: int, length: int | None = None) -> StrFunc:
        """Replace ``length`` characters from 1-based ``pos`` with ``replacement``.

        SQL ``OVERLAY``: ``length`` defaults to the replacement's length. Returns
        Utf8.

        Args:
            replacement: The string to splice in.
            pos: 1-based index where the replacement begins.
            length: Characters to overwrite; defaults to ``len(replacement)``.

        Returns:
            A new Utf8 expression with the range replaced.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello"]})
                >>> ds.select(bt.col("s").str.overlay("XY", 2).alias("r")).to_pydict()
                {'r': ['hXYlo']}
        """
        pos = require_int(pos, func="str.overlay", arg="pos")
        if length is not None:
            length = require_int(length, func="str.overlay", arg="length")
        return StrFunc("overlay", self._e, replacement=replacement, start=pos, length=length)

    def regexp_extract_all(self, pattern: str, group: int = 0) -> StrFunc:
        """Collect every regex match as a list of strings (DuckDB ``regexp_extract_all``).

        Returns an empty list when there are no matches. Chain ``.list`` to operate
        on the result. Returns List<Utf8>.

        With a ``group`` above 0 the list holds that capture group of each match rather
        than the whole match, and an element is null where the group did not participate
        in its match. Asking for a group the pattern does not have is an error, matching
        DuckDB rather than returning empty lists.

        Args:
            pattern: The regular expression to match.
            group: Capture group index; 0 (default) is the whole match.

        Returns:
            A new List<Utf8> expression of all matches.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> d = bt.from_pydict({"s": ["2024-01-15"]})
                >>> d.select(
                ...     bt.col("s").str.regexp_extract_all(r"\\d+").alias("r")
                ... ).to_pydict()
                {'r': [['2024', '01', '15']]}

                >>> d = bt.from_pydict({"s": ["100-200, 300-400"]})
                >>> d.select(
                ...     bt.col("s").str.regexp_extract_all(r"(\\d+)-(\\d+)", 1).alias("r")
                ... ).to_pydict()
                {'r': [['100', '300']]}
        """
        return StrFunc("regexp_extract_all", self._e, pattern=pattern, start=group)

    def regexp_count(self, pattern: str) -> StrFunc:
        """Count non-overlapping regex matches (DuckDB ``regexp_count``).

        Returns Int64.

        Args:
            pattern: The regular expression to match.

        Returns:
            A new Int64 expression: the match count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a1b2c3"]})
                >>> ds.select(bt.col("s").str.regexp_count(r"\\d").alias("r")).to_pydict()
                {'r': [3]}
        """
        return StrFunc("regexp_count", self._e, pattern=pattern)

    def levenshtein(self, target: str) -> StrFunc:
        """Compute the Levenshtein edit distance to the constant string ``target``.

        DuckDB ``levenshtein`` against a literal — the basis for fuzzy matching and
        dedup against a reference value. Returns Int64.

        Args:
            target: The literal string to measure distance to.

        Returns:
            A new Int64 expression: the edit distance.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["sitting"]})
                >>> ds.select(bt.col("s").str.levenshtein("kitten").alias("r")).to_pydict()
                {'r': [3]}
        """
        return StrFunc("levenshtein", self._e, pattern=target)

    def damerau_levenshtein(self, target: str) -> StrFunc:
        """Damerau-Levenshtein edit distance to the constant string ``target`` (→ Int64).

        DuckDB ``damerau_levenshtein`` against a literal: like `levenshtein`, but a swap of
        two adjacent characters counts as **one** edit rather than two — so it scores a
        typo like ``"teh"`` vs ``"the"`` as distance 1. The better default for matching
        human-typed text (search queries, names) where transpositions are common.

        Args:
            target: The literal string to measure distance to.

        Returns:
            A new Int64 expression: the Damerau-Levenshtein distance.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["teh"]})
                >>> ds.select(r=bt.col("s").str.damerau_levenshtein("the")).to_pydict()
                {'r': [1]}
        """
        return StrFunc("damerau_levenshtein", self._e, pattern=target)

    def jaro_similarity(self, target: str) -> StrFunc:
        """Compute the Jaro similarity to the constant string ``target`` (→ Float64).

        DuckDB ``jaro_similarity`` against a literal: a ``[0, 1]`` fuzzy-match score (1.0
        identical, 0.0 nothing in common) based on matching characters and transpositions.
        The go-to metric for **entity resolution / record linkage** on short strings like
        names, where an edit distance is too coarse.

        Args:
            target: The literal string to score against.

        Returns:
            A new Float64 expression: the Jaro similarity.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["MARTHA"]})
                >>> r = ds.select(sim=bt.col("s").str.jaro_similarity("MARHTA")).to_pydict()
                >>> round(r["sim"][0], 4)
                0.9444
        """
        return StrFunc("jaro_similarity", self._e, pattern=target)

    def jaro_winkler_similarity(self, target: str) -> StrFunc:
        """Compute the Jaro-Winkler similarity to the constant string ``target`` (→ Float64).

        DuckDB ``jaro_winkler_similarity`` against a literal: Jaro plus a bonus for a shared
        prefix (up to 4 characters), so strings that agree at the start — as names usually
        do — score higher. ``[0, 1]``. Preferred over plain Jaro for name matching.

        Args:
            target: The literal string to score against.

        Returns:
            A new Float64 expression: the Jaro-Winkler similarity.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["MARTHA"]})
                >>> r = ds.select(sim=bt.col("s").str.jaro_winkler_similarity("MARHTA")).to_pydict()
                >>> round(r["sim"][0], 4)
                0.9611
        """
        return StrFunc("jaro_winkler_similarity", self._e, pattern=target)

    def hamming(self, target: str) -> StrFunc:
        """Count the positions at which the value and ``target`` differ (→ Int64).

        DuckDB ``hamming`` (also spelled ``mismatches``). Defined only for strings of
        equal length: an unequal length raises rather than comparing a prefix, because a
        prefix comparison would answer a caller's mistake with a plausible number.
        Counted in Unicode scalar values, as :meth:`len` counts them.

        Args:
            target: The literal string to compare against, of the same length.

        Returns:
            A new Int64 expression: the number of differing positions.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abc", "abd", "xyz"]})
                >>> ds.select(d=bt.col("s").str.hamming("abc")).to_pydict()
                {'d': [0, 1, 3]}
        """
        return StrFunc("hamming", self._e, pattern=target)

    def jaccard(self, target: str) -> StrFunc:
        """Compute the Jaccard similarity of the two strings' character sets (→ Float64).

        DuckDB ``jaccard``: the size of the intersection over the size of the union of the
        two values' distinct characters, in ``[0, 1]``. A repeated character does not
        change the answer, which is what makes it a *set* similarity — for element-wise
        list similarity use ``.list.jaccard``.

        Args:
            target: The literal string to score against.

        Returns:
            A new Float64 expression: the Jaccard similarity.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abc", "aab"]})
                >>> ds.select(j=bt.col("s").str.jaccard("abd")).to_pydict()
                {'j': [0.5, 0.6666666666666666]}
        """
        return StrFunc("jaccard_similarity", self._e, pattern=target)

    def url_encode(self) -> StrFunc:
        """Percent-encode the value for use in a URL (→ Utf8).

        DuckDB ``url_encode``. Everything outside the RFC 3986 unreserved set
        (``A-Za-z0-9-_.~``) becomes ``%XX`` over the UTF-8 bytes, ``/`` and ``+``
        included — this encodes a URL *component*, not a whole URL, so it is safe to
        paste into a query string or a path segment.

        Returns:
            A new Utf8 expression: the percent-encoded value.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a b/c"]})
                >>> ds.select(u=bt.col("s").str.url_encode()).to_pydict()
                {'u': ['a%20b%2Fc']}
        """
        return StrFunc("url_encode", self._e)

    def url_decode(self) -> StrFunc:
        """Percent-decode the value (→ Utf8).

        DuckDB ``url_decode``, the inverse of :meth:`url_encode`. A malformed escape (a
        ``%`` not followed by two hex digits, or bytes that do not decode as UTF-8) is
        left as written rather than raising or nulling the row, matching DuckDB.

        Returns:
            A new Utf8 expression: the decoded value.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a%20b%2Fc", "100%"]})
                >>> ds.select(u=bt.col("s").str.url_decode()).to_pydict()
                {'u': ['a b/c', '100%']}
        """
        return StrFunc("url_decode", self._e)

    def join(self, delimiter: str = "") -> Expr:
        """Concatenate every value into one string (Polars ``str.join``, → Utf8).

        An **aggregate**, not a row operation: it collects the column and joins it, so it
        belongs in a `select`, a `group_by().agg(...)`, or anywhere else an aggregate is
        allowed. SQL spells the same thing ``string_agg``.

        Args:
            delimiter: The separator placed between values; none by default.

        Returns:
            A Utf8 expression: the joined text of the group.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "s": ["x", "y", "z"]})
                >>> ds.group_by("g").agg(r=bt.col("s").str.join("-")).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': ['x-y', 'z']}
        """

        return ListJoin(AggExpr("list_agg", self._e), delimiter)

    def escape_regex(self) -> StrFunc:
        """Escape the regex metacharacters, spelled as Polars ``str.escape_regex``.

        Returns:
            A new Utf8 expression, safe to embed in a pattern as a literal.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a.b"]})
                >>> ds.select(r=bt.col("s").str.escape_regex()).to_pydict()
                {'r': ['a\\\\.b']}
        """
        return self.regexp_escape()

    def regexp_escape(self) -> StrFunc:
        """Escape the regex metacharacters in the value (→ Utf8).

        DuckDB ``regexp_escape``. Use it to embed data in a pattern as a literal, so a
        value containing ``.`` or ``[`` matches itself instead of acting as syntax.

        Returns:
            A new Utf8 expression: the value with its metacharacters escaped.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a.b"]})
                >>> ds.select(e=bt.col("s").str.regexp_escape()).to_pydict()
                {'e': ['a\\\\.b']}
        """
        return StrFunc("regexp_escape", self._e)

    def parse_filename(self) -> StrFunc:
        """Take the final component of a path (→ Utf8).

        DuckDB ``parse_filename``: everything after the last ``/`` or ``\\\\``, or the
        whole value when there is no separator.

        Returns:
            A new Utf8 expression: the filename.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"p": ["/data/2024/events.parquet"]})
                >>> ds.select(f=bt.col("p").str.parse_filename()).to_pydict()
                {'f': ['events.parquet']}
        """
        return StrFunc("parse_filename", self._e)

    def parse_dirname(self) -> StrFunc:
        """Take the *first* component of a path (→ Utf8).

        DuckDB ``parse_dirname``: ``/`` for an absolute POSIX path, the leading directory
        for a relative one, and the empty string when there is no separator. This is not
        the directory holding the file — that is :meth:`parse_dirpath`, and the two
        differ on every path deeper than one level.

        Returns:
            A new Utf8 expression: the first path component.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"p": ["/data/2024/events.parquet", "a/b/c.txt"]})
                >>> ds.select(d=bt.col("p").str.parse_dirname()).to_pydict()
                {'d': ['/', 'a']}
        """
        return StrFunc("parse_dirname", self._e)

    def parse_dirpath(self) -> StrFunc:
        """Take everything before the last separator of a path (→ Utf8).

        DuckDB ``parse_dirpath``: the directory holding the file, empty when the value has
        no separator.

        Returns:
            A new Utf8 expression: the directory path.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"p": ["/data/2024/events.parquet", "a/b/c.txt"]})
                >>> ds.select(d=bt.col("p").str.parse_dirpath()).to_pydict()
                {'d': ['/data/2024', 'a/b']}
        """
        return StrFunc("parse_dirpath", self._e)

    def parse_path(self) -> StrFunc:
        """Split a path into its components (→ List<Utf8>).

        DuckDB ``parse_path``. A leading separator is kept as its own first element, so an
        absolute path stays distinguishable from a relative one.

        Returns:
            A new List<Utf8> expression: the path components.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"p": ["/data/2024/events.parquet"]})
                >>> ds.select(c=bt.col("p").str.parse_path()).to_pydict()
                {'c': [['/', 'data', '2024', 'events.parquet']]}
        """
        return StrFunc("parse_path", self._e)

    def to_binary(self) -> StrFunc:
        """Render the value's UTF-8 bytes as ``0``/``1`` characters (→ Utf8).

        DuckDB ``to_binary``: eight characters per byte, most significant bit first.

        Returns:
            A new Utf8 expression: the binary text.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a"]})
                >>> ds.select(b=bt.col("s").str.to_binary()).to_pydict()
                {'b': ['01100001']}
        """
        return StrFunc("to_binary", self._e)

    def from_binary(self) -> StrFunc:
        """Read ``0``/``1`` characters back into text (→ Utf8, nullable).

        DuckDB ``from_binary``, the inverse of :meth:`to_binary`. Input that is not a
        whole number of eight binary digits, or whose bytes are not UTF-8, becomes null
        rather than raising — one corrupt row is a bad row, not a bad query, the same rule
        :meth:`unhex` follows.

        Returns:
            A new Utf8 expression: the decoded text, or null.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"b": ["01100001", "0110000"]})
                >>> ds.select(s=bt.col("b").str.from_binary()).to_pydict()
                {'s': ['a', None]}
        """
        return StrFunc("from_binary", self._e)

    def soundex(self) -> StrFunc:
        """Compute the American Soundex phonetic code, a 4-character key.

        Groups words that sound alike (DuckDB ``soundex``). Returns Utf8.

        Returns:
            A new Utf8 expression: the 4-character Soundex code.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["Robert"]})
                >>> ds.select(bt.col("s").str.soundex().alias("r")).to_pydict()
                {'r': ['R163']}
        """
        return StrFunc("soundex", self._e)

    def right(self, n: int) -> StrFunc:
        """Take the last ``n`` characters (SQL ``right``).

        Args:
            n: Number of trailing characters to keep.

        Returns:
            A new Utf8 expression: the trailing characters.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello"]})
                >>> ds.select(bt.col("s").str.right(3).alias("r")).to_pydict()
                {'r': ['llo']}
        """
        n = require_int(n, func="str.right", arg="n")
        return StrFunc("right", self._e, start=n)

    def ascii(self) -> StrFunc:
        """Return the Unicode codepoint of the first character, 0 if empty (→ Int64).

        Despite the name, returns the full codepoint, not just ASCII.

        Returns:
            A new Int64 expression: the first codepoint, or 0.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["A", "a"]})
                >>> ds.select(bt.col("s").str.ascii().alias("r")).to_pydict()
                {'r': [65, 97]}
        """
        return StrFunc("ascii", self._e)

    def split(self, delimiter: str) -> StrFunc:
        """Split on ``delimiter`` into a list of strings (chain with ``.list``).

        Returns List<Utf8>; a string with no delimiter yields a one-element list.

        Args:
            delimiter: The literal separator to split on.

        Returns:
            A new List<Utf8> expression of the split fields.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a-b-c"]})
                >>> ds.select(bt.col("s").str.split("-").alias("r")).to_pydict()
                {'r': [['a', 'b', 'c']]}
        """
        return StrFunc("split", self._e, pattern=delimiter)

    def regexp_split(self, pattern: str) -> StrFunc:
        """Split on every match of the regex `pattern` into a list of strings.

        The regex counterpart of :meth:`split`, whose delimiter is a literal. Use it where
        the separator varies: a run of whitespace, one of several punctuation marks, a
        digit boundary.

        Args:
            pattern: The regular expression matching each separator.

        Returns:
            A new List<Utf8> :class:`~batcher.Expr` of the pieces between matches; a null
            input gives a null list, and a string with no match gives a one-element list.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a1b22c", "no digits here"]})
                >>> ds.select(r=bt.col("s").str.regexp_split("[0-9]+")).to_pydict()
                {'r': [['a', 'b', 'c'], ['no digits here']]}
        """
        return StrFunc("regexp_split", self._e, pattern=pattern)

    def strip_html(self) -> StrFunc:
        """Recover the readable text of an HTML document → Utf8.

        The first stage of an unstructured-text ingest: scraped pages, product
        descriptions, and email bodies arrive as markup and must become prose before
        chunking, embedding, or training on them.

        This is strictly more correct than the ``regexp_replace('<[^>]*>', '')`` idiom,
        which quietly poisons a corpus in three ways. It leaves the *contents* of
        ``<script>`` and ``<style>`` in the text; it leaves ``&amp;`` and ``&nbsp;``
        undecoded; and it welds ``<p>a</p><p>b</p>`` into ``ab``. Here, tags and comments
        are dropped along with script/style content, entities (named, ``&#38;``, and
        ``&#x26;``) are decoded, element boundaries become a single space, and runs of
        whitespace collapse.

        It is a text extractor, not an HTML parser — it builds no DOM and validates no
        nesting. Malformed markup never raises: a ``<`` that never closes is kept as
        literal text, and an unclosed ``<script>`` consumes the rest of the value (the
        safe direction — the alternative is emitting JavaScript as prose). Null → null.

        Returns:
            A Utf8 expression carrying the extracted text.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"page": ["<p>Tom &amp; Jerry</p><p>x</p>"]})
                >>> ds.select(text=bt.col("page").str.strip_html()).to_pydict()
                {'text': ['Tom & Jerry x']}

                >>> noisy = bt.from_pydict({"page": ["<div>hi<script>f()</script></div>"]})
                >>> noisy.select(text=bt.col("page").str.strip_html()).to_pydict()
                {'text': ['hi']}
        """
        return StrFunc("strip_html", self._e)

    def chunk(self, size: int, overlap: int = 0, boundary: str = "char") -> StrFunc:
        """Slice text into overlapping windows, optionally on word/sentence lines → List<Utf8>.

        The *split* stage of a RAG ingest pipeline: chunk each document, `explode` the
        list into one row per chunk, then embed. Chunks start every ``size - overlap``
        characters and stop once one reaches the end of the text, so the last chunk may
        be shorter but is never wholly contained in its predecessor. `overlap` carries
        context across a boundary, so a sentence cut in half still appears whole in one
        chunk.

        Sizes are in **characters** (Unicode scalar values, as :meth:`len` counts them),
        never bytes, so a chunk boundary never splits a codepoint. An empty string
        yields an empty list; null yields null.

        `boundary` decides where a cut is allowed. The default ``"char"`` cuts at exactly
        `size` characters, which can split a word in half — and a chunk ending
        ``…diagnosed with hyperten`` embeds as something the query ``hypertension
        treatment`` will not match, so a mid-word cut silently costs recall on the very
        chunk that should have answered. ``"word"``, ``"sentence"`` and ``"line"`` back
        the cut off to the last such separator inside the window. When a window holds
        none, the mode degrades to the next-finer one — ``"sentence"`` and ``"line"`` fall
        back to a word boundary before ever cutting mid-word — and only a token longer
        than `size` itself is hard-cut, since it must still be emitted.

        With ``overlap=0`` every mode is lossless: the chunks concatenate back to the
        input, because a separator ends the chunk it belongs to instead of being skipped.

        Args:
            size: Characters per chunk. Must be at least 1.
            overlap: Characters each chunk repeats from the previous one. Must be in
                ``[0, size)`` — an overlap equal to `size` would never advance.
            boundary: Where a chunk may end — ``"char"`` (anywhere), ``"word"`` (after
                whitespace), ``"sentence"`` (after ``.``/``!``/``?``) or ``"line"``
                (after a newline).

        Returns:
            A List<Utf8> expression, one element per chunk.

        Raises:
            PlanError: If `size` < 1, `overlap` is outside ``[0, size)``, or `boundary`
                is not one of the four modes.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"doc": ["abcdefg"]})
                >>> ds.select(r=bt.col("doc").str.chunk(3)).to_pydict()
                {'r': [['abc', 'def', 'g']]}

                >>> ds.select(r=bt.col("doc").str.chunk(4, overlap=2)).to_pydict()
                {'r': [['abcd', 'cdef', 'efg']]}

                >>> # Word boundaries keep each chunk's last word whole.
                >>> words = bt.from_pydict({"doc": ["alpha beta gamma"]})
                >>> words.select(r=bt.col("doc").str.chunk(9, boundary="word")).to_pydict()
                {'r': [['alpha ', 'beta ', 'gamma']]}
        """
        size = require_int(size, func="str.chunk", arg="size", minimum=1)
        overlap = require_int(overlap, func="str.chunk", arg="overlap", minimum=0)
        if not 0 <= overlap < size:
            raise PlanError(f"str.chunk(): overlap must be in [0, {size}), got {overlap}")
        if boundary not in _CHUNK_BOUNDARIES:
            raise PlanError(
                f"str.chunk(): boundary must be one of {sorted(_CHUNK_BOUNDARIES)}, "
                f"got {boundary!r}"
            )
        # `length` carries the chunk size and `start` the overlap — the same reuse of
        # the two scalar slots that `repeat`/`right`/`split_part` already make — and
        # `pattern` carries the boundary mode, which is otherwise unused by `chunk`.
        return StrFunc("chunk", self._e, pattern=boundary, start=overlap, length=size)

    def token_ngrams(self, n: int) -> StrFunc:
        """Every window of `n` adjacent whitespace tokens, joined by a space → List<Utf8>.

        The token-level counterpart of :meth:`chunk`'s character windows, and the unit the
        generation metrics are defined on: BLEU compares n-gram bags, ROUGE-N counts how
        many reference n-grams the output reproduced, and distinct-n measures how many of
        an output's n-grams are unique. Pair it with
        :meth:`~batcher.Expr.list.multiset_overlap` to score two texts against each other,
        or with :meth:`~batcher.Expr.list.n_unique` to score one against itself.

        Tokens are split on whitespace with no normalization, so casing and punctuation
        survive — normalize first (`str.lower`, `str.remove_punctuation`) when the metric
        should ignore them. A text with fewer than `n` tokens yields the single n-gram of
        all of them rather than an empty list, so a short document still contributes.
        An empty or whitespace-only string yields an empty list; null yields null.

        Args:
            n: Tokens per n-gram. Must be at least 1.

        Returns:
            A List<Utf8> expression, one element per n-gram window.

        Raises:
            PlanError: If `n` is less than 1.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": ["the cat sat down"]})
                >>> ds.select(g=bt.col("t").str.token_ngrams(2)).to_pydict()
                {'g': [['the cat', 'cat sat', 'sat down']]}

                >>> # Fewer tokens than `n` still yields one gram.
                >>> bt.from_pydict({"t": ["hi"]}).select(
                ...     g=bt.col("t").str.token_ngrams(3)
                ... ).to_pydict()
                {'g': [['hi']]}
        """
        n = require_int(n, func="str.token_ngrams", arg="n", minimum=1)
        # `length` carries `n`, the same reuse of the scalar slot `chunk`/`repeat` make.
        return StrFunc("token_ngrams", self._e, length=n)

    def compress(self, codec: str) -> StrFunc:
        """Compress each value's raw bytes with `codec` (→ Binary).

        Accepts a text or a binary column; a text column compresses its UTF-8 bytes, so
        the two spellings give identical output. The inverse is :meth:`decompress` under
        the same codec.

        The codecs are ``gzip``, ``zlib``, ``deflate``, ``zstd``, ``brotli``, and ``lz4``.
        Each is a general-purpose default level; a compression *level* argument is
        deliberately not exposed, since it is a second dimension on every codec.

        Args:
            codec: One of the six codec names above.

        Returns:
            A new Binary :class:`~batcher.Expr` of the compressed frames; null stays null.

        Raises:
            PlanError: If `codec` is not a recognized codec name.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello " * 20]})
                >>> out = ds.select(
                ...     n=bt.col("s").str.compress("gzip").str.len_bytes(),
                ...     raw=bt.col("s").str.len_bytes(),
                ... ).to_pydict()
                >>> out["n"][0] < out["raw"][0]
                True
        """
        return StrFunc("compress", self._e, pattern=_require_codec("compress", codec))

    def decompress(self, codec: str) -> StrFunc:
        """Decompress each value's bytes with `codec` (→ Binary); a bad frame is null.

        The inverse of :meth:`compress`. Input that is not a valid frame for `codec`
        yields null rather than failing the query, the same leniency
        :meth:`from_base64` and :meth:`unhex` take: one corrupt blob in a scan of a
        billion rows is a bad row, not a bad query.

        Args:
            codec: One of ``gzip``, ``zlib``, ``deflate``, ``zstd``, ``brotli``, ``lz4``.

        Returns:
            A new Binary :class:`~batcher.Expr` of the decompressed payloads; null where
            the input is null or is not a valid frame.

        Raises:
            PlanError: If `codec` is not a recognized codec name.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["round trip"]})
                >>> ds.select(
                ...     r=bt.col("s").str.compress("zstd").str.decompress("zstd").cast("string")
                ... ).to_pydict()
                {'r': ['round trip']}
        """
        return StrFunc("decompress", self._e, pattern=_require_codec("decompress", codec))

    def to_case(self, style: str) -> StrFunc:
        """Re-case an identifier into `style`, e.g. ``"userID name"`` to ``user_id_name``.

        One word splitter serves every style, so the styles never disagree about where
        the words were: separators (any non-alphanumeric run) split, a lower-to-upper
        transition splits, and an acronym run splits before its final capital, so
        ``parseHTTPResponse`` is three words rather than two or five. Digits stay with
        the word they touch, which keeps ``sha256`` intact.

        The recognized styles, shown on the input ``"userID name"``:

        =============  ==================
        Style          Result
        =============  ==================
        ``snake``      ``user_id_name``
        ``upper_snake``  ``USER_ID_NAME``
        ``camel``      ``userIdName``
        ``pascal``     ``UserIdName``
        ``kebab``      ``user-id-name``
        ``upper_kebab``  ``USER-ID-NAME``
        ``title``      ``User Id Name``
        ``sentence``   ``User id name``
        ``dot``        ``user.id.name``
        ``train``      ``User-Id-Name``
        =============  ==================

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["parseHTTPResponse", "hello world"]})
                >>> ds.select(r=bt.col("s").str.to_case("snake")).to_pydict()
                {'r': ['parse_http_response', 'hello_world']}

                >>> ds.select(r=bt.col("s").str.to_case("pascal")).to_pydict()
                {'r': ['ParseHttpResponse', 'HelloWorld']}

        Args:
            style: One of the styles in the table above.

        Recasing is idempotent in every separator-bearing style. It is not in ``camel``
        or ``pascal`` when the input has consecutive single-letter words, because those
        styles join without a separator: ``a_b_c`` becomes ``aBC``, which reads back as
        two words rather than three. No splitter can recover that, so prefer a separator
        style when the result will be re-parsed.

        Returns:
            A new string :class:`~batcher.Expr` re-cased into `style`; null stays null,
            and an input with no alphanumerics becomes the empty string.

        Raises:
            PlanError: If `style` is not a recognized style name.
        """
        if style not in _CASE_STYLES:
            raise PlanError(
                f"str.to_case(): style must be one of {sorted(_CASE_STYLES)}, got {style!r}"
            )
        # `pattern` carries the style: `to_case` uses none of the other scalar slots.
        return StrFunc("to_case", self._e, pattern=style)

    def minhash(self, num_perm: int = 128, ngram: int = 5) -> StrFunc:
        """A MinHash signature of the text → List<Int64> of `num_perm` values.

        The primitive behind fuzzy deduplication. Two signatures agree on a fraction of
        their positions that estimates the documents' **Jaccard similarity** over their
        character `ngram`-shingles — see :meth:`~batcher.plan.expr_ir._ListNamespace.jaccard`.
        So near-duplicates are found by comparing 128 integers rather than two documents,
        which is what makes deduplicating a web-scale corpus tractable.

        The estimator's standard error is ``1/sqrt(num_perm)``: 128 permutations resolve
        Jaccard to about ±0.09, 256 to ±0.06. Signature values are bounded to 32 bits so
        `jaccard` counts agreements exactly.

        `ds.ml.near_duplicates` / `ds.ml.drop_near_duplicates` build the LSH banding on
        top of this; reach for those first.

        Args:
            num_perm: Hash permutations — the signature length. More is more accurate
                and proportionally slower.
            ngram: Shingle width in **characters**. Larger is stricter (fewer accidental
                matches on short common substrings).

        Returns:
            A List<Int64> expression: the row's signature. Null text → null signature.

        Raises:
            PlanError: If `num_perm` or `ngram` is less than 1.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": ["the quick brown fox"]})
                >>> len(ds.select(s=bt.col("t").str.minhash(64)).to_pydict()["s"][0])
                64
        """
        if num_perm < 1:
            raise PlanError(f"str.minhash(): num_perm must be >= 1, got {num_perm}")
        if ngram < 1:
            raise PlanError(f"str.minhash(): ngram must be >= 1, got {ngram}")
        # `length` carries num_perm and `start` the shingle width — the same reuse of the
        # two scalar slots `chunk`/`repeat`/`split_part` make.
        return StrFunc("minhash", self._e, start=ngram, length=num_perm)

    def regexp_matches(self, pattern: str) -> StrFunc:
        """Test whether the regex ``pattern`` matches anywhere in the string (→ Bool).

        An unanchored search; see :meth:`like` for SQL wildcard matching.

        Args:
            pattern: The regular expression to test.

        Returns:
            A new Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a1", "bb"]})
                >>> ds.select(bt.col("s").str.regexp_matches(r"\\d+").alias("r")).to_pydict()
                {'r': [True, False]}
        """
        return StrFunc("regexp_matches", self._e, pattern=pattern)

    def like(self, pattern: str) -> StrFunc:
        """Match the SQL ``LIKE`` pattern, anchored to the whole string (→ Bool).

        ``%`` matches any run of characters and ``_`` matches exactly one.

        Args:
            pattern: A SQL ``LIKE`` pattern using ``%`` and ``_`` wildcards.

        Returns:
            A new Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello", "world"]})
                >>> ds.select(bt.col("s").str.like("h%o").alias("r")).to_pydict()
                {'r': [True, False]}
        """
        return StrFunc("like", self._e, pattern=pattern)

    def ilike(self, pattern: str) -> StrFunc:
        """Match a case-insensitive SQL ``LIKE`` pattern (→ Bool).

        Args:
            pattern: A SQL ``LIKE`` pattern using ``%`` and ``_`` wildcards.

        Returns:
            A new Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello", "world"]})
                >>> ds.select(bt.col("s").str.ilike("H%O").alias("r")).to_pydict()
                {'r': [True, False]}
        """
        return StrFunc("ilike", self._e, pattern=pattern)

    def regexp_replace(self, pattern: str, replacement: str) -> StrFunc:
        """Replace only the first regex match with ``replacement`` (``$1`` backrefs).

        Use :meth:`regexp_replace_all` to replace every match.

        Args:
            pattern: The regular expression to match.
            replacement: The replacement text; ``$1``…​ refer to capture groups.

        Returns:
            A new Utf8 expression with the first match replaced.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a1b2"]})
                >>> ds.select(bt.col("s").str.regexp_replace(r"\\d", "X").alias("r")).to_pydict()
                {'r': ['aXb2']}
        """
        return StrFunc("regexp_replace", self._e, pattern=pattern, replacement=replacement)

    def regexp_extract(self, pattern: str, group: int = 0) -> StrFunc:
        """Extract one capture group of the regex; ``''`` if no match.

        Args:
            pattern: The regular expression to match.
            group: Capture group index; 0 (default) is the whole match.

        Returns:
            A new Utf8 expression: the captured group, or ``''``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abc123"]})
                >>> ds.select(bt.col("s").str.regexp_extract(r"(\\d+)", 1).alias("r")).to_pydict()
                {'r': ['123']}
        """
        return StrFunc("regexp_extract", self._e, pattern=pattern, start=group)

    def replace(self, pattern: str, replacement: str) -> StrFunc:
        """Replace every occurrence of the literal ``pattern`` with ``replacement``.

        A plain (non-regex) substring replacement of all matches; use
        :meth:`regexp_replace_all` for a regex.

        Args:
            pattern: The literal substring to find.
            replacement: The literal text to substitute.

        Returns:
            A new Utf8 expression with every match replaced.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> d = bt.from_pydict({"s": ["a-b-c"]})
                >>> d.select(bt.col("s").str.replace("-", "_").alias("r")).to_pydict()
                {'r': ['a_b_c']}
        """
        return StrFunc("replace", self._e, pattern=pattern, replacement=replacement)

    def trim(self, chars: str | None = None) -> StrFunc:
        """Trim from both ends: any of ``chars`` if given, else the ASCII space.

        DuckDB ``trim``; Polars ``strip_chars``. ``chars`` is treated as a set of
        characters to strip, not a prefix/suffix string.

        Args:
            chars: The set of characters to strip; whitespace if omitted.

        Returns:
            A new Utf8 expression: the trimmed string.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["  hi  "]})
                >>> ds.select(bt.col("s").str.trim().alias("r")).to_pydict()
                {'r': ['hi']}
        """
        return StrFunc("trim", self._e, pattern=chars)

    def normalize_whitespace(self) -> StrFunc:
        """Collapse every run of whitespace to a single space and trim the ends.

        The common text-cleanup step for messy free-text columns. Composes
        existing ops (``regexp_replace_all`` + ``trim``), no new IR.

        Returns:
            A new Utf8 expression: the whitespace-normalized string.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> d = bt.from_pydict({"s": ["one two  three"]})
                >>> d.select(
                ...     bt.col("s").str.normalize_whitespace().alias("r")
                ... ).to_pydict()
                {'r': ['one two three']}
        """
        return StrFunc("trim", self.regexp_replace_all(r"\s+", " "))

    def lstrip(self, chars: str | None = None) -> StrFunc:
        """Trim from the left: any of ``chars`` if given, else the ASCII space.

        Args:
            chars: The set of characters to strip; whitespace if omitted.

        Returns:
            A new Utf8 expression: the left-trimmed string.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["  hi  "]})
                >>> ds.select(bt.col("s").str.lstrip().alias("r")).to_pydict()
                {'r': ['hi  ']}
        """
        return StrFunc("l_trim", self._e, pattern=chars)

    def rstrip(self, chars: str | None = None) -> StrFunc:
        """Trim from the right: any of ``chars`` if given, else the ASCII space.

        Args:
            chars: The set of characters to strip; whitespace if omitted.

        Returns:
            A new Utf8 expression: the right-trimmed string.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["  hi  "]})
                >>> ds.select(bt.col("s").str.rstrip().alias("r")).to_pydict()
                {'r': ['  hi']}
        """
        return StrFunc("r_trim", self._e, pattern=chars)

    def split_part(self, delimiter: str, n: int) -> StrFunc:
        """Return the ``n``-th field (1-based) after splitting on ``delimiter``.

        Yields ``''`` when ``n`` is out of range (DuckDB/Spark ``split_part``).

        Args:
            delimiter: The literal separator to split on.
            n: 1-based index of the field to return.

        Returns:
            A new Utf8 expression: the selected field, or ``''``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a-b-c"]})
                >>> ds.select(bt.col("s").str.split_part("-", 2).alias("r")).to_pydict()
                {'r': ['b']}
        """
        n = require_int(n, func="str.split_part", arg="n")
        return StrFunc("split_part", self._e, pattern=delimiter, start=n)

    def regexp_replace_all(self, pattern: str, replacement: str) -> StrFunc:
        """Replace every regex match of ``pattern`` with ``replacement``.

        DuckDB ``regexp_replace(..., 'g')``; Polars ``replace_all``; ``$1`` backrefs.

        Args:
            pattern: The regular expression to match.
            replacement: The replacement text; ``$1``…​ refer to capture groups.

        Returns:
            A new Utf8 expression with every match replaced.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a1b2"]})
                >>> r = bt.col("s").str.regexp_replace_all(r"\\d", "X")
                >>> ds.select(r.alias("r")).to_pydict()
                {'r': ['aXbX']}
        """
        return StrFunc("regexp_replace_all", self._e, pattern=pattern, replacement=replacement)

    def initcap(self) -> StrFunc:
        """Title-case each word: uppercase its first letter, lowercase the rest (``initcap``).

        A word starts after whitespace or punctuation, so ``"a-b c"`` → ``"A-B C"``;
        null → null.

        Returns:
            A new Utf8 expression with each word title-cased.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello world"]})
                >>> ds.select(bt.col("s").str.initcap().alias("r")).to_pydict()
                {'r': ['Hello World']}
        """
        return StrFunc("initcap", self._e)

    def octet_length(self) -> StrFunc:
        """Count the UTF-8 bytes, not characters, in the string (→ Int64).

        Differs from :meth:`len` (character count) for multi-byte text; null → null.

        Returns:
            A new Int64 expression: the byte length of each string.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["héllo"]})
                >>> ds.select(bt.col("s").str.octet_length().alias("r")).to_pydict()
                {'r': [6]}
        """
        return StrFunc("octet_length", self._e)

    def bit_length(self) -> StrFunc:
        """Count the bits in the string, i.e. UTF-8 bytes times 8 (→ Int64); null → null.

        Returns:
            A new Int64 expression: the bit length.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abc"]})
                >>> ds.select(bt.col("s").str.bit_length().alias("r")).to_pydict()
                {'r': [24]}
        """
        return StrFunc("bit_length", self._e)

    def hex(self) -> StrFunc:
        """Encode the UTF-8 bytes as uppercase hexadecimal; inverse of :meth:`unhex` (→ Utf8).

        Returns:
            A new Utf8 expression: the uppercase hex encoding.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abc"]})
                >>> ds.select(bt.col("s").str.hex().alias("r")).to_pydict()
                {'r': ['616263']}
        """
        return StrFunc("hex", self._e)

    def base64(self) -> StrFunc:
        """Encode the UTF-8 bytes as standard base64; inverse of :meth:`from_base64` (→ Utf8).

        Returns:
            A new Utf8 expression: the base64 encoding.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["abc"]})
                >>> ds.select(bt.col("s").str.base64().alias("r")).to_pydict()
                {'r': ['YWJj']}
        """
        return StrFunc("base64", self._e)

    def from_base64(self) -> StrFunc:
        """Decode standard base64 to a UTF-8 string; null if invalid or null (→ Utf8).

        Returns:
            A new Utf8 expression: the decoded string, or null.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["YWJj"]})
                >>> ds.select(bt.col("s").str.from_base64().alias("r")).to_pydict()
                {'r': ['abc']}
        """
        return StrFunc("from_base64", self._e)

    def unhex(self) -> StrFunc:
        """Decode pairs of hex digits to a UTF-8 string; null if invalid or null (→ Utf8).

        Returns:
            A new Utf8 expression: the decoded string, or null.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["616263"]})
                >>> ds.select(bt.col("s").str.unhex().alias("r")).to_pydict()
                {'r': ['abc']}
        """
        return StrFunc("unhex", self._e)

    def translate(self, from_chars: str, to_chars: str) -> StrFunc:
        """Map each character in ``from_chars`` to the one at the same index of ``to_chars``.

        SQL/DuckDB ``translate``: characters in ``from_chars`` beyond ``to_chars``'s
        length are deleted; characters not in ``from_chars`` pass through unchanged.

        Args:
            from_chars: Characters to map from.
            to_chars: Characters to map to, positionally; shorter than
                ``from_chars`` deletes the surplus.

        Returns:
            A new Utf8 expression with characters mapped.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["aabbcc"]})
                >>> ds.select(bt.col("s").str.translate("abc", "xyz").alias("r")).to_pydict()
                {'r': ['xxyyzz']}
        """
        return StrFunc("translate", self._e, pattern=from_chars, replacement=to_chars)


# Parameterless string→string transforms: accessor name → engine `StrFunc` tag.
# (`trim`/`lstrip`/`rstrip` are explicit methods — they take an optional char set.)
_STR_TRANSFORMS = {
    "upper": "upper",
    "lower": "lower",
    "reverse": "reverse",
}


def _str_transform_doc(name: str) -> str:
    """Fallback docstring for a ``.str`` transform without a curated entry.

    ``upper``/``lower`` carry curated entries; only ``reverse`` falls through to
    here, so the summary and example reflect a character-reversing transform.
    """
    return (
        f"Return each string with its characters {name}d.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"s": ["Hello"]})\n'
        f'        >>> ds.select(r=bt.col("s").str.{name}()).to_pydict()\n'
        "        {'r': ['olleH']}"
    )


_bind_accessors(
    _StrNamespace,
    _STR_TRANSFORMS,
    lambda e, t: StrFunc(t, e),
    _str_transform_doc,
    "A new string :class:`~batcher.Expr` with the transform applied.",
)
