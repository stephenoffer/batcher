"""The `.str` accessor namespace.

`col("s").str.upper()`, `.str.md5()`, `.str.regexp_extract_all(...)`, … — each
method is a thin builder over a `bc-expr` `StrFunc` node. The parameterless
string→string transforms are generated from `_STR_TRANSFORMS` (data, not code).
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Cast, Expr
from batcher.plan.expr_ir.func_nodes import StrFunc, Strptime
from batcher.plan.expr_ir.namespaces._bind import _bind_accessors


class _StrNamespace:
    """String functions: ``col("s").str.upper()``, ``.str.contains("x")``, …

    The parameterless string→string transforms are **data, not code**
    (``_STR_TRANSFORMS``: accessor name → ``bc-expr`` ``StrFunc`` tag) and are
    generated below — adding one is a single table entry. The functions that take
    arguments (search / slice / replace) and ``len`` (returns Int64) stay explicit.

    Every method returns a new lazy :class:`Expr`; null inputs propagate to null
    outputs throughout.
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        """Wrap the parent :class:`Expr` so its `.str` methods can build on it."""
        self._e = e

    def len(self) -> StrFunc:
        """Count the characters in the string (→ Int64).

        Counts Unicode characters, not bytes (see :meth:`octet_length`). Null → null.

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

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello"]})
                >>> ds.select(bt.col("s").str.substr(2, 3).alias("r")).to_pydict()
                {'r': ['ell']}
        """
        return StrFunc("substr", self._e, start=start, length=length)

    def left(self, n: int) -> StrFunc:
        """Take the first ``n`` characters (SQL ``left``) — a 1-based ``substr``.

        Args:
            n: Number of leading characters to keep.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello"]})
                >>> ds.select(bt.col("s").str.left(3).alias("r")).to_pydict()
                {'r': ['hel']}
        """
        return StrFunc("substr", self._e, start=1, length=n)

    def repeat(self, n: int) -> StrFunc:
        """Concatenate ``n`` copies of the string.

        Args:
            n: Repeat count; ``n`` ≤ 0 yields an empty string.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ab"]})
                >>> ds.select(bt.col("s").str.repeat(3).alias("r")).to_pydict()
                {'r': ['ababab']}
        """
        return StrFunc("repeat", self._e, start=n)

    def lpad(self, width: int, fill: str = " ") -> StrFunc:
        """Left-pad the string to ``width`` characters, truncating if longer.

        Args:
            width: Target character width.
            fill: Pad string, cycled as needed; defaults to a space.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ab"]})
                >>> ds.select(bt.col("s").str.lpad(5, "*").alias("r")).to_pydict()
                {'r': ['***ab']}
        """
        return StrFunc("lpad", self._e, start=width, pattern=fill)

    def rpad(self, width: int, fill: str = " ") -> StrFunc:
        """Right-pad the string to ``width`` characters, truncating if longer.

        Args:
            width: Target character width.
            fill: Pad string, cycled as needed; defaults to a space.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ab"]})
                >>> ds.select(bt.col("s").str.rpad(5, "*").alias("r")).to_pydict()
                {'r': ['ab***']}
        """
        return StrFunc("rpad", self._e, start=width, pattern=fill)

    def position(self, pattern: str) -> StrFunc:
        """Find the 1-based index of ``pattern`` in the string, or 0 if absent.

        Returns Int64.

        Args:
            pattern: The literal substring to locate.

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

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a.b.c.d"]})
                >>> ds.select(bt.col("s").str.substring_index(".", 2).alias("r")).to_pydict()
                {'r': ['a.b']}
        """
        return StrFunc("substring_index", self._e, pattern=delimiter, start=count)

    def overlay(self, replacement: str, pos: int, length: int | None = None) -> StrFunc:
        """Replace ``length`` characters from 1-based ``pos`` with ``replacement``.

        SQL ``OVERLAY``: ``length`` defaults to the replacement's length. Returns
        Utf8.

        Args:
            replacement: The string to splice in.
            pos: 1-based index where the replacement begins.
            length: Characters to overwrite; defaults to ``len(replacement)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello"]})
                >>> ds.select(bt.col("s").str.overlay("XY", 2).alias("r")).to_pydict()
                {'r': ['hXYlo']}
        """
        return StrFunc("overlay", self._e, replacement=replacement, start=pos, length=length)

    def regexp_extract_all(self, pattern: str) -> StrFunc:
        """Collect every regex match as a list of strings (DuckDB ``regexp_extract_all``).

        Returns an empty list when there are no matches. Chain ``.list`` to operate
        on the result. Returns List<Utf8>.

        Args:
            pattern: The regular expression to match.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> d = bt.from_pydict({"s": ["2024-01-15"]})
                >>> d.select(
                ...     bt.col("s").str.regexp_extract_all(r"\\d+").alias("r")
                ... ).to_pydict()
                {'r': [['2024', '01', '15']]}
        """
        return StrFunc("regexp_extract_all", self._e, pattern=pattern)

    def regexp_count(self, pattern: str) -> StrFunc:
        """Count non-overlapping regex matches (DuckDB ``regexp_count``).

        Returns Int64.

        Args:
            pattern: The regular expression to match.

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

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["sitting"]})
                >>> ds.select(bt.col("s").str.levenshtein("kitten").alias("r")).to_pydict()
                {'r': [3]}
        """
        return StrFunc("levenshtein", self._e, pattern=target)

    def soundex(self) -> StrFunc:
        """Compute the American Soundex phonetic code, a 4-character key.

        Groups words that sound alike (DuckDB ``soundex``). Returns Utf8.

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

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello"]})
                >>> ds.select(bt.col("s").str.right(3).alias("r")).to_pydict()
                {'r': ['llo']}
        """
        return StrFunc("right", self._e, start=n)

    def ascii(self) -> StrFunc:
        """Return the Unicode codepoint of the first character, 0 if empty (→ Int64).

        Despite the name, returns the full codepoint, not just ASCII.

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

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a-b-c"]})
                >>> ds.select(bt.col("s").str.split("-").alias("r")).to_pydict()
                {'r': [['a', 'b', 'c']]}
        """
        return StrFunc("split", self._e, pattern=delimiter)

    def regexp_matches(self, pattern: str) -> StrFunc:
        """Test whether the regex ``pattern`` matches anywhere in the string (→ Bool).

        An unanchored search; see :meth:`like` for SQL wildcard matching.

        Args:
            pattern: The regular expression to test.

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

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> d = bt.from_pydict({"s": ["a-b-c"]})
                >>> d.select(bt.col("s").str.replace("-", "_").alias("r")).to_pydict()
                {'r': ['a_b_c']}
        """
        return StrFunc("replace", self._e, pattern=pattern, replacement=replacement)

    def trim(self, chars: str | None = None) -> StrFunc:
        """Trim from both ends: any of ``chars`` if given, else whitespace.

        DuckDB ``trim``; Polars ``strip_chars``. ``chars`` is treated as a set of
        characters to strip, not a prefix/suffix string.

        Args:
            chars: The set of characters to strip; whitespace if omitted.

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
        """Trim from the left: any of ``chars`` if given, else whitespace.

        Args:
            chars: The set of characters to strip; whitespace if omitted.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["  hi  "]})
                >>> ds.select(bt.col("s").str.lstrip().alias("r")).to_pydict()
                {'r': ['hi  ']}
        """
        return StrFunc("l_trim", self._e, pattern=chars)

    def rstrip(self, chars: str | None = None) -> StrFunc:
        """Trim from the right: any of ``chars`` if given, else whitespace.

        Args:
            chars: The set of characters to strip; whitespace if omitted.

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

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["a-b-c"]})
                >>> ds.select(bt.col("s").str.split_part("-", 2).alias("r")).to_pydict()
                {'r': ['b']}
        """
        return StrFunc("split_part", self._e, pattern=delimiter, start=n)

    def regexp_replace_all(self, pattern: str, replacement: str) -> StrFunc:
        """Replace every regex match of ``pattern`` with ``replacement``.

        DuckDB ``regexp_replace(..., 'g')``; Polars ``replace_all``; ``$1`` backrefs.

        Args:
            pattern: The regular expression to match.
            replacement: The replacement text; ``$1``…​ refer to capture groups.

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
        """Uppercase each word's first letter, lowercasing the rest; a word starts after
        whitespace or punctuation, so ``"a-b c"`` → ``"A-B C"`` (``initcap``; null → null).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["hello world"]})
                >>> ds.select(bt.col("s").str.initcap().alias("r")).to_pydict()
                {'r': ['Hello World']}
        """
        return StrFunc("initcap", self._e)

    def octet_length(self) -> StrFunc:
        """Count the UTF-8 bytes (not characters) in the string (→ Int64); differs from
        :meth:`len` (character count) for multi-byte text; null → null.

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


_bind_accessors(
    _StrNamespace,
    _STR_TRANSFORMS,
    lambda e, t: StrFunc(t, e),
    lambda n: f"Return the string with {n} applied.",
)
