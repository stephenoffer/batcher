"""The `.list`/`.arr`, `.struct`, and `.json` accessor namespaces.

Each method is a thin builder over a `bc-expr` list/struct/json node. The
parameterless list reductions are generated from `_LIST_FUNCS` (data, not code).
"""

from __future__ import annotations

from typing import Any

from batcher.plan.expr_ir.core import Expr, _wrap
from batcher.plan.expr_ir.func_nodes import (
    ListBinary,
    ListContains,
    ListFilter,
    ListFunc,
    ListGet,
    ListPosition,
    ListSet,
    ListSlice,
    ListTransform,
    MapFunc,
    StrFunc,
    StructField,
)
from batcher.plan.expr_ir.namespaces._bind import _bind_accessors
from batcher.plan.expr_ir.nodes import ListJoin


class _StructNamespace:
    """Struct accessors: ``col("s").struct.field("x")``."""

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        self._e = e

    def field(self, name: str) -> StructField:
        """Extract the named field from a struct column as its own column.

        The result keeps the field's own type and per-row nulls. Selecting a name
        that is not in the struct's schema is a plan-build error.

        Args:
            name: The struct field to project out.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]})
                >>> ds.select(bt.col("s").struct.field("x").alias("x")).to_pydict()
                {'x': [1, 2]}
        """
        return StructField(self._e, name)


class _JsonNamespace:
    """JSON accessors on a string column: ``col("j").json.extract_string("$.a.b")``."""

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        self._e = e

    def extract_string(self, path: str) -> StrFunc:
        """Read the value at a JSON path as text (→ Utf8); null if the path is absent.

        The column holds JSON-encoded text. A value that is itself an object or array
        is returned as its compact JSON serialization (e.g. ``{"b":7}``), not unwrapped.

        Args:
            path: A JSONPath, e.g. ``"$.a.b"`` or ``"$.items[0]"``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"j": ['{"a": {"b": 7}}', "{}"]})
                >>> ds.select(bt.col("j").json.extract_string("$.a").alias("r")).to_pydict()
                {'r': ['{"b":7}', None]}
        """
        return StrFunc("json_extract_string", self._e, pattern=path)

    def extract_int(self, path: str) -> StrFunc:
        """Read the value at a JSON path as an integer (→ Int64); null if absent or non-integral.

        Args:
            path: A JSONPath, e.g. ``"$.a.b"``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"j": ['{"a": {"b": 7}}', "{}"]})
                >>> ds.select(bt.col("j").json.extract_int("$.a.b").alias("r")).to_pydict()
                {'r': [7, None]}
        """
        return StrFunc("json_extract_int", self._e, pattern=path)

    def extract_float(self, path: str) -> StrFunc:
        """Read the value at a JSON path as a float (→ Float64); null if absent or non-numeric.

        Args:
            path: A JSONPath, e.g. ``"$.price"``.
        """
        return StrFunc("json_extract_float", self._e, pattern=path)

    def extract_bool(self, path: str) -> StrFunc:
        """Read the value at a JSON path as a boolean (→ Boolean); null if absent or non-boolean.

        Args:
            path: A JSONPath, e.g. ``"$.active"``.
        """
        return StrFunc("json_extract_bool", self._e, pattern=path)


class _MapNamespace:
    """Map-column accessors: ``col("m").map.keys()``, ``.values()``, ``.get(key)``.

    For an Arrow ``Map`` column (``map<K, V>``). `keys`/`values` return `List`
    columns; `get(key)`/`element_at(key)` look up the value for a literal key (null
    if absent).
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        self._e = e

    def keys(self) -> MapFunc:
        """Return each row's map keys as a ``List`` column (DuckDB ``map_keys``).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import pyarrow as pa
                >>> col = pa.array([[("a", 1), ("b", 2)], [("c", 3)]],
                ...                type=pa.map_(pa.string(), pa.int64()))
                >>> ds = bt.from_arrow(pa.table({"m": col}))
                >>> ds.select(bt.col("m").map.keys().alias("k")).to_pydict()
                {'k': [['a', 'b'], ['c']]}
        """
        return MapFunc("map_keys", self._e)

    def values(self) -> MapFunc:
        """Return each row's map values as a ``List`` column (DuckDB ``map_values``).

        Keys and values stay positionally aligned with :meth:`keys`.
        """
        return MapFunc("map_values", self._e)

    def get(self, key: object) -> MapFunc:
        """Look up the value for a literal ``key`` in each row's map; null if absent.

        SQL ``element_at``. ``key`` is a plan-time literal, not an expression.

        Args:
            key: The map key to look up in every row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import pyarrow as pa
                >>> col = pa.array([[("a", 1), ("b", 2)], [("c", 3)]],
                ...                type=pa.map_(pa.string(), pa.int64()))
                >>> ds = bt.from_arrow(pa.table({"m": col}))
                >>> ds.select(bt.col("m").map.get("a").alias("v")).to_pydict()
                {'v': [1, None]}
        """
        return MapFunc("element_at", self._e, key=key)

    # SQL spelling of `get`.
    element_at = get


class _EmbeddingNamespace:
    """Embedding-vector operations: ``col("v").embedding.cosine_distance(other)``, …

    A Daft-style accessor for vector columns (fixed-size or variable lists). Thin
    sugar over the same `ListBinary` vector kernels as `.list`, named for the
    retrieval/RAG use case where similarity and distance are the natural verbs.
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        self._e = e

    def cosine_similarity(self, other: Any) -> ListBinary:
        """Cosine similarity with another embedding column, in ``[-1, 1]`` (→ Float64).

        Args:
            other: The other embedding column (or an ``array(...)`` literal), paired
                element-wise; both vectors must have the same length.
        """
        return ListBinary("cosine_similarity", self._e, _wrap(other))

    def cosine_distance(self, other: Any) -> Expr:
        """Cosine distance ``1 - cosine_similarity`` (→ Float64).

        The common nearest-neighbour ranking metric for embeddings: 0 for identical
        direction, 1 for orthogonal, 2 for opposite.

        Args:
            other: The other embedding column (or an ``array(...)`` literal).
        """
        return 1.0 - ListBinary("cosine_similarity", self._e, _wrap(other))

    def l2_distance(self, other: Any) -> ListBinary:
        """Euclidean (L2) distance to another embedding column (→ Float64).

        Args:
            other: The other embedding column (or an ``array(...)`` literal).
        """
        return ListBinary("l2_distance", self._e, _wrap(other))

    def dot(self, other: Any) -> ListBinary:
        """Dot product with another embedding column (→ Float64).

        Args:
            other: The other embedding column (or an ``array(...)`` literal).
        """
        return ListBinary("dot", self._e, _wrap(other))


class _ListNamespace:
    """List/array reductions: ``col("a").list.len()``, ``.list.sum()``, …

    Generated from ``_LIST_FUNCS`` (accessor name → ``bc-expr`` ``ListFunc`` tag) —
    a single table entry adds a reduction. `get` carries an index, so it is explicit.
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        self._e = e

    def get(self, index: int) -> ListGet:
        """Return the element at ``index`` of each list; null if out of range.

        Negative indices count from the end, ``get(-1)`` being the last element
        (Polars/Python indexing). A null or empty list yields null.

        Args:
            index: 0-based position; negatives index from the end.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[3, 1, 2], [], None]})
                >>> ds.select(bt.col("a").list.get(-1).alias("r")).to_pydict()
                {'r': [2, None, None]}
        """
        return ListGet(self._e, index)

    def first(self) -> ListGet:
        """Return the first element of each list (null for an empty or null list)."""
        return ListGet(self._e, 0)

    def last(self) -> ListGet:
        """Return the last element of each list (null for an empty or null list)."""
        return ListGet(self._e, -1)

    def contains(self, value: int | float | bool | str) -> ListContains:
        """Test whether any element of each list equals ``value`` (→ Bool).

        An empty list is ``False``; a null list is null.

        Args:
            value: The literal to search for.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[3, 1, 2], [], None]})
                >>> ds.select(bt.col("a").list.contains(1).alias("r")).to_pydict()
                {'r': [True, False, None]}
        """
        return ListContains(self._e, value)

    def position(self, value: int | float | bool | str) -> ListPosition:
        """Return the 1-based index of the first element equal to ``value``; null if absent.

        DuckDB ``list_position`` (→ Int64). The first matching element is index 1.

        Args:
            value: The literal to locate.
        """
        return ListPosition(self._e, value)

    def intersect(self, other: Any) -> ListSet:
        """The distinct elements present in **both** this list and ``other`` (Spark
        ``array_intersect``), in this list's order. → List."""
        return ListSet("array_intersect", self._e, _wrap(other))

    def difference(self, other: Any) -> ListSet:
        """The distinct elements in this list but **not** in ``other`` (Spark
        ``array_except``), in this list's order. → List."""
        return ListSet("array_except", self._e, _wrap(other))

    def union(self, other: Any) -> ListSet:
        """The distinct elements in **either** this list or ``other`` (Spark
        ``array_union``) — this list's distinct elements followed by the new ones from
        ``other``. → List."""
        return ListSet("array_union", self._e, _wrap(other))

    def transform(self, func: Any) -> ListTransform:
        """Apply `func` to every element, preserving lengths (DuckDB ``list_transform``;
        Polars ``list.eval``). `func` is an expression over ``element()`` (the current
        element), e.g. ``col("a").list.transform(element() * 2)``. → List."""
        return ListTransform(self._e, _wrap(func))

    def filter(self, predicate: Any) -> ListFilter:
        """Keep the elements where `predicate` (an expression over ``element()``) is
        true (DuckDB ``list_filter``), e.g. ``col("a").list.filter(element() > 0)``.
        → List."""
        return ListFilter(self._e, _wrap(predicate))

    def slice(self, offset: int, length: int | None = None) -> ListSlice:
        """Return the 0-based sub-range ``[offset, offset+length)`` of each list.

        With no ``length`` the slice runs to the end of the list. A null list stays
        null; an empty list stays empty.

        Args:
            offset: 0-based start index.
            length: Number of elements to take; ``None`` means to the end.
        """
        return ListSlice(self._e, offset, length)

    def join(self, separator: str) -> ListJoin:
        """Concatenate each list's elements into one string, joined by ``separator``.

        Elements are cast to text and null elements are skipped. A null or empty
        list yields null (→ Utf8).

        Args:
            separator: The text inserted between consecutive elements.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [["x", "y", "z"], ["q"]]})
                >>> ds.select(bt.col("a").list.join("-").alias("r")).to_pydict()
                {'r': ['x-y-z', 'q']}
        """
        return ListJoin(self._e, separator)

    def flatten(self) -> ListFunc:
        """Concatenate a list-of-lists into one list per row, preserving order.

        DuckDB ``flatten``: one level of nesting is removed. Null inner lists are
        skipped; a null row stays null.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[[1, 2], [3]], [[4]]]})
                >>> ds.select(bt.col("a").list.flatten().alias("r")).to_pydict()
                {'r': [[1, 2, 3], [4]]}
        """
        return ListFunc("flatten", self._e)

    def dot(self, other: Any) -> ListBinary:
        """Dot product with another vector column, paired element-wise (→ Float64).

        The unnormalized similarity score. Both vectors must have the same length.

        Args:
            other: The other vector column (or an ``array(...)`` literal).
        """
        return ListBinary("dot", self._e, _wrap(other))

    def cosine_similarity(self, other: Any) -> ListBinary:
        """Cosine similarity with another vector column, in ``[-1, 1]`` (→ Float64).

        The standard embedding-similarity score for retrieval / RAG; null if either
        vector has zero magnitude. Both vectors must have the same length.

        Args:
            other: The other vector column (or an ``array(...)`` literal).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[1.0, 1.0]], "b": [[0.0, 1.0]]})
                >>> r = ds.select(bt.col("a").list.cosine_similarity(bt.col("b")).alias("r"))
                >>> round(r.to_pydict()["r"][0], 4)
                0.7071
        """
        return ListBinary("cosine_similarity", self._e, _wrap(other))

    def l2_distance(self, other: Any) -> ListBinary:
        """Euclidean (L2) distance to another vector column (→ Float64).

        The metric for nearest-neighbour vector search. Both vectors must have the
        same length.

        Args:
            other: The other vector column (or an ``array(...)`` literal).
        """
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
    "normalize": "normalize",  # L2-normalize to unit length (→ list); embedding prep
}


_bind_accessors(
    _ListNamespace,
    _LIST_FUNCS,
    lambda e, t: ListFunc(t, e),
    lambda n: f"Per-row {n} over each list value.",
)
