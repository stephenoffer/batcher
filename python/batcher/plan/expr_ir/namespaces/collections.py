"""The `.list`, `.struct`, `.json`, and `.map` accessor namespaces.

Each method is a thin builder over a `bc-expr` list/struct/json node. The
parameterless list reductions are generated from `_LIST_FUNCS` (data, not code).
"""

from __future__ import annotations

from typing import Any

from batcher._internal.errors import PlanError, require_int
from batcher.plan.expr_ir.compat.guidance import LIST_UNSUPPORTED, accessor_attribute_error
from batcher.plan.expr_ir.core import Expr, Lit, _wrap
from batcher.plan.expr_ir.func_nodes import (
    ListBinary,
    ListContains,
    ListFilter,
    ListFunc,
    ListGet,
    ListPosition,
    ListSet,
    ListSimhash,
    ListSlice,
    ListTransform,
    ListZip,
    MapFunc,
    StrFunc,
    StructField,
)
from batcher.plan.expr_ir.namespaces._bind import _bind_accessors
from batcher.plan.expr_ir.nodes import ListJoin


class _StructNamespace:
    """Struct accessors: ``col("s").struct.field("x")``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"s": [{"x": 1, "y": "a"}]})
            >>> ds.select(bt.col("s").struct.field("x").alias("x")).to_pydict()
            {'x': [1]}
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        """Wrap the parent :class:`Expr` so its `.struct` methods can build on it."""
        self._e = e

    def __repr__(self) -> str:
        """Show the accessor and its parent, e.g. ``<.struct accessor of col('c')>``."""
        return f"<.struct accessor of {self._e!r}>"

    def field(self, name: str) -> StructField:
        """Extract the named field from a struct column as its own column.

        The result keeps the field's own type and per-row nulls. Selecting a name
        that is not in the struct's schema is a plan-build error.

        Args:
            name: The struct field to project out.

        Returns:
            A new expression: the projected field, keeping its type and per-row nulls.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]})
                >>> ds.select(bt.col("s").struct.field("x").alias("x")).to_pydict()
                {'x': [1, 2]}
        """
        return StructField(self._e, name)

    def keys(self) -> MapFunc:
        """Return the struct's field names as a ``List`` column (DuckDB ``struct_keys``).

        A struct's keys come from its *type*, so every row carries the same list. A null
        struct row still answers null rather than the names, which is what distinguishes
        this from a constant.

        Returns:
            A new List<Utf8> expression of the struct's field names.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]})
                >>> ds.select(bt.col("s").struct.keys().alias("k")).to_pydict()
                {'k': [['x', 'y'], ['x', 'y']]}
        """
        return MapFunc("map_keys", self._e)

    def get(self, name: str) -> MapFunc:
        """Return the named field, the subscript spelling of :meth:`field`.

        This is what ``s['x']`` lowers to, and what DuckDB's ``struct_extract`` and
        Spark's ``element_at(s, 'x')`` reach. Naming a field the struct does not have is
        an error rather than a null, because a struct's fields are fixed by its type.

        Args:
            name: The struct field to project out.

        Returns:
            A new expression: the named field, under the struct's own row nulls.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]})
                >>> ds.select(bt.col("s").struct.get("x").alias("x")).to_pydict()
                {'x': [1, 2]}
        """
        return MapFunc("element_at", self._e, key=name)


class _JsonNamespace:
    """JSON accessors on a string column: ``col("j").json.extract_string("$.a.b")``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"j": ['{"a": {"b": 7}}', "{}"]})
            >>> ds.select(bt.col("j").json.extract_int("$.a.b").alias("r")).to_pydict()
            {'r': [7, None]}
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        """Wrap the parent :class:`Expr` so its `.json` methods can build on it."""
        self._e = e

    def __repr__(self) -> str:
        """Show the accessor and its parent, e.g. ``<.json accessor of col('c')>``."""
        return f"<.json accessor of {self._e!r}>"

    def extract_string(self, path: str) -> StrFunc:
        """Read the value at a JSON path as text (→ Utf8); null if the path is absent.

        The column holds JSON-encoded text. A value that is itself an object or array
        is returned as its compact JSON serialization (e.g. ``{"b":7}``), not unwrapped.

        Args:
            path: A JSONPath, e.g. ``"$.a.b"`` or ``"$.items[0]"``.

        Returns:
            A new Utf8 expression, or null if the path is absent.

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

        Returns:
            A new Int64 expression, or null if absent or non-integral.

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

        Returns:
            A new Float64 expression, or null if absent or non-numeric.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"j": ['{"p": 3.5}', "{}"]})
                >>> ds.select(bt.col("j").json.extract_float("$.p").alias("r")).to_pydict()
                {'r': [3.5, None]}
        """
        return StrFunc("json_extract_float", self._e, pattern=path)

    def extract_bool(self, path: str) -> StrFunc:
        """Read the value at a JSON path as a boolean (→ Boolean); null if absent or non-boolean.

        Args:
            path: A JSONPath, e.g. ``"$.active"``.

        Returns:
            A new Boolean expression, or null if absent or non-boolean.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"j": ['{"active": true}', "{}"]})
                >>> ds.select(bt.col("j").json.extract_bool("$.active").alias("r")).to_pydict()
                {'r': [True, None]}
        """
        return StrFunc("json_extract_bool", self._e, pattern=path)

    def array_length(self, path: str = "$") -> StrFunc:
        """Count the elements of the JSON array at `path` (→ Int64).

        The count is taken by skipping over each element structurally, so a long array
        of large objects costs one pass over its bytes and parses none of them.

        Args:
            path: A JSONPath to the array; the document root by default.

        Returns:
            A new Int64 expression, or null if the path is absent or the value there is
            not an array.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"j": ['{"xs": [1, 2, 3]}', '{"xs": 4}']})
                >>> ds.select(r=bt.col("j").json.array_length("$.xs")).to_pydict()
                {'r': [3, None]}
        """
        return StrFunc("json_array_length", self._e, pattern=path)

    def keys(self, path: str = "$") -> StrFunc:
        """List the keys of the JSON object at `path`, in source order (→ List<Utf8>).

        Source order, not sorted order, so the keys line up with the document as written.

        Args:
            path: A JSONPath to the object; the document root by default.

        Returns:
            A new List<Utf8> expression, or a null list if the path is absent or the
            value there is not an object.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"j": ['{"z": 1, "a": 2}', "[]"]})
                >>> ds.select(r=bt.col("j").json.keys()).to_pydict()
                {'r': [['z', 'a'], None]}
        """
        return StrFunc("json_object_keys", self._e, pattern=path)

    def values(self, path: str = "$") -> StrFunc:
        """Read the JSON array at `path` as a list of texts (→ List<Utf8>).

        Each element is rendered the way :meth:`extract_string` renders a value: a string
        element verbatim, an object or array as its compact JSON, a JSON ``null`` as a
        null element. This is the bridge from a JSON array column to a Batcher list
        column, so :meth:`~batcher.Dataset.explode` and the ``.list`` namespace apply.

        Args:
            path: A JSONPath to the array; the document root by default.

        Returns:
            A new List<Utf8> expression, or a null list if the path is absent or the
            value there is not an array.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"j": ['{"xs": ["a", 1, {"b": 2}]}']})
                >>> ds.select(r=bt.col("j").json.values("$.xs")).to_pydict()
                {'r': [['a', '1', '{"b":2}']]}
        """
        return StrFunc("json_array_values", self._e, pattern=path)

    def type_of(self, path: str = "$") -> StrFunc:
        """Name the JSON type at `path` (→ Utf8).

        One of ``object``, ``array``, ``string``, ``number``, ``boolean``, or ``null``.
        Use it to route a heterogeneous field before extracting it, rather than
        extracting into every type and coalescing.

        Args:
            path: A JSONPath; the document root by default.

        Returns:
            A new Utf8 expression, or null if the path is absent.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"j": ['{"v": [1]}', '{"v": "x"}', "{}"]})
                >>> ds.select(r=bt.col("j").json.type_of("$.v")).to_pydict()
                {'r': ['array', 'string', None]}
        """
        return StrFunc("json_type", self._e, pattern=path)

    def value(self, path: str) -> StrFunc:
        """The **scalar** at `path` as its JSON token, null for a container (→ Utf8).

        DuckDB ``json_value``. The difference from :meth:`extract_string` is deliberate
        and is DuckDB's: this one answers only for a scalar and keeps a string's quotes
        (it returns the JSON token), where ``extract_string`` unquotes a string and
        renders an object or array as compact JSON.

        Args:
            path: A JSONPath, e.g. ``"$.user.age"``.

        Returns:
            A new Utf8 expression; null at a container, an absent path, or a JSON null.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"j": ['{"a": 1, "b": "x", "c": [1]}']})
                >>> ds.select(r=bt.col("j").json.value("$.b")).to_pydict()
                {'r': ['"x"']}
        """
        return StrFunc("json_value", self._e, pattern=path)

    def contains(self, value: str) -> StrFunc:
        """Whether the document contains `value` as an element or field value (→ Boolean).

        DuckDB ``json_contains``. `value` is itself JSON, so a string needs its quotes.
        Comparison is on the parsed values, so whitespace and key order cannot change
        the answer.

        Args:
            value: The JSON text to look for, e.g. ``"1"`` or ``'"needle"'``.

        Returns:
            A new Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"j": ["[1, 2, 3]", '{"a": 5}']})
                >>> ds.select(r=bt.col("j").json.contains("5")).to_pydict()
                {'r': [False, True]}
        """
        return StrFunc("json_contains", self._e, pattern=value)

    def pretty(self) -> StrFunc:
        """Re-render the document with four-space indentation (→ Utf8).

        DuckDB ``json_pretty``. Text that is not valid JSON is null rather than an error.

        Returns:
            A new Utf8 expression: the indented document.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"j": ['{"a":1}']})
                >>> print(ds.select(r=bt.col("j").json.pretty()).to_pydict()["r"][0])
                {
                    "a": 1
                }
        """
        return StrFunc("json_pretty", self._e)

    def structure(self) -> StrFunc:
        """The document's shape with each leaf replaced by its type name (→ Utf8).

        DuckDB ``json_structure``: the schema-on-read summary to group by when you are
        finding out what shapes a JSON column actually holds. An array is described by
        its first element, as in DuckDB.

        Returns:
            A new Utf8 expression: the structure document.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"j": ['{"a": 1, "b": "x"}']})
                >>> ds.select(r=bt.col("j").json.structure()).to_pydict()
                {'r': ['{"a":"UBIGINT","b":"VARCHAR"}']}
        """
        return StrFunc("json_structure", self._e)

    def exists(self, path: str) -> StrFunc:
        """Test whether a value exists at `path` (→ Boolean).

        A JSON ``null`` counts as present. That is the distinction the ``extract_*``
        methods cannot express, since an absent path and a JSON ``null`` both extract to
        SQL null, and the two mean different things in a schema-on-read pipeline.

        Args:
            path: A JSONPath, e.g. ``"$.user.email"``.

        Returns:
            A new Boolean expression; null only where the input itself is null.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"j": ['{"v": null}', "{}"]})
                >>> ds.select(r=bt.col("j").json.exists("$.v")).to_pydict()
                {'r': [True, False]}
        """
        return StrFunc("json_exists", self._e, pattern=path)


class _MapNamespace:
    """Map-column accessors: ``col("m").map.keys()``, ``.values()``, ``.get(key)``.

    For an Arrow ``Map`` column (``map<K, V>``). `keys`/`values` return `List`
    columns; `get(key)` looks up the value for a literal key (null if absent).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import pyarrow as pa
            >>> col = pa.array([[("a", 1), ("b", 2)]], type=pa.map_(pa.string(), pa.int64()))
            >>> ds = bt.from_arrow(pa.table({"m": col}))
            >>> ds.select(bt.col("m").map.keys().alias("k")).to_pydict()
            {'k': [['a', 'b']]}
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        """Wrap the parent :class:`Expr` so its `.map` methods can build on it."""
        self._e = e

    def __repr__(self) -> str:
        """Show the accessor and its parent, e.g. ``<.map accessor of col('c')>``."""
        return f"<.map accessor of {self._e!r}>"

    def keys(self) -> MapFunc:
        """Return each row's map keys as a ``List`` column (DuckDB ``map_keys``).

        Returns:
            A new List expression of each row's map keys.

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

        Returns:
            A new List expression of each row's map values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import pyarrow as pa
                >>> col = pa.array([[("a", 1), ("b", 2)], [("c", 3)]],
                ...                type=pa.map_(pa.string(), pa.int64()))
                >>> ds = bt.from_arrow(pa.table({"m": col}))
                >>> ds.select(bt.col("m").map.values().alias("v")).to_pydict()
                {'v': [[1, 2], [3]]}
        """
        return MapFunc("map_values", self._e)

    def len(self) -> Expr:
        """Count each row's entries (DuckDB ``cardinality``; → Int64).

        Composed from :meth:`keys` rather than given its own kernel: the key list has one
        element per entry by construction, so its length *is* the cardinality, and a null
        map stays null through both steps.

        Returns:
            A new Int64 expression: the number of entries, or null for a null map.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import pyarrow as pa
                >>> col = pa.array([[("a", 1), ("b", 2)], [("c", 3)]],
                ...                type=pa.map_(pa.string(), pa.int64()))
                >>> ds = bt.from_arrow(pa.table({"m": col}))
                >>> ds.select(bt.col("m").map.len().alias("n")).to_pydict()
                {'n': [2, 1]}
        """
        return self.keys().list.len()

    def contains(self, key: object) -> Expr:
        """Whether each row's map holds ``key`` (DuckDB ``map_contains``; → Boolean).

        Composed from :meth:`keys`, so it answers null for a null map and ``False`` for an
        empty one — matching DuckDB, and distinguishing "no such key" from "no map".

        Args:
            key: The map key to look for in every row; a plan-time literal.

        Returns:
            A new Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import pyarrow as pa
                >>> col = pa.array([[("a", 1), ("b", 2)], [("c", 3)]],
                ...                type=pa.map_(pa.string(), pa.int64()))
                >>> ds = bt.from_arrow(pa.table({"m": col}))
                >>> ds.select(bt.col("m").map.contains("a").alias("has")).to_pydict()
                {'has': [True, False]}
        """
        return self.keys().list.contains(key)

    def get(self, key: object) -> MapFunc:
        """Look up the value for a literal ``key`` in each row's map; null if absent.

        SQL ``element_at``. ``key`` is a plan-time literal, not an expression.

        Args:
            key: The map key to look up in every row.

        Returns:
            A new expression: the value for ``key``, or null if absent.

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


class _ListNamespace:
    """List/array reductions and transforms: ``col("a").list.len()``, ``.list.sum()``.

    Generated from ``_LIST_FUNCS`` (accessor name → ``bc-expr`` ``ListFunc`` tag) —
    a single table entry adds a reduction. `get` carries an index, so it is explicit.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [[3, 1, 2]]})
            >>> ds.select(bt.col("a").list.sum().alias("s")).to_pydict()
            {'s': [6]}
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        """Wrap the parent :class:`Expr` so its `.list` methods can build on it."""
        self._e = e

    def __repr__(self) -> str:
        """Show the accessor and its parent, e.g. ``<.list accessor of col('c')>``."""
        return f"<.list accessor of {self._e!r}>"

    def __getattr__(self, name: str) -> Any:
        """Point a Polars/Daft ``.list`` (``.arr``) idiom at its Batcher spelling.

        Only reached when normal lookup fails, so it never shadows a real ``.list``
        method. ``.list.eval``, ``.list.gather``, ``.list.explode`` come back naming
        ``.list.transform``, ``.list.get``, ``ds.explode`` — see
        `batcher.plan.expr_ir.compat.guidance`.

        Args:
            name: The attribute name that was not found.

        Raises:
            AttributeError: Always, with guidance for `name`.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        raise accessor_attribute_error(self, "'.list' accessor", name, LIST_UNSUPPORTED)

    def get(self, index: int) -> ListGet:
        """Return the element at ``index`` of each list; null if out of range.

        Negative indices count from the end, ``get(-1)`` being the last element
        (Polars/Python indexing). A null or empty list yields null.

        Args:
            index: 0-based position; negatives index from the end.

        Returns:
            A new expression: the element at ``index``, or null.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[3, 1, 2], [], None]})
                >>> ds.select(bt.col("a").list.get(-1).alias("r")).to_pydict()
                {'r': [2, None, None]}
        """
        return ListGet(self._e, require_int(index, func="list.get", arg="index"))

    def first(self) -> ListGet:
        """Return the first element of each list; null if the list is null or empty.

        The idiomatic spelling of ``.list.get(0)``.

        Returns:
            A new expression: the first element, or null.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[3, 1, 2], [], None]})
                >>> ds.select(bt.col("a").list.first().alias("r")).to_pydict()
                {'r': [3, None, None]}
        """
        return ListGet(self._e, 0)

    def last(self) -> ListGet:
        """Return the last element of each list; null if the list is null or empty.

        The idiomatic spelling of ``.list.get(-1)``.

        Returns:
            A new expression: the last element, or null.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[3, 1, 2], [], None]})
                >>> ds.select(bt.col("a").list.last().alias("r")).to_pydict()
                {'r': [2, None, None]}
        """
        return ListGet(self._e, -1)

    def contains(self, value: int | float | bool | str) -> ListContains:
        """Test whether any element of each list equals ``value`` (→ Bool).

        An empty list is ``False``; a null list is null.

        Args:
            value: The literal to search for.

        Returns:
            A new Boolean expression.

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

        Returns:
            A new Int64 expression: the 1-based index, or null.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[3, 1, 2]]})
                >>> ds.select(bt.col("a").list.position(2).alias("r")).to_pydict()
                {'r': [3]}
        """
        return ListPosition(self._e, value)

    def intersect(self, other: Any) -> ListSet:
        """The distinct elements present in both this list and ``other`` (→ List).

        Spark ``array_intersect``, in this list's order.

        Args:
            other: The other list column (or an ``array(...)`` literal).

        Returns:
            A new List expression of the shared distinct elements.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[1, 2, 3]], "b": [[2, 3, 4]]})
                >>> ds.select(bt.col("a").list.intersect(bt.col("b")).alias("r")).to_pydict()
                {'r': [[2, 3]]}
        """
        return ListSet("array_intersect", self._e, _wrap(other))

    def concat(self, other: Any) -> ListSet:
        """This list's elements followed by ``other``'s, keeping duplicates (→ List).

        DuckDB ``list_concat``. Unlike :meth:`union` this does not deduplicate or reorder,
        and a null list counts as **empty** rather than making the row null — so
        ``concat`` of a null and ``[1]`` is ``[1]``, where ``union`` of the two is null.
        The result is null only when both sides are.

        Args:
            other: The other list column (or an ``array(...)`` literal).

        Returns:
            A new List expression of the two lists appended.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[1, 2]], "b": [[2, 3]]})
                >>> ds.select(bt.col("a").list.concat(bt.col("b")).alias("r")).to_pydict()
                {'r': [[1, 2, 2, 3]]}
        """
        return ListSet("array_concat", self._e, _wrap(other))

    def gather(self, indices: Any) -> ListSet:
        """Take each row's elements at the positions `indices` names (→ list).

        The operation that makes :meth:`arg_sort` usable. `arg_sort` hands back the positions
        that put a row's scores in order; without a way to spend them the reranking has to
        leave the engine and happen a row at a time in Python. Together they are a rerank in
        two expressions: sort the scores, reverse for descending, `head(k)` for the cutoff,
        then gather the candidates with the result.

        A negative index counts from the end, as :meth:`get` does. An index outside the row
        yields a null element rather than an error, because a `head(k)` wider than the row is
        an ordinary thing to write. A null row on either side gives a null row.

        Args:
            indices: A list-of-integer column giving the positions to take, per row.

        Returns:
            A new list expression holding the gathered elements, in the order given.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict(
                ...     {"docs": [["low", "high", "mid"]], "scores": [[0.1, 0.9, 0.5]]}
                ... )
                >>> ranked = bt.col("scores").list.arg_sort().list.reverse()
                >>> ds.select(top2=bt.col("docs").list.gather(ranked.list.head(2))).to_pydict()
                {'top2': [['high', 'mid']]}
        """
        return ListSet("array_gather", self._e, _wrap(indices))

    def has_all(self, other: Any) -> Expr:
        """Whether every element of ``other`` is present in this list (→ Boolean).

        DuckDB ``list_has_all``. An empty ``other`` is trivially contained, so the result
        is true; a null list on either side gives null.

        Composed as "the intersection holds as many distinct elements as ``other`` does"
        rather than the more obvious "``other`` minus this list is empty". Both are
        correct on non-null input, but the difference form reads `other` first and so
        stays non-null when *this* list is null, answering `False` where DuckDB answers
        null. Going through the intersection makes both operands load-bearing, so
        nullness propagates from either side on its own.

        Args:
            other: The list of elements to look for.

        Returns:
            A new Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[1, 2, 3], [1, 2]], "b": [[1, 2], [1, 5]]})
                >>> ds.select(bt.col("a").list.has_all(bt.col("b")).alias("r")).to_pydict()
                {'r': [True, False]}
        """
        return self.intersect(_wrap(other)).list.len() == _wrap(other).list.n_unique()

    def has_any(self, other: Any) -> Expr:
        """Whether this list shares any element with ``other`` (→ Boolean).

        DuckDB ``list_has_any``. The two share an element exactly when their intersection
        is non-empty, so an empty list on either side is false and a null list on either
        side is null.

        The trailing ``* 0`` is what carries `other`'s nullness. `intersect` treats a null
        right operand as an *empty* list (DuckDB does the same:
        ``list_intersect([1,2], NULL)`` is ``[]``), so the intersection alone would answer
        `False` where ``list_has_any`` answers null. Multiplying `other`'s length by zero
        contributes nothing for a real list and null for a null one.

        Args:
            other: The list of elements to look for.

        Returns:
            A new Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[1, 2], [1, 2]], "b": [[2, 5], [5, 6]]})
                >>> ds.select(bt.col("a").list.has_any(bt.col("b")).alias("r")).to_pydict()
                {'r': [True, False]}
        """
        other_expr = _wrap(other)
        return (self.intersect(other_expr).list.len() + other_expr.list.len() * 0) > 0

    def difference(self, other: Any) -> ListSet:
        """The distinct elements in this list but not in ``other`` (→ List).

        Spark ``array_except``, in this list's order.

        Args:
            other: The other list column (or an ``array(...)`` literal).

        Returns:
            A new List expression of the elements unique to this list.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[1, 2, 3]], "b": [[2, 3, 4]]})
                >>> ds.select(bt.col("a").list.difference(bt.col("b")).alias("r")).to_pydict()
                {'r': [[1]]}
        """
        return ListSet("array_except", self._e, _wrap(other))

    def union(self, other: Any) -> ListSet:
        """The distinct elements in either this list or ``other`` (→ List).

        Spark ``array_union``: this list's distinct elements followed by the new ones
        from ``other``.

        Args:
            other: The other list column (or an ``array(...)`` literal).

        Returns:
            A new List expression of the combined distinct elements.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[1, 2]], "b": [[2, 3]]})
                >>> ds.select(bt.col("a").list.union(bt.col("b")).alias("r")).to_pydict()
                {'r': [[1, 2, 3]]}
        """
        return ListSet("array_union", self._e, _wrap(other))

    def add(self, other: Any) -> ListZip:
        """Element-wise sum of this vector and ``other`` (→ List<Float64>).

        The embedding-math primitive: combine two embedding columns, or add a bias
        vector. Both must be the same length per row (a mismatch raises); a null element
        on either side yields null at that position. A null list row yields a null row.

        Args:
            other: The other vector column (or an ``array(...)`` literal).

        Returns:
            A new List<Float64> expression of the per-element sums.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[1.0, 2.0]], "b": [[10.0, 20.0]]})
                >>> ds.select(bt.col("a").list.add(bt.col("b")).alias("r")).to_pydict()
                {'r': [[11.0, 22.0]]}
        """
        return ListZip("list_add", self._e, _wrap(other))

    def subtract(self, other: Any) -> ListZip:
        """Element-wise difference ``this - other`` (→ List<Float64>).

        Mean-center an embedding by subtracting a centroid, or take a difference vector.
        Same length rules as :meth:`add`.

        Args:
            other: The other vector column (or an ``array(...)`` literal).

        Returns:
            A new List<Float64> expression of the per-element differences.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[10.0, 20.0]], "b": [[1.0, 2.0]]})
                >>> ds.select(bt.col("a").list.subtract(bt.col("b")).alias("r")).to_pydict()
                {'r': [[9.0, 18.0]]}
        """
        return ListZip("list_subtract", self._e, _wrap(other))

    def multiply(self, other: Any) -> ListZip:
        """Element-wise (Hadamard) product of this vector and ``other`` (→ List<Float64>).

        Gate or weight an embedding per dimension (e.g. a learned feature mask). Same
        length rules as :meth:`add`.

        Args:
            other: The other vector column (or an ``array(...)`` literal).

        Returns:
            A new List<Float64> expression of the per-element products.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[2.0, 3.0]], "b": [[5.0, 10.0]]})
                >>> ds.select(bt.col("a").list.multiply(bt.col("b")).alias("r")).to_pydict()
                {'r': [[10.0, 30.0]]}
        """
        return ListZip("list_multiply", self._e, _wrap(other))

    # --- embedding / vector helpers -------------------------------------------------

    def magnitude(self) -> ListFunc:
        """Euclidean length of the vector — the ``l2_norm`` spelling used in ML code.

        Returns:
            A Float64 expression of the vector magnitude.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"v": [[3.0, 4.0]]})
                >>> ds.select(r=bt.col("v").list.magnitude()).to_pydict()
                {'r': [5.0]}
        """
        return self.l2_norm()

    def is_unit_norm(self, tolerance: float = 1e-6) -> Expr:
        """True where the vector's magnitude is 1 within `tolerance`.

        Embedding pipelines normalize before a cosine/dot search; this asserts the
        invariant held, catching an un-normalized batch before it silently skews
        similarity scores.

        Args:
            tolerance: How far the magnitude may stray from 1.

        Returns:
            A Boolean expression, true for unit-length vectors.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"v": [[1.0, 0.0], [3.0, 4.0]]})
                >>> ds.select(r=bt.col("v").list.is_unit_norm()).to_pydict()
                {'r': [True, False]}
        """

        return (self.l2_norm() - Lit(1.0)).abs() < Lit(tolerance)

    def euclidean_distance(self, other: Any) -> ListBinary:
        """Straight-line distance between two vectors — the ``l2_distance`` spelling.

        Args:
            other: The other vector column (or an ``array(...)`` literal).

        Returns:
            A Float64 expression of the Euclidean distance.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[0.0, 0.0]], "b": [[3.0, 4.0]]})
                >>> ds.select(r=bt.col("a").list.euclidean_distance(bt.col("b"))).to_pydict()
                {'r': [5.0]}
        """
        return self.l2_distance(other)

    def angular_distance(self, other: Any) -> Expr:
        """Normalized angle between two vectors, in ``[0, 1]`` — ``acos(cosine) / pi``.

        Unlike ``1 - cosine_similarity``, this is a true metric (it satisfies the triangle
        inequality), which is what nearest-neighbour indexes and clustering algorithms
        need to stay correct.

        Args:
            other: The other vector column (or an ``array(...)`` literal).

        Returns:
            A Float64 expression of the angular distance.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[1.0, 0.0]], "b": [[0.0, 1.0]]})
                >>> ds.select(r=bt.col("a").list.angular_distance(bt.col("b"))).to_pydict()
                {'r': [0.5]}
        """
        import math

        return self.cosine_similarity(other).acos() / Lit(math.pi)

    def dim(self) -> ListFunc:
        """Number of components in the vector — the embedding dimension.

        The named spelling of ``len`` for embedding columns; asserting it is uniform is
        the first check when two models' outputs get mixed in one table.

        Returns:
            An Int64 expression of the vector dimension.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"v": [[3.0, 4.0]]})
                >>> ds.select(r=bt.col("v").list.dim()).to_pydict()
                {'r': [2]}
        """
        return self.len()

    def is_zero_vector(self) -> Expr:
        """True where every component is zero — the failed-embedding check.

        A zero vector has no direction, so cosine similarity against it is undefined; a
        batch of them usually means the encoder silently failed.

        Returns:
            A Boolean expression, true for all-zero vectors.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"v": [[3.0, 4.0], [0.0, 0.0]]})
                >>> ds.select(r=bt.col("v").list.is_zero_vector()).to_pydict()
                {'r': [False, True]}
        """

        return self.l2_norm() == Lit(0.0)

    def sum_squares(self) -> ListBinary:
        """Sum of squared components — the squared magnitude, ``dot(v, v)``.

        Cheaper than :meth:`magnitude` when only relative distances matter, since it
        skips the square root.

        Returns:
            A Float64 expression of the squared magnitude.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"v": [[3.0, 4.0]]})
                >>> ds.select(r=bt.col("v").list.sum_squares()).to_pydict()
                {'r': [25.0]}
        """
        return self.dot(self._e)

    def mean_pool(self) -> ListFunc:
        """Average of the components — mean pooling over a token-embedding sequence.

        Returns:
            A Float64 expression of the mean component.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"v": [[1.0, 3.0]]})
                >>> ds.select(r=bt.col("v").list.mean_pool()).to_pydict()
                {'r': [2.0]}
        """
        return self.mean()

    def max_pool(self) -> ListFunc:
        """Largest component — max pooling over a token-embedding sequence.

        Returns:
            An expression of the maximum component.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"v": [[1.0, 3.0]]})
                >>> ds.select(r=bt.col("v").list.max_pool()).to_pydict()
                {'r': [3.0]}
        """
        return self.max()

    def set_union(self, other: Any) -> ListSet:
        """Set union of the two lists — the Polars ``set_union`` spelling of :meth:`union`.

        Args:
            other: The other list column (or an ``array(...)`` literal).

        Returns:
            A new List expression of the combined distinct elements.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[1, 2]], "b": [[2, 3]]})
                >>> ds.select(bt.col("a").list.set_union(bt.col("b")).alias("r")).to_pydict()
                {'r': [[1, 2, 3]]}
        """
        return self.union(other)

    def set_intersection(self, other: Any) -> ListSet:
        """Set intersection — the Polars ``set_intersection`` spelling of :meth:`intersect`.

        Args:
            other: The other list column (or an ``array(...)`` literal).

        Returns:
            A new List expression of the elements present in both lists.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[1, 2, 3]], "b": [[2, 3, 4]]})
                >>> ds.select(bt.col("a").list.set_intersection(bt.col("b")).alias("r")).to_pydict()
                {'r': [[2, 3]]}
        """
        return self.intersect(other)

    def set_difference(self, other: Any) -> ListSet:
        """Set difference — the Polars ``set_difference`` spelling of :meth:`difference`.

        Args:
            other: The other list column (or an ``array(...)`` literal).

        Returns:
            A new List expression of the elements in this list but not ``other``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[1, 2, 3]], "b": [[2, 3, 4]]})
                >>> ds.select(bt.col("a").list.set_difference(bt.col("b")).alias("r")).to_pydict()
                {'r': [[1]]}
        """
        return self.difference(other)

    def transform(self, func: Any) -> ListTransform:
        """Apply ``func`` to every element, preserving list lengths (→ List).

        DuckDB ``list_transform`` / Polars ``list.eval``. ``func`` is an expression over
        ``element()`` (the current element), e.g. ``col("a").list.transform(element() * 2)``.

        Args:
            func: An expression over ``element()`` applied to each list element.

        Returns:
            A new List expression with ``func`` applied element-wise.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[1, 2, 3]]})
                >>> ds.select(bt.col("a").list.transform(bt.element() * 2).alias("r")).to_pydict()
                {'r': [[2, 4, 6]]}
        """
        return ListTransform(self._e, _wrap(func))

    def drop_nulls(self) -> ListFilter:
        """Drop the null elements of each list (Polars ``list.drop_nulls``, → List).

        Returns:
            A new List expression with the nulls removed; list lengths change, and a
            null *list* stays null.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[1, None, 3], [None]]})
                >>> ds.select(r=bt.col("a").list.drop_nulls()).to_pydict()
                {'r': [[1, 3], []]}
        """
        from batcher.plan.functions.collection import element

        return self.filter(element().is_not_null())

    def filter(self, predicate: Any) -> ListFilter:
        """Keep the elements where ``predicate`` is true (→ List).

        DuckDB ``list_filter``. ``predicate`` is an expression over ``element()`` (the
        current element), e.g. ``col("a").list.filter(element() > 0)``.

        Args:
            predicate: A boolean expression over ``element()`` selecting elements to keep.

        Returns:
            A new List expression of the elements satisfying ``predicate``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[-1, 2, -3, 4]]})
                >>> ds.select(bt.col("a").list.filter(bt.element() > 0).alias("r")).to_pydict()
                {'r': [[2, 4]]}
        """
        return ListFilter(self._e, _wrap(predicate))

    def simhash(self, num_bits: int = 64, *, seed: int = 0) -> ListSimhash:
        """A random-hyperplane (SimHash) signature of an embedding → List<Int64> of bits.

        The vector-space counterpart of :meth:`~batcher.Expr.str.minhash`. `minhash`
        estimates *Jaccard* similarity between sets of shingles and says nothing about
        vectors; `simhash` estimates the *cosine* similarity between embeddings. Two
        vectors separated by an angle ``θ`` agree on each bit with probability
        ``1 - θ/π``, so the fraction of positions two signatures agree on
        (:meth:`jaccard`, which is exactly that fraction) estimates their angle.

        This is the blocking key a similarity join needs. Comparing every pair of `n`
        embeddings is ``O(n²)``; banding the bits and hashing each band means two rows
        become candidates only when they are close, and the exact
        :meth:`cosine_similarity` then scores the survivors. LSH governs recall, never
        precision — see :meth:`~batcher.Dataset.ml.similarity_join`.

        The hyperplanes are derived by hashing ``(seed, bit, dimension)``, never stored,
        so every partition and every machine draws the same ones: a signature computed
        on one node is comparable with one computed on another. A null or empty list has
        no direction and yields null; a null element reads as ``0.0``. Only the vector's
        *direction* matters, so the signature is scale-invariant.

        Each bit occupies a whole Int64 element. That is deliberately fat — it gives the
        signature the same ``List<Int64>`` shape as a MinHash signature, so one banding
        implementation serves both — and a signature is a transient blocking key, not
        something you store.

        Args:
            num_bits: Signature length, in ``[1, 4096]``. More bits sharpen the estimate
                and cost proportionally. Choose a multiple of the band count.
            seed: Selects the set of hyperplanes. Two datasets must share a seed for
                their signatures to be comparable.

        Returns:
            A List<Int64> expression of `num_bits` values, each 0 or 1.

        Raises:
            PlanError: If `num_bits` is outside ``[1, 4096]``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"v": [[1.0, 0.0], [10.0, 0.0], [0.0, 1.0]]})
                >>> sig = ds.select(s=bt.col("v").list.simhash(8))
                >>> sig.to_pydict()["s"][0] == sig.to_pydict()["s"][1]
                True

                >>> # Agreement estimates the angle: parallel rows agree everywhere,
                >>> # orthogonal rows agree about half the time.
                >>> pairs = ds.select(
                ...     a=bt.col("v").list.simhash(256), b=bt.col("v").list.simhash(256)
                ... )
                >>> pairs.select(same=bt.col("a").list.jaccard(bt.col("b"))).to_pydict()["same"]
                [1.0, 1.0, 1.0]
        """
        if not 1 <= num_bits <= 4096:
            raise PlanError(f"list.simhash(): num_bits must be in [1, 4096], got {num_bits}")
        return ListSimhash(self._e, num_bits, seed)

    def slice(self, offset: int, length: int | None = None) -> ListSlice:
        """Return the 0-based sub-range ``[offset, offset+length)`` of each list.

        With no ``length`` the slice runs to the end of the list. A null list stays
        null; an empty list stays empty.

        Args:
            offset: 0-based start index.
            length: Number of elements to take; ``None`` means to the end.

        Returns:
            A new List expression: the selected sub-range.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[10, 20, 30, 40]]})
                >>> ds.select(bt.col("a").list.slice(1, 2).alias("r")).to_pydict()
                {'r': [[20, 30]]}
        """
        offset = require_int(offset, func="list.slice", arg="offset")
        if length is not None:
            length = require_int(length, func="list.slice", arg="length")
        return ListSlice(self._e, offset, length)

    def head(self, n: int = 5) -> ListSlice:
        """Return the first ``n`` elements of each list — the leading sub-range.

        A convenience for ``slice(0, n)`` (Polars ``list.head``): a null list stays null,
        an empty or shorter list yields all it has.

        Args:
            n: How many leading elements to keep.

        Returns:
            A new List expression: the first ``n`` elements of each list.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[10, 20, 30, 40]]})
                >>> ds.select(bt.col("a").list.head(2).alias("r")).to_pydict()
                {'r': [[10, 20]]}
        """
        return ListSlice(self._e, 0, n)

    def join(self, separator: str) -> ListJoin:
        """Concatenate each list's elements into one string, joined by ``separator``.

        Elements are cast to text and null elements are skipped. A null or empty
        list yields null (→ Utf8).

        Args:
            separator: The text inserted between consecutive elements.

        Returns:
            A new Utf8 expression: the joined string, or null.

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

        Returns:
            A new List expression with one level of nesting removed.

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

        Returns:
            A new Float64 expression: the dot product.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[1.0, 2.0, 3.0]], "b": [[4.0, 5.0, 6.0]]})
                >>> ds.select(bt.col("a").list.dot(bt.col("b")).alias("r")).to_pydict()
                {'r': [32.0]}
        """
        return ListBinary("dot", self._e, _wrap(other))

    def jaccard(self, other: Any) -> ListBinary:
        """The fraction of positions where this list and `other` hold the same value.

        Over two `str.minhash` signatures this is the unbiased estimator of the two
        documents' Jaccard similarity — the near-duplicate score. Over arbitrary equal
        length lists it is simply the agreement rate. Null if either list is null, and
        null for two empty lists (no positions to agree on).

        Args:
            other: The other equal-length list column to compare position-by-position.

        Returns:
            A new Float64 expression: the fraction of agreeing positions.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": ["hello world", "hello world!"]})
                >>> sigs = ds.select(s=bt.col("t").str.minhash(128)).to_pydict()["s"]
                >>> pair = bt.from_pydict({"a": [sigs[0]], "b": [sigs[1]]})
                >>> pair.select(j=bt.col("a").list.jaccard(bt.col("b"))).to_pydict()["j"][0] > 0.5
                True
        """
        return ListBinary("jaccard", self._e, _wrap(other))

    def multiset_overlap(self, other: Any) -> ListBinary:
        """How many of this list's elements `other` accounts for, counting repeats (→ Float64).

        The clipped multiset intersection size ``Σ min(count_here(v), count_there(v))``. It
        differs from ``set_intersection(other).len()`` in exactly one way, and that way is the
        point: a value repeated four times here against one occurrence there contributes 1,
        not 4. That clip is how BLEU's modified n-gram precision refuses to reward a
        degenerate ``the the the the``, and it is ROUGE-N's numerator read from the other
        side. Pair it with :meth:`~batcher.Expr.str.token_ngrams` to score generated text.

        Order does not matter and the lists need not be the same length. Null if either list
        is null; a null element matches nothing.

        Args:
            other: The other list column to account for this one's elements.

        Returns:
            A new Float64 expression: the clipped overlap count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [["the", "the", "cat"]], "b": [["the", "cat"]]})
                >>> ds.select(o=bt.col("a").list.multiset_overlap(bt.col("b"))).to_pydict()
                {'o': [2.0]}
        """
        return ListBinary("multiset_overlap", self._e, _wrap(other))

    def lcs_length(self, other: Any) -> ListBinary:
        """The length of the longest common subsequence of the two lists (→ Float64).

        The one overlap measure that reads *order*. :meth:`multiset_overlap` cannot tell
        ``the cat sat`` from ``sat cat the`` — both share the same three tokens — while an LCS
        scores the reordering far lower. That difference is exactly what separates ROUGE-N from
        ROUGE-L, and why summarization is scored with the latter: a summary using the right
        words in the wrong order is not a summary.

        The subsequence need not be contiguous, so ``a x b y c`` and ``a b c`` share three.

        **This is the expensive one.** It is ``O(n·m)`` in the two rows' lengths, against
        ``O(n+m)`` for every other list operation here. On tokenized sentences that is nothing;
        on two thousand-token documents it is a million cell updates per row. Truncate, or score
        at the sentence level, rather than reaching for it on whole documents.

        Args:
            other: The other list column to find a common subsequence with.

        Returns:
            A new Float64 expression: the longest common subsequence length.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict(
                ...     {"a": [["the", "cat", "sat"]], "b": [["sat", "cat", "the"]]}
                ... )
                >>> ds.select(
                ...     ordered=bt.col("a").list.lcs_length(bt.col("a")),
                ...     shuffled=bt.col("a").list.lcs_length(bt.col("b")),
                ... ).to_pydict()
                {'ordered': [3.0], 'shuffled': [1.0]}
        """
        return ListBinary("lcs_length", self._e, _wrap(other))

    def cosine_similarity(self, other: Any) -> ListBinary:
        """Cosine similarity with another vector column, in ``[-1, 1]`` (→ Float64).

        The standard embedding-similarity score for retrieval / RAG; null if either
        vector has zero magnitude. Both vectors must have the same length.

        Args:
            other: The other vector column (or an ``array(...)`` literal).

        Returns:
            A new Float64 expression in ``[-1, 1]``, or null.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[1.0, 1.0]], "b": [[0.0, 1.0]]})
                >>> r = ds.select(bt.col("a").list.cosine_similarity(bt.col("b")).alias("r"))
                >>> round(r.to_pydict()["r"][0], 4)
                0.7071
        """
        return ListBinary("cosine_similarity", self._e, _wrap(other))

    def cosine_distance(self, other: Any) -> Expr:
        """Cosine distance ``1 - cosine_similarity`` to another vector column (→ Float64).

        The common nearest-neighbour ranking metric for embeddings: 0 for identical
        direction, 1 for orthogonal, 2 for opposite. Both vectors must have the same
        length.

        Args:
            other: The other vector column (or an ``array(...)`` literal).

        Returns:
            A new Float64 expression: ``1 - cosine_similarity``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[1.0, 0.0]], "b": [[0.0, 1.0]]})
                >>> ds.select(bt.col("a").list.cosine_distance(bt.col("b")).alias("r")).to_pydict()
                {'r': [1.0]}
        """
        return 1.0 - ListBinary("cosine_similarity", self._e, _wrap(other))

    def l2_distance(self, other: Any) -> ListBinary:
        """Euclidean (L2) distance to another vector column (→ Float64).

        The metric for nearest-neighbour vector search. Both vectors must have the
        same length.

        Args:
            other: The other vector column (or an ``array(...)`` literal).

        Returns:
            A new Float64 expression: the Euclidean distance.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[0.0, 0.0]], "b": [[3.0, 4.0]]})
                >>> ds.select(bt.col("a").list.l2_distance(bt.col("b")).alias("r")).to_pydict()
                {'r': [5.0]}
        """
        return ListBinary("l2_distance", self._e, _wrap(other))

    def l1_distance(self, other: Any) -> ListBinary:
        """Manhattan (L1) distance to another vector column (→ Float64).

        The sum of absolute per-element differences ``Σ|aᵢ - bᵢ|`` — the metric some
        embedding models and sparse features are trained under. Both vectors must have
        the same length.

        Args:
            other: The other vector column (or an ``array(...)`` literal).

        Returns:
            A new Float64 expression: the Manhattan distance.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[0.0, 0.0]], "b": [[3.0, 4.0]]})
                >>> ds.select(bt.col("a").list.l1_distance(bt.col("b")).alias("r")).to_pydict()
                {'r': [7.0]}
        """
        return ListBinary("l1_distance", self._e, _wrap(other))

    def hamming_distance(self, other: Any) -> ListBinary:
        """Number of positions where two vectors differ (→ Float64).

        The distance for **binary or quantized embeddings** (each element ``0``/``1`` or a
        small integer), where it is far cheaper than a float metric and is what a binary
        vector index ranks by. Both vectors must have the same length.

        Args:
            other: The other vector column (or an ``array(...)`` literal).

        Returns:
            A new Float64 expression: the count of differing positions.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[1, 0, 1, 1]], "b": [[1, 1, 0, 1]]})
                >>> ds.select(bt.col("a").list.hamming_distance(bt.col("b")).alias("r")).to_pydict()
                {'r': [2.0]}
        """
        return ListBinary("hamming", self._e, _wrap(other))


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
    "arg_sort": "arg_sort",  # indices that sort ascending (→ list of Int64)
    "l2_norm": "l2_norm",  # Euclidean norm = sqrt(sum of squares) (-> Float64)
    "l1_norm": "l1_norm",  # Manhattan norm = sum of absolute values (-> Float64)
    "max_abs": "max_abs",  # max absolute value = the MaxAbs-scaling divisor (-> Float64)
    "normalize": "normalize",  # L2-normalize to unit length (→ list); embedding prep
    "softmax": "softmax",  # logits → probability distribution per row (→ list)
    "log_softmax": "log_softmax",  # logits → log-domain distribution (→ list); no underflow
    "entropy": "entropy",  # per-row Shannon entropy in nats (→ Float64); uncertainty
    "cum_sum": "cum_sum",  # cumulative sum per row (→ list)
    "diff": "diff",  # first difference xᵢ−xᵢ₋₁ per row, leading null (→ list)
}


def _list_reduction_doc(name: str) -> str:
    """Fallback docstring for a ``.list`` reduction without a curated entry.

    Every reduction but ``reverse`` carries a curated entry; only ``reverse`` falls
    through to here, so the summary and example reflect an element-reversing list op.
    """
    return (
        f"Return each list with its elements {name}d.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[1, 2, 3]]})\n'
        f'        >>> ds.select(r=bt.col("xs").list.{name}()).to_pydict()\n'
        "        {'r': [[3, 2, 1]]}"
    )


_bind_accessors(
    _ListNamespace,
    _LIST_FUNCS,
    lambda e, t: ListFunc(t, e),
    _list_reduction_doc,
    "A new :class:`~batcher.Expr` carrying the per-row reduction.",
)
