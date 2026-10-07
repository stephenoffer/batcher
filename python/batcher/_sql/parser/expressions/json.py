"""SQL JSON functions — extraction (``json_extract`` / ``->`` / ``->>``) and inspection.

SQL has *two* extraction forms and they differ on two leaf kinds: ``json_extract`` / ``->``
report the value **as JSON text** (a string keeps its quotes, a JSON null is the token
``null``), while ``json_extract_string`` / ``->>`` unquote the string and report a JSON null
as SQL NULL. Lowering both to the unquoting accessor — which is what this used to do —
answered ``json_extract('{"a":"x"}', '$.a')`` with ``x`` instead of ``"x"``, and collapsed
"the key is absent" and "the key is present and null" into one answer. A surrounding
``CAST`` supplies numeric/boolean typing either way (the ``CAST(json_extract(...) AS
BIGINT)`` idiom).

The *inspection* functions (``json_valid``, ``json_exists``, ``json_keys``,
``json_array_length``) go through the same accessor family. ``json_type`` is deliberately
absent: DuckDB names the SQL type the value would cast to (``UBIGINT``, ``VARCHAR``), where
the engine's ``.json.type_of`` names the JSON type (``number``, ``string``). Mapping one to
the other would answer a different question with a plausible-looking string.
"""

from __future__ import annotations

import json

import pyarrow as pa
from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Expr, coalesce, lit, nullif, when
from batcher.plan.expr_ir.namespaces._json_path import check_json_path, split_wildcard_tail
from batcher.plan.types.registry import canonical_dtype_name, resolve_dtype


def json_path(node) -> str:
    """Reconstruct a ``$.a.b[0]`` path string from sqlglot's parsed JSON path.

    The path arrives either as a ``JSONPath`` node (a list of root/key/subscript parts,
    from ``json_extract(j, '$.a')`` or ``j -> '$.a'``) or as a plain string literal; both
    normalize to the ``$``-rooted form the engine's ``.json`` accessor consumes. A key
    holding a character the dotted form cannot carry (``x.y``) is re-quoted, so
    ``'$."x.y"'`` keeps meaning the key ``x.y``. A trailing ``[*]`` is kept for
    `json_extract` to peel off; every other multi-value selector is refused with the
    same reasons the accessor gives.
    """
    if isinstance(node, exp.Literal):
        return node.this if node.this.startswith("$") else f"$.{node.this}"
    if not isinstance(node, exp.JSONPath):
        raise NotImplementedError("JSON path must be a constant path expression")
    out = "$"
    for part in node.expressions:
        out += _path_step(part)
    return out


def _path_step(part) -> str:
    """One sqlglot JSON path part as path text, refusing the selectors the engine lacks."""
    if isinstance(part, exp.JSONPathRoot):
        return ""
    if isinstance(part, exp.JSONPathKey):
        if isinstance(part.this, exp.JSONPathWildcard):
            return ".*"
        return f".{_quote_key(str(part.this))}"
    if isinstance(part, exp.JSONPathSubscript):
        if isinstance(part.this, exp.JSONPathWildcard):
            return "[*]"
        if isinstance(part.this, exp.JSONPathSlice):
            return "[:]"
        return f"[{part.this}]"
    if isinstance(part, exp.JSONPathRecursive):
        return f"..{part.this or ''}"
    if isinstance(part, exp.JSONPathUnion):
        return "[,]"
    if isinstance(part, exp.JSONPathSelector):
        return "[?]"
    raise PlanError(f"unsupported JSON path element: {type(part).__name__}")


def _quote_key(key: str) -> str:
    """`key` bare when the dotted form can carry it, else double-quoted with escapes."""
    if key and not any(c in key for c in ".[]\"'*\\ "):
        return key
    escaped = key.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def json_extract(tr, node) -> Expr:
    """``json_extract`` / ``->`` / ``json_extract_string`` / ``->>`` → the right accessor.

    sqlglot separates the two: ``JSONExtract`` is the JSON-text form and
    ``JSONExtractScalar`` the unquoting one. The difference shows only on a string leaf
    (quotes) and a JSON null (token vs SQL NULL), which is what made it invisible.

    Args:
        tr: The translator.
        node: The `JSONExtract` or `JSONExtractScalar` node.

    Returns:
        The extraction expression.
    """
    from batcher.plan.expr_ir import StrFunc, StrFuncDyn

    doc = tr._scalar(node.this)
    fn = "json_extract_string" if isinstance(node, exp.JSONExtractScalar) else "json_extract"
    if not isinstance(node.expression, (exp.Literal, exp.JSONPath)):
        # A path computed per row (`j ->> path_col`). `StrFuncDyn` groups the rows by their
        # distinct path and runs the same extraction kernel per group, so a per-row path
        # answers exactly what the constant one does. It must be `$`-rooted: the prefix a
        # bare key gets below is a plan-time rewrite of a constant.
        return StrFuncDyn(fn, doc, pattern=tr._scalar(node.expression))
    path = json_path(node.expression)
    scalar = isinstance(node, exp.JSONExtractScalar)
    prefix = split_wildcard_tail(path)
    if prefix is not None:
        # `$.a[*]` lists every element of `$.a`, which DuckDB answers as a LIST of the
        # same renderings the scalar forms use. Answering it with the array as one string
        # -- what dropping the `[*]` did -- was a different type and a different value.
        fn_all = "json_extract_string_all" if scalar else "json_extract_all"
        return StrFunc(fn_all, doc, pattern=prefix)
    if scalar:
        return doc.json.extract_string(path)
    return StrFunc("json_extract", doc, pattern=check_json_path(path))


# `f(doc[, path])` → the `.json` accessor of the same shape. The path defaults to the
# document root, which is what DuckDB's one-argument forms mean.
_JSON_PATH_FNS = {
    "json_keys": "keys",
    "json_array_length": "array_length",
    "json_exists": "exists",
    "json_value": "value",
}

# DuckDB's typed decode, and the accessor's own name for it. The structure argument is
# DuckDB's: a JSON document whose leaves name a type (`{"a": "BIGINT", "b": ["VARCHAR"]}`).
_JSON_DECODE_FNS = frozenset({"json_transform", "json_transform_strict", "json_decode"})


def structure_type(spec: str) -> pa.DataType:
    """The Arrow type a DuckDB `json_transform` structure names.

    A string leaf is a type name (`"BIGINT"`, `"VARCHAR"`), a one-element array is a list
    of that element, and an object is a struct of its keys in document order. A structure
    that is not JSON at all is read as one flat type name, so ``json_decode(j, 'int64')``
    works as well.

    Args:
        spec: The structure text.

    Returns:
        The Arrow type.

    Raises:
        PlanError: A leaf names no type the engine knows, or an array has other than one
            element.
    """
    try:
        parsed = json.loads(spec)
    except ValueError:
        parsed = spec
    return _structure_node(parsed, spec)


def _structure_node(node: object, spec: str) -> pa.DataType:
    if isinstance(node, str):
        resolved = resolve_dtype(canonical_dtype_name(node))
        if resolved is None:
            raise PlanError(f"json_transform structure {spec!r}: unknown type {node!r}")
        return resolved
    if isinstance(node, list) and len(node) == 1:
        return pa.list_(_structure_node(node[0], spec))
    if isinstance(node, dict) and node:
        return pa.struct([(k, _structure_node(v, spec)) for k, v in node.items()])
    raise PlanError(
        f"json_transform structure {spec!r}: expected a type name, a one-element array or "
        f"an object, got {json.dumps(node)}"
    )


# `f(doc)` → a `.json` accessor that reads the whole document.
_JSON_WHOLE_FNS = {"json_pretty": "pretty", "json_structure": "structure"}


def json_function(tr, node) -> Expr | None:
    """Translate a JSON inspection call, or return None when the name is not one of them.

    Handles both the typed node sqlglot promotes (``JSONKeys``) and the names it leaves
    anonymous (``json_valid``, ``json_exists``, ``json_array_length``).
    """
    if isinstance(node, exp.JSONType):
        # A refusal with a reason, not a node-type error: DuckDB names the *SQL* type the
        # value would cast to (`UBIGINT`, `VARCHAR`) where the engine's `.json.type_of`
        # names the *JSON* type (`number`, `string`). Answering one with the other returns
        # a plausible string for a different question.
        raise NotImplementedError(
            "json_type() is not supported: DuckDB reports the SQL type a value would cast "
            "to (UBIGINT, VARCHAR), which is not the JSON type the engine's accessor "
            "reports (number, string). Use json_valid() to test parseability, or cast the "
            "extracted value to the type you expect"
        )
    if isinstance(node, exp.JSONKeys):
        doc = tr._scalar(node.this)
        path = node.args.get("expression")
        return doc.json.keys(json_path(path) if path is not None else "$")

    if not isinstance(node, exp.Anonymous):
        return None
    name = node.name.lower()
    args = list(node.expressions)
    if not args:
        return None

    whole = _JSON_WHOLE_FNS.get(name)
    if whole is not None and len(args) == 1:
        return getattr(tr._scalar(args[0]).json, whole)()
    if name in _JSON_DECODE_FNS and len(args) == 2:
        from batcher._sql.parser.expressions.literals import _const_str_arg

        spec = _const_str_arg(args[1], f"{name}()", "structure")
        strict = name == "json_transform_strict"
        return tr._scalar(args[0]).json.decode(structure_type(spec), strict=strict)
    if name == "json_contains" and len(args) == 2:
        from batcher._sql.parser.expressions.literals import _const_str_arg

        needle = _const_str_arg(args[1], "json_contains()", "value")
        return tr._scalar(args[0]).json.contains(needle)
    if name == "json_valid":
        # A document is valid JSON exactly when the root has a JSON type, and the kernel
        # answers null for text it cannot parse — so `type_of() IS NOT NULL` is the test.
        #
        # It is not the whole test, because the kernel answers null for *two* reasons and
        # they need different answers: unparseable text is FALSE, but a NULL document is
        # NULL, the way every SQL predicate propagates its input's nullness. Reading both
        # as FALSE made a NULL row report as *invalid JSON* rather than unknown, so
        # `WHERE NOT json_valid(j)` — how you isolate bad documents — returned every NULL
        # row alongside the genuinely malformed ones. Same query, no error, wrong row set.
        doc = tr._scalar(args[0])
        valid = doc.json.type_of().is_not_null()
        return when(doc.is_null()).then(nullif(lit(True), lit(True))).otherwise(valid)

    if name == "json_array_length" and len(args) == 1:
        # A document that parses but is not an array has length 0, not null: DuckDB
        # answers 0 for `{"a":1}`, `"s"` and `5` alike. The kernel returns null for those
        # *and* for a document it cannot parse *and* for a null input, so the three are
        # separated by asking the root's type first — null type means unparseable or
        # null, which stays null, and anything else that is not an array coalesces to 0.
        doc = tr._scalar(args[0])
        parsed = doc.json.type_of().is_not_null()
        length = coalesce(doc.json.array_length(), lit(0))
        return when(parsed).then(length).otherwise(nullif(lit(0), lit(0)))

    method = _JSON_PATH_FNS.get(name)
    if method is None:
        return None
    doc = tr._scalar(args[0])
    if len(args) == 1:
        if name == "json_exists":  # the path is required, not defaulted
            return None
        return getattr(doc.json, method)()
    if len(args) != 2:
        return None
    return getattr(doc.json, method)(json_path(args[1]))
