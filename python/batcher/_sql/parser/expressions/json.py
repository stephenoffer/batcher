"""SQL JSON functions — extraction (``json_extract`` / ``->`` / ``->>``) and inspection.

Every SQL JSON *extraction* form lowers to the one ``.json.extract_string`` accessor: it
reads the value at a path as text (scalars verbatim, objects/arrays as compact JSON), and a
surrounding ``CAST`` supplies numeric/boolean typing — the ``CAST(json_extract(...) AS
BIGINT)`` idiom. This keeps a single JSON path in the engine (the Rust ``.json`` kernel)
behind both the SQL and the DataFrame surfaces.

The *inspection* functions (``json_valid``, ``json_exists``, ``json_keys``,
``json_array_length``) go through the same accessor family. ``json_type`` is deliberately
absent: DuckDB names the SQL type the value would cast to (``UBIGINT``, ``VARCHAR``), where
the engine's ``.json.type_of`` names the JSON type (``number``, ``string``). Mapping one to
the other would answer a different question with a plausible-looking string.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher.plan.expr_ir import Expr


def json_path(node) -> str:
    """Reconstruct a ``$.a.b[0]`` path string from sqlglot's parsed JSON path.

    The path arrives either as a ``JSONPath`` node (a list of root/key/subscript parts,
    from ``json_extract(j, '$.a')`` or ``j -> '$.a'``) or as a plain string literal; both
    normalize to the ``$``-rooted dotted form the engine's ``.json`` accessor consumes.
    """
    if isinstance(node, exp.Literal):
        return node.this if node.this.startswith("$") else f"$.{node.this}"
    if not isinstance(node, exp.JSONPath):
        raise NotImplementedError("JSON path must be a constant path expression")
    out = "$"
    for part in node.expressions:
        if isinstance(part, exp.JSONPathRoot):
            continue
        if isinstance(part, exp.JSONPathKey):
            out += f".{part.this}"
        elif isinstance(part, exp.JSONPathSubscript):
            out += f"[{part.this}]"
        else:
            raise NotImplementedError(f"unsupported JSON path element: {type(part).__name__}")
    return out


def json_extract(tr, node) -> Expr:
    """``json_extract`` / ``json_extract_string`` / ``->`` / ``->>`` → ``.json.extract_string``.

    Every form returns the value at the path as text; a surrounding ``CAST`` supplies the
    numeric/boolean typing when the caller wants it.
    """
    return tr._scalar(node.this).json.extract_string(json_path(node.expression))


# `f(doc[, path])` → the `.json` accessor of the same shape. The path defaults to the
# document root, which is what DuckDB's one-argument forms mean.
_JSON_PATH_FNS = {
    "json_keys": "keys",
    "json_array_length": "array_length",
    "json_exists": "exists",
    "json_value": "value",
}

# `f(doc)` → a `.json` accessor that reads the whole document.
_JSON_WHOLE_FNS = {"json_pretty": "pretty", "json_structure": "structure"}


def json_function(tr, node) -> Expr | None:
    """Translate a JSON inspection call, or return None when the name is not one of them.

    Handles both the typed node sqlglot promotes (``JSONKeys``) and the names it leaves
    anonymous (``json_valid``, ``json_exists``, ``json_array_length``).
    """
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
    if name == "json_contains" and len(args) == 2:
        from batcher._sql.parser.expressions.literals import _const_str_arg

        needle = _const_str_arg(args[1], "json_contains()", "value")
        return tr._scalar(args[0]).json.contains(needle)
    if name == "json_valid":
        # A document is valid JSON exactly when the root has a JSON type; the kernel
        # answers null for text it cannot parse, which is the same test.
        return tr._scalar(args[0]).json.type_of().is_not_null()

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
