"""SQL JSON extraction — ``json_extract`` / ``json_extract_string`` / ``->`` / ``->>``.

Every SQL JSON form lowers to the one ``.json.extract_string`` accessor: it reads the value
at a path as text (scalars verbatim, objects/arrays as compact JSON), and a surrounding
``CAST`` supplies numeric/boolean typing — the ``CAST(json_extract(...) AS BIGINT)`` idiom.
This keeps a single JSON path in the engine (the Rust ``.json`` kernel) behind both the SQL
and the DataFrame surfaces.
"""

from __future__ import annotations

from batcher.plan.expr_ir import Expr


def json_path(node) -> str:
    """Reconstruct a ``$.a.b[0]`` path string from sqlglot's parsed JSON path.

    The path arrives either as a ``JSONPath`` node (a list of root/key/subscript parts,
    from ``json_extract(j, '$.a')`` or ``j -> '$.a'``) or as a plain string literal; both
    normalize to the ``$``-rooted dotted form the engine's ``.json`` accessor consumes.
    """
    from sqlglot import expressions as exp

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
