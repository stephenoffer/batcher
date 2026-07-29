"""Spark SQL names whose translation is a composition rather than a rename.

`bt.sql(dialect="spark")` selects a *parser*, so a Spark query reaches the same expression
surface every other dialect does. Most Spark builtins are the DuckDB function under a
different name and are handled by the shared tables; the ones here need an expression
built out of two or more engine primitives (`pmod`, `width_bucket`, `find_in_set`,
`elt`), or a null rule the tables cannot express (`nvl2`, `equal_null`, `nullifzero`).

Where Spark and DuckDB genuinely disagree, the engine keeps DuckDB's semantics — that is
the standing policy, because DuckDB is the differential oracle the suite is written
against. Nothing here overrides it; every entry is a name Spark has and DuckDB does not.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher.plan.expr_ir import Expr, coalesce, lit, nullif, when

__all__ = ["spark_function"]

# `f(x)` → a sign-preserving no-op / negation. Spark spells the unary operators as
# functions; `positive` is the identity and `negative` is `0 - x`.
_SIGN = {"positive": lambda v: v, "negative": lambda v: lit(0) - v}

# Spark's UTF-8 validity family. An engine string column is Arrow `Utf8`, whose bytes
# Arrow has already validated, so every non-null value *is* valid UTF-8: the predicates
# are true and the repair functions are the identity. Listed by name with what each
# returns, rather than left to raise, because a ported query that calls them is asking a
# question this engine has already answered.
_UTF8_VALID = {
    "is_valid_utf8": "predicate",
    "validate_utf8": "identity",
    "try_validate_utf8": "identity",
    "make_valid_utf8": "identity",
}


def spark_function(tr, node) -> Expr | None:
    """Translate a Spark-only scalar call, or None when the node is not one of them."""
    if isinstance(node, exp.If):
        return (
            when(tr._scalar(node.this))
            .then(tr._scalar(node.args["true"]))
            .otherwise(tr._scalar(node.args["false"]))
        )
    if isinstance(node, exp.Nvl2):
        # `nvl2(a, b, c)` is `b` when `a` is not null, else `c` — the test is on `a`,
        # whose value is otherwise unused.
        return (
            when(tr._scalar(node.this).is_not_null())
            .then(tr._scalar(node.args["true"]))
            .otherwise(tr._scalar(node.args["false"]))
        )
    if isinstance(node, exp.EqualNull):
        # Null-safe equality: two nulls are equal, and a null never equals a value.
        return tr._scalar(node.this).eq_missing(tr._scalar(node.expression))
    if isinstance(node, exp.Space):
        from batcher._sql.parser.expressions.literals import _const_int_arg

        return lit(" ").str.repeat(_const_int_arg(node.this, "space(): count"))
    if isinstance(node, exp.Elt):
        return _elt(tr, [node.this, *node.expressions])
    if isinstance(node, exp.WidthBucket):
        return _width_bucket(tr, node)
    if isinstance(node, exp.Getbit):
        # Spark's `bit_get(x, n)` — the nth bit of an **integer**, counting from the
        # least significant. sqlglot gives DuckDB's `get_bit(BITSTRING, n)` the same
        # node, and that one indexes a bit *string* from the left, so answering it with
        # this shift would return the wrong bit rather than an error. The engine has no
        # BIT type, so a non-integer argument is declined instead.
        if not _is_integer_operand(node.this):
            return None
        # The bitwise methods, not `>>`/`&`: those operators are the *logical* ones on
        # an `Expr`, and `&` on two integers raises rather than masking.
        shifted = tr._scalar(node.this).bitwise_right_shift(tr._scalar(node.expression))
        return shifted.bitwise_and(lit(1))
    if isinstance(node, exp.RegexpCount):
        from batcher._sql.parser.expressions.literals import _const_str_arg

        pat = _const_str_arg(node.expression, "regexp_count()", "pattern")
        return tr._scalar(node.this).str.regexp_count(pat)
    if isinstance(node, exp.RegexpSubstr):
        from batcher._sql.parser.expressions.literals import _const_str_arg

        pat = _const_str_arg(node.expression, "regexp_substr()", "pattern")
        return tr._scalar(node.this).str.regexp_extract(pat)
    if isinstance(node, exp.ParseUrl):
        return _parse_url(tr, node)
    if isinstance(node, exp.Struct):
        return _struct_node(tr, node)
    if isinstance(node, exp.SubstringIndex):
        from batcher._sql.parser.expressions.literals import _const_int_arg, _const_str_arg

        delim = _const_str_arg(node.args.get("delimiter"), "substring_index()", "delimiter")
        count = _const_int_arg(node.args.get("count"), "substring_index(): count")
        return tr._scalar(node.this).str.substring_index(delim, count)

    if not isinstance(node, exp.Anonymous):
        return None
    from batcher._sql.parser.expressions.literals import _const_int_arg

    name = node.name.lower()
    args = list(node.expressions)

    if name in _SIGN and len(args) == 1:
        return _SIGN[name](tr._scalar(args[0]))
    if name == "nullifzero" and len(args) == 1:
        return nullif(tr._scalar(args[0]), lit(0))
    if name == "zeroifnull" and len(args) == 1:
        return coalesce(tr._scalar(args[0]), lit(0))
    if name == "pmod" and len(args) == 2:
        # The *positive* modulus: `pmod(-10, 3)` is 2 where `%` gives -1.
        left, right = tr._scalar(args[0]), tr._scalar(args[1])
        return (left.mod(right) + right).mod(right)
    if name == "btrim" and len(args) in (1, 2):
        if len(args) == 1:
            return tr._scalar(args[0]).str.trim()
        from batcher._sql.parser.expressions.literals import _const_str_arg

        chars = _const_str_arg(args[1], "btrim()", "character set")
        return tr._scalar(args[0]).str.trim(chars)
    if name == "find_in_set" and len(args) == 2:
        return _find_in_set(tr, args)
    if name == "bround" and len(args) == 2:
        # Banker's rounding: half goes to the *even* neighbour, which is what `rint`
        # does. Scaling by 10^d and back is the standard way to get it at `d` digits.
        digits = _const_int_arg(args[1], "bround(): digits")
        factor = lit(10.0**digits)
        return (tr._scalar(args[0]) * factor).rint() / factor
    if name in _UTF8_VALID and len(args) == 1:
        return _utf8_validity(tr._scalar(args[0]), _UTF8_VALID[name])
    if name in ("timezone_hour", "timezone_minute") and len(args) == 1:
        # Engine timestamps are tz-naive, so the UTC offset is zero by construction —
        # and DuckDB answers 0 for a naive TIMESTAMP too. Null in, null out.
        value = tr._scalar(args[0])
        return when(value.is_not_null()).then(lit(0)).otherwise(nullif(lit(0), lit(0)))
    return None


# `parse_url(url, part)` → the regex that captures that part, group 1. Written against
# the URL grammar rather than a parser because the engine's regex kernel is already the
# fast path, and every part below is a single unambiguous capture.
_URL_PART = {
    "PROTOCOL": r"^([a-zA-Z][a-zA-Z0-9+.-]*):",
    "HOST": r"^[a-zA-Z][a-zA-Z0-9+.-]*://(?:[^@/]*@)?([^:/?#]+)",
    "PATH": r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^/?#]*([^?#]*)",
    "QUERY": r"\?([^#]*)",
    "REF": r"#(.*)$",
    "AUTHORITY": r"^[a-zA-Z][a-zA-Z0-9+.-]*://([^/?#]*)",
    "USERINFO": r"^[a-zA-Z][a-zA-Z0-9+.-]*://([^@/?#]*)@",
}


def _parse_url(tr, node) -> Expr | None:
    """`parse_url(url, 'HOST')` — the named component of a URL, or null if absent.

    `parse_url(url, 'QUERY', 'k')` (the three-argument form that reads one query
    parameter) is declined rather than approximated: it needs the key escaped into the
    pattern, and getting that subtly wrong would return a neighbouring parameter's value.
    """
    from batcher._sql.parser.expressions.literals import _const_str_arg

    if node.args.get("key") is not None:
        return None
    part = _const_str_arg(node.args.get("part_to_extract"), "parse_url()", "part").upper()
    pattern = _URL_PART.get(part)
    if pattern is None:
        return None
    return tr._scalar(node.this).str.regexp_extract(pattern, 1)


def _is_integer_operand(node) -> bool:
    """Whether an argument is an integer value rather than a bit string or text."""
    from sqlglot import expressions as exp

    if isinstance(node, exp.Literal):
        return not node.is_string
    if isinstance(node, exp.Neg):
        return _is_integer_operand(node.this)
    return isinstance(node, exp.Column)


def _utf8_validity(value: Expr, kind: str) -> Expr:
    """`is_valid_utf8` (a predicate) / `validate_utf8` (the identity), preserving nulls."""
    if kind == "identity":
        return value
    return when(value.is_not_null()).then(lit(True)).otherwise(nullif(lit(True), lit(True)))


def _struct_node(tr, node) -> Expr | None:
    """`named_struct('a', 1, …)` / `struct(x, y)` → the engine's struct constructor.

    sqlglot normalizes both spellings to one `Struct` node whose members are
    `PropertyEQ(name, value)` pairs — it has already resolved Spark's positional
    `struct(x, y)` to the `col1`/`col2` names Spark gives those fields — so both
    spellings need only the field pairs read off.
    """
    from sqlglot import expressions as exp

    from batcher.plan.functions.collection import named_struct

    fields = {}
    for member in node.expressions:
        if not isinstance(member, (exp.PropertyEQ, exp.Alias)):
            return None
        fields[member.this.name if isinstance(member, exp.PropertyEQ) else member.alias] = (
            tr._scalar(member.expression if isinstance(member, exp.PropertyEQ) else member.this)
        )
    if not fields:
        return None
    flat: list[object] = []
    for name, value in fields.items():
        flat += [name, value]
    return named_struct(*flat)


def _elt(tr, args) -> Expr | None:
    """`elt(n, a, b, ...)` — the nth argument, 1-based, null outside the range."""
    if len(args) < 2:
        return None
    index = args[0]
    if not isinstance(index, exp.Literal) or index.is_string:
        return None  # a per-row index would need a runtime switch, not a CASE
    position = int(index.this)
    if not 1 <= position < len(args):
        value = tr._scalar(args[1])
        return nullif(value, value)
    return tr._scalar(args[position])


def _width_bucket(tr, node) -> Expr:
    """`width_bucket(v, lo, hi, n)` — the 1-based equi-width bucket of `v`.

    Out-of-range values get the sentinel buckets SQL defines: 0 below `lo` and `n + 1`
    at or above `hi`, which is why this is a CASE rather than the bare arithmetic.
    """
    value = tr._scalar(node.this)
    lo = tr._scalar(node.args["min_value"])
    hi = tr._scalar(node.args["max_value"])
    n = tr._scalar(node.args["num_buckets"])
    inside = ((value - lo) * n / (hi - lo)).floor().cast("int64") + lit(1)
    return when(value < lo).then(lit(0)).when(value >= hi).then(n + lit(1)).otherwise(inside)


def _find_in_set(tr, args) -> Expr | None:
    """`find_in_set(needle, 'a,b,c')` — the 1-based position of `needle`, or 0.

    The needle must be a constant: `.list.position` compares against a scalar value, not
    a per-row expression, so a column needle is declined rather than mistranslated.
    """
    needle = args[0]
    if not isinstance(needle, exp.Literal) or not needle.is_string:
        return None
    parts = tr._scalar(args[1]).str.split(",")
    return coalesce(parts.list.position(needle.this), lit(0))
