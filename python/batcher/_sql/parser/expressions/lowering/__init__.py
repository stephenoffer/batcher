"""The scalar lowerings big enough to own a module, kept out of the `scalar` dispatch.

`scalar.py` is the node-type dispatch table; these are the four rules it hands off to,
each with a correctness argument of its own: typing an untyped ``NULL``, the set and
null-safe comparisons, ``LIKE`` classification, and building a string function whose
parameters are computed per row.
"""

from __future__ import annotations

from batcher._sql.parser.expressions.lowering.dynamic import (
    const_int,
    const_str,
    dynamic_left,
    str_call,
)
from batcher._sql.parser.expressions.lowering.matching import like
from batcher._sql.parser.expressions.lowering.membership import (
    between,
    in_membership,
    is_distinct_from,
)
from batcher._sql.parser.expressions.lowering.nulls import (
    binop_with_null,
    null_boolean,
    positional_null,
    typed_null,
)

__all__ = [
    "between",
    "binop_with_null",
    "const_int",
    "const_str",
    "dynamic_left",
    "in_membership",
    "is_distinct_from",
    "like",
    "null_boolean",
    "positional_null",
    "str_call",
    "typed_null",
]
