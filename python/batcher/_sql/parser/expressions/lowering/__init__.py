"""The scalar lowerings big enough to own a module, kept out of the `scalar` dispatch.

`scalar.py` is the node-type dispatch table; these are the rules it hands off to, each
with a correctness argument of its own: typing an untyped ``NULL``, the set and null-safe
comparisons, ``LIKE`` classification, building a string function whose parameters are
computed per row, and the two dispatches *derived* from the public expression surface
rather than tabulated -- `families` over the free-function library and `accessors` over the
typed accessor namespaces.
"""

from __future__ import annotations

from batcher._sql.parser.expressions.lowering.accessors import (
    accessor_function,
    accessor_vocabulary,
)
from batcher._sql.parser.expressions.lowering.derived import derived_function
from batcher._sql.parser.expressions.lowering.dynamic import (
    const_bool,
    const_float,
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
    "accessor_function",
    "accessor_vocabulary",
    "between",
    "binop_with_null",
    "const_bool",
    "const_float",
    "const_int",
    "const_str",
    "derived_function",
    "dynamic_left",
    "in_membership",
    "is_distinct_from",
    "like",
    "null_boolean",
    "positional_null",
    "str_call",
    "typed_null",
]
