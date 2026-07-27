"""No-argument string folds: lengths, digests, hex, the pads, and the trims.

`extra/strings` folds `upper`/`lower`, `len`, `substr`, and `concat` over a literal.
This module covers the rest of the surface: the length variants, the digests, the
padding and repetition builders, and the trims. A string function applied to a
literal is evaluated once here instead of once per row, and -- more usefully -- the
resulting literal feeds constant propagation, `IN`-list folding, and the sargable
normalizers, none of which can see through a function call.

Every fold here must be checked against the engine value by value, never assumed from the
function's name, because Python and Rust agree on fewer string operations than they appear
to: the digests and the length functions agree exactly, while anything touching case or
whitespace diverges outside ASCII.

Three folds shipped without that check and disagreed with the runtime. `hex` used
`bytes.hex()`, which is lowercase where the engine returns uppercase, so
`hex(col) = hex('needle')` compared a lowercase constant against an uppercase column and
matched nothing. `lpad`/`rpad` used `str.rjust`/`str.ljust`, which return the input untouched
when it is already wider than the target, where SQL truncates it -- so
`lpad('Hello World', 10)` folded to `'Hello World'` against the runtime's `'Hello Worl'`.
`tests/differential/test_diff_expr_rewrite_rules.py` now asserts fold-equals-runtime for every
fold in this module, which is the check that would have caught all three.

Two guards therefore recur, and both are the reason a rule declines rather than
guesses:

* **ASCII only, where the operation is locale- or Unicode-sensitive.** `initcap`,
  `reverse`, and the trims are restricted to ASCII input. Full Unicode case mapping
  is where implementations legitimately differ (`upper('ss')` is two characters in
  one engine and one in another), Rust's `trim` strips the Unicode whitespace set
  while Python's `strip` strips its own, and reversing text with combining marks is
  defined per code point in both but is not worth asserting from an ASCII test.
* **Default arguments only.** A `trim` carrying an explicit character set, or a pad
  with a multi-character fill, is left alone: the fold reproduces the default
  behaviour and nothing more.

The digests (`md5`, `sha1`, `sha256`, `crc32`) and `hex` need neither guard. They are
byte-exact functions of the UTF-8 encoding with one right answer -- but "one right answer"
covers the bytes, not their presentation, which is exactly where `hex` went wrong.
"""

from __future__ import annotations

import hashlib
import zlib

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.exprs.text_folds.literals import (
    MAX_FOLDED_CHARS,
    ascii_text,
    fold,
    literal_text,
    plain_text_fold,
)
from batcher.kyber.rules.leaf_rewrite import rewrite_node
from batcher.plan.expr_ir.func_nodes import StrFunc
from batcher.plan.logical import Filter, LogicalPlan, Project

__all__ = [
    "fold_ascii_of_literal",
    "fold_bit_length_of_literal",
    "fold_crc32_of_literal",
    "fold_hex_of_literal",
    "fold_initcap_of_literal",
    "fold_left_pad_of_literal",
    "fold_left_trim_of_literal",
    "fold_md5_of_literal",
    "fold_octet_length_of_literal",
    "fold_repeat_of_literal",
    "fold_reverse_of_literal",
    "fold_right_pad_of_literal",
    "fold_right_trim_of_literal",
    "fold_sha1_of_literal",
    "fold_sha256_of_literal",
    "fold_trim_of_literal",
]

# --- lengths ------------------------------------------------------------------

_OCTET_LENGTH = plain_text_fold("octet_length", lambda s: len(s.encode()))
_BIT_LENGTH = plain_text_fold("bit_length", lambda s: 8 * len(s.encode()))
_ASCII = plain_text_fold("ascii", lambda s: ord(s[0]) if s else None)


@rule(
    name="fold_octet_length_of_literal",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_OCTET_LENGTH,
)
def fold_octet_length_of_literal(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`octet_length('abc') -> 3`. The byte length of the UTF-8 encoding, which Python
    reproduces exactly with `len(s.encode())`. No ASCII guard is needed: this counts
    bytes, and the encoding is the same on both sides."""
    return rewrite_node(node, _OCTET_LENGTH)


@rule(
    name="fold_bit_length_of_literal",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_BIT_LENGTH,
)
def fold_bit_length_of_literal(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`bit_length('abc') -> 24`. Eight times the octet length, verified against the
    engine."""
    return rewrite_node(node, _BIT_LENGTH)


@rule(name="fold_ascii_of_literal", phase=Phase.NORMALIZE, matches=(Filter, Project), expr=_ASCII)
def fold_ascii_of_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`ascii('abc') -> 97`, the code point of the first character. Declines on the
    empty string, where there is no first character and the engine's answer is not
    something this rule should assume."""
    return rewrite_node(node, _ASCII)


# --- digests ------------------------------------------------------------------

_MD5 = plain_text_fold("md5", lambda s: hashlib.md5(s.encode()).hexdigest())
_SHA1 = plain_text_fold("sha1", lambda s: hashlib.sha1(s.encode()).hexdigest())
_SHA256 = plain_text_fold("sha256", lambda s: hashlib.sha256(s.encode()).hexdigest())
_CRC32 = plain_text_fold("crc32", lambda s: zlib.crc32(s.encode()))
# `bytes.hex()` is lowercase; the engine and DuckDB both return uppercase, and a fold that
# disagrees with the runtime makes `hex(col) = hex('needle')` unsatisfiable.
_HEX = plain_text_fold("hex", lambda s: s.encode().hex().upper())


@rule(name="fold_md5_of_literal", phase=Phase.NORMALIZE, matches=(Filter, Project), expr=_MD5)
def fold_md5_of_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`md5('abc')` folds to its hex digest. A digest of a constant is a constant, and
    MD5 has exactly one answer for a given byte string -- verified against `hashlib`.
    Folding it removes a per-row hash over a value that never changes."""
    return rewrite_node(node, _MD5)


@rule(name="fold_sha1_of_literal", phase=Phase.NORMALIZE, matches=(Filter, Project), expr=_SHA1)
def fold_sha1_of_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`sha1('abc')` folds to its hex digest, verified against `hashlib`."""
    return rewrite_node(node, _SHA1)


@rule(name="fold_sha256_of_literal", phase=Phase.NORMALIZE, matches=(Filter, Project), expr=_SHA256)
def fold_sha256_of_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`sha256('abc')` folds to its hex digest, verified against `hashlib`. The most
    valuable member of the group, since SHA-256 is the most expensive of them per
    row."""
    return rewrite_node(node, _SHA256)


@rule(name="fold_crc32_of_literal", phase=Phase.NORMALIZE, matches=(Filter, Project), expr=_CRC32)
def fold_crc32_of_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`crc32('abc') -> 891568578`. CRC-32 is fully specified by its polynomial, so
    `zlib.crc32` and the engine agree by construction -- checked, not assumed."""
    return rewrite_node(node, _CRC32)


@rule(name="fold_hex_of_literal", phase=Phase.NORMALIZE, matches=(Filter, Project), expr=_HEX)
def fold_hex_of_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`hex('abc') -> '616263'`, the uppercase hex of the UTF-8 bytes.

    Byte-exact on both sides, so no ASCII guard is required -- but the *case* is not
    incidental. `bytes.hex()` is lowercase and the engine returns uppercase, which made
    `hex(col) = hex('needle')` compare a lowercase constant against an uppercase column and
    never match. Verified against the runtime and DuckDB rather than assumed."""
    return rewrite_node(node, _HEX)


# --- shape-changing builders --------------------------------------------------


def _repeat(expr: StrFunc):
    text = literal_text(expr.input)
    count = expr.start
    if text is None or count is None or count < 0:
        return None
    if len(text) * count > MAX_FOLDED_CHARS:
        return None
    return text * count


def _pad(justify):
    def compute(expr: StrFunc):
        text = ascii_text(expr.input)
        width = expr.start
        fill = expr.pattern
        if text is None or width is None or width < 0 or width > MAX_FOLDED_CHARS:
            return None
        if fill is None or len(fill) != 1:
            return None
        # SQL `lpad`/`rpad` *truncate* a string longer than the target width, taking its first
        # `width` characters. Python's `rjust`/`ljust` return the input untouched instead, so
        # folding a literal wider than the pad disagreed with the engine: `lpad('Hello World', 10)`
        # folded to 'Hello World' where the runtime and DuckDB both give 'Hello Worl'. Truncating
        # first leaves the padding branch unchanged, because `text[:width]` is `text` there.
        return justify(text[:width], width, fill)

    return compute


_REPEAT = fold("repeat", _repeat)
_LPAD = fold("lpad", _pad(lambda s, w, f: s.rjust(w, f)))
_RPAD = fold("rpad", _pad(lambda s, w, f: s.ljust(w, f)))
_REVERSE = plain_text_fold("reverse", lambda s: s[::-1], ascii_only=True)
_INITCAP = plain_text_fold("initcap", lambda s: s.title(), ascii_only=True)


@rule(name="fold_repeat_of_literal", phase=Phase.NORMALIZE, matches=(Filter, Project), expr=_REPEAT)
def fold_repeat_of_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`repeat('ab', 3) -> 'ababab'`.

    Capped at `MAX_FOLDED_CHARS`. `repeat` is the one function here that can grow its
    input without bound, and a plan literal is materialized into every batch -- so
    folding a megabyte-long constant would trade a cheap per-row build for an
    expensive per-batch copy. Above the cap the rule declines and the engine builds it
    as before."""
    return rewrite_node(node, _REPEAT)


@rule(name="fold_left_pad_of_literal", phase=Phase.NORMALIZE, matches=(Filter, Project), expr=_LPAD)
def fold_left_pad_of_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`lpad('abc', 5) -> '  abc'`. Width counts characters on both sides, so Python's
    `str.rjust` matches. Restricted to a single-character fill and to ASCII input,
    which is where the character-count equivalence was verified."""
    return rewrite_node(node, _LPAD)


@rule(
    name="fold_right_pad_of_literal", phase=Phase.NORMALIZE, matches=(Filter, Project), expr=_RPAD
)
def fold_right_pad_of_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`rpad('abc', 5) -> 'abc  '`, the `lpad` mirror with the same guards."""
    return rewrite_node(node, _RPAD)


@rule(
    name="fold_reverse_of_literal", phase=Phase.NORMALIZE, matches=(Filter, Project), expr=_REVERSE
)
def fold_reverse_of_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`reverse('abc') -> 'cba'`. ASCII-guarded: both engines reverse code points, but
    that equivalence was only verified on ASCII, and text carrying combining marks is
    not worth asserting from an ASCII test."""
    return rewrite_node(node, _REVERSE)


@rule(
    name="fold_initcap_of_literal", phase=Phase.NORMALIZE, matches=(Filter, Project), expr=_INITCAP
)
def fold_initcap_of_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`initcap('hello world') -> 'Hello World'`. ASCII-guarded, because title-casing
    is exactly the operation where Unicode implementations diverge."""
    return rewrite_node(node, _INITCAP)


# --- trims --------------------------------------------------------------------


def _trim(strip):
    def compute(expr: StrFunc):
        # A custom character set is a different function; only the whitespace default
        # is reproduced here.
        if expr.pattern is not None:
            return None
        text = ascii_text(expr.input)
        return None if text is None else strip(text)

    return compute


_TRIM = fold("trim", _trim(str.strip))
_LTRIM = fold("l_trim", _trim(str.lstrip))
_RTRIM = fold("r_trim", _trim(str.rstrip))


@rule(name="fold_trim_of_literal", phase=Phase.NORMALIZE, matches=(Filter, Project), expr=_TRIM)
def fold_trim_of_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`trim('  ab  ') -> 'ab'`.

    Two guards. A `trim` carrying an explicit character set is left alone -- that is a
    different function, and this fold only reproduces the whitespace default. And the
    input must be ASCII, because Rust's `trim` strips the Unicode whitespace set while
    Python's `strip` strips its own; the two coincide on ASCII and are not guaranteed
    to elsewhere."""
    return rewrite_node(node, _TRIM)


@rule(
    name="fold_left_trim_of_literal", phase=Phase.NORMALIZE, matches=(Filter, Project), expr=_LTRIM
)
def fold_left_trim_of_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`l_trim('  ab  ') -> 'ab  '`, with the same two guards as `trim`."""
    return rewrite_node(node, _LTRIM)


@rule(
    name="fold_right_trim_of_literal",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_RTRIM,
)
def fold_right_trim_of_literal(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`r_trim('  ab  ') -> '  ab'`, with the same two guards as `trim`."""
    return rewrite_node(node, _RTRIM)
