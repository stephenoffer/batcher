"""The migration templates for string calls that need no parameter, against each engine's rule.

Some Spark and Ray Data string calls differ from their Batcher namesake in a way a codemod
can undo at rewrite time with spellings that already exist: a literal start position of
``0``, a negative count, a stop index instead of a length. No parameter should model those,
so the codemod emits a template instead, and a template is only as good as the rows it was
checked on. Each test states the template and holds it against the other engine as the
oracle. Ray Data's string namespace is Arrow compute
(`ray/data/namespace_expressions/string_namespace.py`) and runs here for real. Spark has no
JVM in this environment, so its rule is transcribed from the Spark source cited beside it
and checked on its documented examples.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt
from batcher.plan.expr_ir.namespaces._dialect import escape_rust_regex

pytestmark = pytest.mark.differential

_WORDS = ["hello", "", None, "héllo wörld", "ab", "Spark SQL"]
_ASCII = ["hello", "", None, "a.b.c", "aaa", "Spark SQL"]

s = bt.col("s")


def _values(expr, words=_WORDS) -> list:
    return bt.from_pydict({"s": words}).select(r=expr).to_pydict()["r"]


def _spark_substring(value: str, pos: int, length: int) -> str:
    """`UTF8String.substringSQL`: position 0 is the first character, a length below 1 is ''."""
    start = pos - 1 if pos > 0 else (len(value) + pos if pos < 0 else 0)
    end = start + length
    start = max(start, 0)
    return "" if start >= end else value[start:end]


@pytest.mark.parametrize("pos", [-20, -3, -1, 0, 1, 2, 6, 30])
@pytest.mark.parametrize("length", [-2, 0, 1, 3, 40])
def test_spark_substring_template(pos, length):
    # F.substring(c, pos, len) -> c.str.substr(pos or 1, max(len, 0))
    template = s.str.substr(pos or 1, max(length, 0))
    want = [None if w is None else _spark_substring(w, pos, length) for w in _WORDS]
    assert _values(template) == want


@pytest.mark.parametrize("n", [-3, 0, 1, 4, 30])
def test_spark_left_and_right_templates(n):
    # `stringExpressions.scala`: Left is Substring(str, 1, len); Right is '' for len <= 0 and
    # Substring(str, -len) otherwise. Both documented examples on 'Spark SQL' with 3.
    left = [None if w is None else _spark_substring(w, 1, n) for w in _WORDS]
    right = [
        None if w is None else "" if n <= 0 else _spark_substring(w, -n, 2**31 - 1) for w in _WORDS
    ]
    assert _values(s.str.left(max(n, 0))) == left
    assert _values(s.str.right(max(n, 0))) == right
    assert _values(s.str.left(3), ["Spark SQL"]) == ["Spa"]
    assert _values(s.str.right(3), ["Spark SQL"]) == ["SQL"]


def test_spark_hex_of_an_integer_is_already_hex():
    # `mathExpressions.scala`, `Hex`: hex(17) -> '11', hex('Spark SQL') -> '537061726B2053514C';
    # a negative long is its two's complement (`Long.toHexString`).
    assert _values(s.str.hex(), [17, -1, 0, None]) == ["11", "FFFFFFFFFFFFFFFF", "0", None]
    assert _values(s.str.hex(), ["Spark SQL"]) == ["537061726B2053514C"]


def test_spark_to_binary_templates():
    # `stringExpressions.scala`: to_binary('abc', 'utf-8') -> abc; `unhex('537061726B2053514C')`
    # decodes to 'Spark SQL'; unbase64('U3BhcmsgU1FM') -> 'Spark SQL'.
    assert _values(s.cast("binary"), ["abc"]) == [b"abc"]
    assert _values(s.str.unhex(as_binary=True), ["537061726B2053514C"]) == [b"Spark SQL"]
    assert _values(s.str.from_base64(as_binary=True), ["U3BhcmsgU1FM"]) == [b"Spark SQL"]


@pytest.mark.parametrize("pattern", ["l", "ö", "", "zz", ".", "llo"])
def test_ray_find_template(pattern):
    # Arrow `find_substring`: 0-based byte offset, -1 when absent.
    before = s.str.regexp_split(escape_rust_regex(pattern), limit=2).list.get(0)
    template = (
        bt.when(s.str.contains(pattern))
        .then(before.str.octet_length())
        .when(s.is_not_null())
        .then(bt.lit(-1))
        .otherwise(bt.lit(None))
    )
    want = pc.find_substring(pa.array(_WORDS), pattern).to_pylist()
    assert _values(template) == want


@pytest.mark.parametrize("pattern", ["l", "aa", ".", "é"])
def test_ray_count_is_the_literal_match_count(pattern):
    want = pc.count_substring(pa.array([*_WORDS, "aaaa"]), pattern).to_pylist()
    assert _values(s.str.regexp_count(pattern, literal=True), [*_WORDS, "aaaa"]) == want


@pytest.mark.parametrize("pattern", ["h%", "%o", "_e%", "%", "a.b", "%\u00f6%"])
def test_ray_match_is_like(pattern):
    want = pc.match_like(pa.array(_WORDS), pattern).to_pylist()
    assert _values(s.str.like(pattern)) == want


@pytest.mark.parametrize(("start", "stop"), [(0, 0), (1, 3), (2, 10), (0, 1)])
def test_ray_replace_slice_template_on_ascii(start, stop):
    # Arrow `binary_replace_slice` counts bytes, which is why the template is exact on ASCII.
    template = s.str.overlay("XY", start + 1, stop - start)
    want = pc.binary_replace_slice(pa.array(_ASCII), start, stop, "XY").to_pylist()
    assert _values(template, _ASCII) == want


@pytest.mark.parametrize(("start", "stop"), [(0, 2), (1, 4), (3, 50), (-3, -1), (0, 0)])
def test_ray_slice_template(start, stop):
    # Arrow `utf8_slice_codeunits` counts characters; the template holds where both ends share
    # a sign, which is every call a codemod can rewrite without reading the data.
    template = s.str.slice(start, stop - start)
    want = pc.utf8_slice_codeunits(pa.array(_WORDS), start, stop).to_pylist()
    assert _values(template) == want
