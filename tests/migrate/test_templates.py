"""The registry's template DSL: validated at load, bound like Python, and never half-applied.

A template is the one place a registry row says "this call becomes that code". A malformed one
must fail when the registry loads rather than when a user runs the codemod, a call that does not
bind must leave the source alone, and every `sem.<name>` a template calls must exist.
"""

from __future__ import annotations

import ast
import textwrap

import pytest

pytest.importorskip("libcst")

import libcst as cst

from batcher._internal.migration import RegistryError, load_registry, parse_template
from batcher._internal.migration.schema import validate
from batcher.migrate.semantics import TRANSFORMS, java_to_strftime, lookup
from batcher.migrate.templates import Bound, Signature, render
from batcher.migrate.translate import translate


def test_a_template_calling_an_unknown_name_fails_at_load() -> None:
    raw = {
        "status": "canonical",
        "batcher": "Dataset.limit",
        "template": "take(n) -> self.limit(m)",
    }
    with pytest.raises(RegistryError, match="unknown name 'm'"):
        validate("pyspark", "DataFrame", "take", raw)


def test_a_template_for_another_name_fails_at_load() -> None:
    raw = {
        "status": "canonical",
        "batcher": "Dataset.limit",
        "template": "head(n) -> self.limit(n)",
    }
    with pytest.raises(RegistryError, match="does not rewrite 'take'"):
        validate("pyspark", "DataFrame", "take", raw)


def test_a_reversible_template_must_be_a_rebinding() -> None:
    with pytest.raises(RegistryError, match="may only re-bind"):
        parse_template("take(n) <-> self.limit(n + 1)", "take")
    assert parse_template("head(n=5) <-> self.limit(n)", "head").reversible


def test_every_registry_template_parses_and_calls_only_real_transforms() -> None:
    templates = [r for r in load_registry().rows.values() if r.template is not None]
    assert len(templates) >= 60, f"only {len(templates)} templates; the registry lost some"
    for row in templates:
        parsed = parse_template(str(row.template), row.name)
        for node in ast.walk(ast.parse(parsed.target, mode="eval")):
            if isinstance(node, ast.Attribute) and getattr(node.value, "id", None) == "sem":
                assert node.attr in TRANSFORMS, f"{row.surface}.{row.name}: sem.{node.attr}"


class _Context:
    engine = "polars"
    bt = "bt"

    def receiver(self, _original: object) -> None:
        return None

    def rewritten(self, original: object) -> object:
        return original

    def consume(self, _original: object) -> None:
        return None

    def definition(self, _name: str) -> None:
        return None

    def note(self, _text: str) -> None:
        return None


def _render(template: str, call: str) -> str | None:
    node = cst.parse_expression(call)
    base = node.func.value
    parsed = parse_template(template, node.func.attr.value)
    new = render(parsed, list(node.args), list(node.args), Bound(base, base), lookup(_Context()))
    return None if new is None else cst.Module([]).code_for_node(new)


def test_render_binds_like_python_and_spells_defaults_explicitly() -> None:
    template = 'with_row_count(name="row_nr", offset=0) -> self.with_row_index(name, offset=offset)'
    assert _render(template, "df.with_row_count()") == 'df.with_row_index("row_nr", offset=0)'
    assert (
        _render(template, "df.with_row_count(offset=5)") == 'df.with_row_index("row_nr", offset=5)'
    )


def test_render_declines_rather_than_guesses() -> None:
    template = "take(num) -> self.limit(num).to_pylist()"
    assert _render(template, "df.take(3)") == "df.limit(3).to_pylist()"
    assert _render(template, "df.take()") is None  # a required argument is missing
    assert _render(template, "df.take(3, 4)") is None  # a surplus positional
    assert _render(template, "df.take(*sizes)") is None  # a starred argument cannot bind


def test_a_keyword_dict_becomes_keywords_and_absent_none_defaults_are_dropped() -> None:
    template = "withColumn(colName, col) -> self.with_columns(**{colName: col})"
    assert _render(template, 'df.withColumn("total", a + b)') == "df.with_columns(total=a + b)"
    assert _render(template, 'df.withColumn("a b", x)') == 'df.with_columns(**{"a b": x})'
    ray = "map_batches(fn, *, batch_size=None) -> self.map_batches(fn, batch_size=batch_size)"
    assert _render(ray, "ds.map_batches(f)") == "ds.map_batches(f)"
    assert _render(ray, "ds.map_batches(f, batch_size=8)") == "ds.map_batches(f, batch_size=8)"


def test_signature_refuses_a_foreign_option_passing_through_kwargs() -> None:
    target = Signature.parse(["*keys", "**named"])  # Batcher `group_by(*keys, **named)`
    source = Signature.parse(["*by", "maintain_order=", "**named_by"])  # Polars `group_by`
    assert target.accepts(1, [], source)
    assert target.accepts(1, ["bucket"], source)  # a named key passes through **named
    assert not target.accepts(1, ["maintain_order"], source)  # a Polars option does not


@pytest.mark.parametrize(
    ("java", "strftime"),
    [
        ("yyyy-MM-dd", "%Y-%m-%d"),
        ("dd/MM/yyyy HH:mm:ss", "%d/%m/%Y %H:%M:%S"),
        ("yyyy-MM-dd'T'HH:mm", "%Y-%m-%dT%H:%M"),
        ("EEE, MMM d", None),  # a single `d` is not zero-padded: no exact strftime
        ("HH:mm:ss.SSS", None),
    ],
)
def test_java_patterns_translate_only_when_exact(java: str, strftime: str | None) -> None:
    assert java_to_strftime(java) == strftime


def test_a_declined_transform_leaves_the_call_as_written() -> None:
    # A date pattern held in a variable cannot be translated, so the call keeps its Spark form.
    source = textwrap.dedent(
        """
        from pyspark.sql import functions as F
        pattern = "yyyy"
        a = F.date_format(F.col("ts"), pattern)
        b = F.date_format(F.col("ts"), "yyyy")
        """
    )
    out, report = translate(source, "pyspark")
    assert 'a = F.date_format(F.col("ts"), pattern)' in out
    assert 'b = bt.col("ts").dt.strftime("%Y")' in out
    assert {s.action for s in report.sites if s.spelling == "functions.date_format"} == {
        "marked",
        "rewritten",
    }
