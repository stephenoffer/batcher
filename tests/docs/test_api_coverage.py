"""Mechanical anti-drift gates: every public API name must be documented.

Five checks over the surface that ``tools/public_surface.py`` defines:

1. **Mentioned.** Every public name appears somewhere in the ``docs/**/*.md`` corpus,
   so the hand-maintained reference tables are a checked contract rather than a hope.
2. **Rendered.** Every public name is actually pulled into Sphinx by an
   ``autosummary`` entry, an ``autoclass``/``autofunction`` directive, or the
   ``:members:`` of an autodoc'd class — a name mentioned only in prose has no
   reference page, so its docstring, signature, and examples never reach the site.
3. **Taught.** Every public name appears somewhere a reader *learns* it — a user
   guide, tutorial, getting-started page, ML guide, configuration page, or a runnable
   script under ``examples/`` — not only in the generated reference. A name that only
   an `autosummary` table knows about is a name nobody discovers.
4. **Expression-complete.** The expression reference page (``docs/api/relational/expressions.md``)
   enumerates *every* method callable on an ``Expr`` — the fluent builder and every
   accessor namespace — so the curated reference can't silently fall behind the
   surface (which is exactly how the ``.map``/``.audio``/``.video`` accessors and a
   dozen ``Expr`` methods went missing from it).
5. **Styled.** Every public callable satisfies the Google docstring style
   (``tools/lint_docstrings.py``): one-line summary inline with the quotes, a
   runnable ``Examples:`` doctest, and typeless ``Args:``/``Returns:``.

The v1 docs rotted precisely because nothing enforced any of this. Add a public name
without documenting it and this module fails.
"""

from __future__ import annotations

import importlib
import inspect
import re
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _ROOT / "docs"
_EXAMPLES = _ROOT / "examples"
sys.path.insert(0, str(_ROOT / "tools"))

from lint_docstrings import collect as collect_style_violations  # noqa: E402
from public_surface import expression_names, public_names  # noqa: E402

# The expression reference: the pages whose job is to enumerate every method a user can
# call on an `Expr` — the fluent builder on one, every accessor namespace on the other.
# Together they must be exhaustive; neither alone is.
_EXPR_REFERENCE = (
    "api/relational/expressions.md",
    "api/relational/expressions-datascience.md",
    "api/relational/expression-accessors.md",
)

# Public names not yet documented (drain toward empty). Keep each with a reason.
KNOWN_UNDOCUMENTED: dict[str, str] = {
    # Not user-facing API names.
    "__version__": "package version string, not an API symbol",
}

# Public names deliberately absent from the Sphinx autodoc tree, with a reason.
KNOWN_UNRENDERED: dict[str, str] = {
    "__version__": "package version string, not an API symbol",
}

# Public names that live only in the generated reference, with a reason. Drain toward
# empty: "the reference lists it" is not the same as "a reader can find it".
# Now empty — `examples/operations/environment.py` teaches the last entry.
KNOWN_UNTAUGHT: dict[str, str] = {}

# `Expr` methods absent from the expression reference page, with a reason. Drain toward
# empty: the reference is meant to be exhaustive.
KNOWN_EXPR_UNLISTED: dict[str, str] = {}

# The generated reference (`docs/api/`) and the design docs (`architecture/`,
# `internals/`) don't count as teaching: the first is the thing we're checking against,
# and the second addresses contributors, not users.
_NON_TEACHING_SECTIONS = frozenset({"api", "architecture", "internals"})

_EVAL_RST = re.compile(r"```\{eval-rst\}(.*?)```", re.DOTALL)
_AUTO_DIRECTIVE = re.compile(
    r"^\s*\.\. auto(class|function|data|exception):: ([\w.]+)(.*?)(?=^\s*\.\. |\Z)",
    re.DOTALL | re.MULTILINE,
)
_AUTOSUMMARY = re.compile(r"^\s*\.\. autosummary::(.*?)(?=^\s*\.\. |\Z)", re.DOTALL | re.MULTILINE)
_CURRENTMODULE = re.compile(r"^\s*\.\. currentmodule:: ([\w.]+)", re.MULTILINE)
_BARE_NAME = re.compile(r"^[\w.]+$")


def _doc_files() -> list[Path]:
    return [p for p in _DOCS.rglob("*.md") if "_build" not in p.parts]


def _docs_corpus() -> str:
    """All Markdown under docs/ concatenated (the searchable documentation text)."""
    return "\n".join(p.read_text(encoding="utf-8") for p in _doc_files())


def _teaching_corpus() -> str:
    """The prose a reader learns from: guides, tutorials, and the runnable examples."""
    parts = []
    for page in _doc_files():
        parts_of_path = page.relative_to(_DOCS).parts
        section = parts_of_path[0] if len(parts_of_path) > 1 else ""
        if section in _NON_TEACHING_SECTIONS:
            continue
        parts.append(page.read_text(encoding="utf-8"))
    parts.extend(
        p.read_text(encoding="utf-8")
        for p in sorted(_EXAMPLES.rglob("*.py"))
        if "__pycache__" not in p.parts
    )
    return "\n".join(parts)


def _resolve(dotted: str) -> object:
    """Import the longest module prefix of `dotted`, then walk the attribute tail."""
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj: object = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(dotted)


def _rendered_names() -> set[str]:
    """Bare names Sphinx will emit a reference entry for, across all doc pages."""
    rendered: set[str] = set()
    documented_classes: list[str] = []

    for page in _doc_files():
        # `.. currentmodule::` makes later directives take relative targets. It is
        # document-scoped state in Sphinx, so it carries across `eval-rst` blocks.
        prefix = ""
        for block in _EVAL_RST.findall(page.read_text(encoding="utf-8")):
            scope = _CURRENTMODULE.search(block)
            if scope:
                prefix = f"{scope.group(1)}."
            for kind, target, body in _AUTO_DIRECTIVE.findall(block):
                rendered.add(target.rsplit(".", 1)[-1])
                if kind == "class" and ":members:" in body:
                    documented_classes.append(target if "." in target else prefix + target)
            for body in _AUTOSUMMARY.findall(block):
                for line in body.splitlines():
                    entry = line.strip()
                    if entry and not entry.startswith(":") and _BARE_NAME.match(entry):
                        rendered.add(entry.rsplit(".", 1)[-1])

    # `:members:` renders every public method/property of the class.
    for dotted in documented_classes:
        cls = _resolve(dotted)
        for name, member in vars(cls).items():
            if name.startswith("_"):
                continue
            if inspect.isfunction(member) or isinstance(
                member, (property, staticmethod, classmethod)
            ):
                rendered.add(name)
    return rendered


def _mentioned(name: str, corpus: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", corpus) is not None


def test_every_public_name_is_documented():
    corpus = _docs_corpus()
    names = public_names()
    documented = {n for n in names if _mentioned(n, corpus)}

    missing = sorted(names - documented - set(KNOWN_UNDOCUMENTED))
    assert not missing, (
        f"{len(missing)} public name(s) not documented in docs/: {missing}\n"
        "Document them (e.g. in docs/api/reference.md) or add to KNOWN_UNDOCUMENTED."
    )

    # The allowlist may not list a name that is actually documented (keep it honest).
    stale = sorted(n for n in KNOWN_UNDOCUMENTED if n in documented)
    assert not stale, f"KNOWN_UNDOCUMENTED lists documented names (remove them): {stale}"


def test_every_public_name_is_rendered_by_sphinx():
    rendered = _rendered_names()
    names = public_names()

    missing = sorted(names - rendered - set(KNOWN_UNRENDERED))
    assert not missing, (
        f"{len(missing)} public name(s) have no Sphinx reference entry: {missing}\n"
        "Add each to an `.. autosummary::` list or an `.. autoclass::`/`.. autofunction::` "
        "directive under docs/api/ (a prose mention alone renders no docstring)."
    )

    stale = sorted(n for n in KNOWN_UNRENDERED if n in rendered)
    assert not stale, f"KNOWN_UNRENDERED lists rendered names (remove them): {stale}"


def test_every_public_name_is_taught_outside_the_reference():
    corpus = _teaching_corpus()
    names = public_names()
    taught = {n for n in names if _mentioned(n, corpus)}

    missing = sorted(names - taught - set(KNOWN_UNTAUGHT))
    assert not missing, (
        f"{len(missing)} public name(s) appear only in the generated reference: {missing}\n"
        "Teach each one where a reader would look for it — a user guide, tutorial, ML or "
        "configuration page, or a runnable script under examples/ — or add it to "
        "KNOWN_UNTAUGHT with a reason."
    )

    stale = sorted(n for n in KNOWN_UNTAUGHT if n in taught)
    assert not stale, f"KNOWN_UNTAUGHT lists taught names (remove them): {stale}"


def test_expression_reference_lists_every_expr_method():
    page = "\n".join((_DOCS / name).read_text(encoding="utf-8") for name in _EXPR_REFERENCE)
    names = expression_names()
    listed = {n for n in names if _mentioned(n, page)}

    missing = sorted(names - listed - set(KNOWN_EXPR_UNLISTED))
    assert not missing, (
        f"{len(missing)} Expr method(s) missing from {list(_EXPR_REFERENCE)}: {missing}\n"
        "Those pages are the exhaustive expression reference — add each method to the "
        "relevant table/section, or add it to KNOWN_EXPR_UNLISTED with a reason."
    )

    stale = sorted(n for n in KNOWN_EXPR_UNLISTED if n in listed)
    assert not stale, f"KNOWN_EXPR_UNLISTED lists documented methods (remove them): {stale}"


def test_public_docstrings_follow_the_google_style():
    violations = collect_style_violations()
    if not violations:
        return

    shown = violations[:40]
    lines = "\n".join(f"  {v.name} ({Path(v.file).name}:{v.line}): {v.rule}" for v in shown)
    elided = "" if len(violations) == len(shown) else f"\n  ... and {len(violations) - 40} more"
    pytest.fail(
        f"{len(violations)} docstring style violation(s) on the public API:\n{lines}{elided}\n\n"
        "Run `just lint-docstrings` for the full report. The style is specified in "
        ".claude/rules/python-quality.md."
    )
