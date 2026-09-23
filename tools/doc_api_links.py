#!/usr/bin/env python3
"""Resolve a Batcher API spelling to a Sphinx cross-reference, and find the ones nobody did.

A reference page is only a reference if its names are clickable. Most of `docs/` gets this
right by hand, and the places that don't are the places a reader needs it most: the
generated migration tables, where every row names a Batcher method and none of them linked,
and the lookup tables on the API pages. 3,592 inline code spans named a public API symbol
and rendered as grey text.

Linking them by hand is not the fix, because a hand-written link is a hand-written
guess. `{py:meth}` against a target Sphinx never registered renders as plain text and
warns about nothing: `nitpicky` is off, so a dead cross-reference is indistinguishable
from an unlinked one in a green `-W` build. So this module derives the set of targets
Sphinx *will* register, from the same directives that register them, and refuses to emit a
reference to anything outside it.

Where the targets come from
---------------------------
Every public name reaches the built site through one of two directives in an
``eval-rst`` block under ``docs/api/``:

- ``.. autosummary:: :toctree: generated`` with the name indented under it. This is how
  the class members are published: ``api/complete/dataset.md`` lists 178 of them, each
  getting its own generated page and its own ``py:method`` target.
- ``.. autoclass::`` / ``.. autofunction::`` / ``.. autodata::``, which register the
  object named in the argument.

Both are read relative to the enclosing ``.. currentmodule::``, exactly as Sphinx reads
them. Nothing here imports `batcher`, so it runs without a built engine, and it cannot
drift from the site: if a page stops publishing a name, this stops linking it in the same
run.

Usage::

The same pass links a repo path -- ``crates/bc-runtime/src/agg/mod.rs`` -- to the file on
GitHub, for the same reason and with the same rule: only when the path exists.

    python tools/doc_api_links.py --list            # every spelling that resolves
    python tools/doc_api_links.py --unlinked        # spans that name a symbol and don't link
    python tools/doc_api_links.py --unlinked --max 40
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"

_EVAL_RST = re.compile(r"```\{eval-rst\}(.*?)```", re.DOTALL)
_CURRENTMODULE = re.compile(r"^\s*\.\.\s+currentmodule::\s*(\S+)\s*$")
_AUTOSUMMARY = re.compile(r"^\s*\.\.\s+autosummary::\s*$")
_AUTODOC = re.compile(r"^\s*\.\.\s+auto(class|function|method|data|attribute|property)::\s*(\S+)")
_OPTION = re.compile(r"^\s*:[\w-]+:")
_ENTRY = re.compile(r"^\s+([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)\s*$")

#: Where a repo path in the prose points. Architecture and benchmark pages cite the file a
#: mechanism lives in, which is the most useful link on the page for anyone checking the
#: claim, and 330 of them were plain text. `docs/` paths are excluded: a page citing another
#: page wants a `{doc}` reference, which Sphinx checks, not a link out to GitHub.
SOURCE_URL = "https://github.com/stephenoffer/batcher/blob/main/"
_SOURCE_DIR_URL = "https://github.com/stephenoffer/batcher/tree/main/"
_SOURCE_ROOTS = ("examples/", "benchmarks/", "tools/", "tests/", "crates/", "python/")

#: The role used for every generated reference. ``py:obj`` resolves against any object
#: type in the Python domain, so one role covers a method, a property, a function, a class
#: and an attribute alike. ``py:meth`` does not: a property published by ``autoattribute``
#: is a ``py:attribute`` target, and a ``{py:meth}`` at it silently renders as plain text.
#: Picking the narrow role per name would mean tracking which directive published each
#: one, to buy nothing a reader can see.
ROLE = "py:obj"


def _targets_in(text: str) -> set[str]:
    """Every fully qualified target the autodoc directives in one page register."""
    found: set[str] = set()
    # ``currentmodule`` is a document-level setting, not a block-level one: a page sets it
    # once and every later ``eval-rst`` block inherits it. Resetting it per block is why
    # this first read indexed ``Dataset.select`` as its own target instead of
    # ``batcher.Dataset.select`` -- the autoclass and the autosummary listing its members
    # are in different blocks, and only the first one carries the directive.
    module = ""
    for block in _EVAL_RST.findall(text):
        in_summary = False
        for line in block.splitlines():
            if not line.strip():
                continue
            if (m := _CURRENTMODULE.match(line)) is not None:
                module, in_summary = m.group(1), False
                continue
            if _AUTOSUMMARY.match(line):
                in_summary = True
                continue
            if (m := _AUTODOC.match(line)) is not None:
                name = m.group(2)
                absolute = "." in name and name.startswith("batcher")
                found.add(name if absolute else f"{module}.{name}")
                in_summary = False
                continue
            if _OPTION.match(line):
                continue
            if in_summary and (m := _ENTRY.match(line)) is not None:
                name = m.group(1)
                found.add(name if name.startswith("batcher.") else f"{module}.{name}")
                continue
            if not line.startswith((" ", "\t")):
                in_summary = False
    return {t.strip(".") for t in found if t.strip(".")}


def _spellings(target: str) -> list[str]:
    """The ways a page is likely to write `target` in prose or a table.

    A reader writes ``Dataset.select``, ``bt.col`` or ``Expr.cast``, never
    ``batcher.plan.expr_ir.core.Expr.cast``. So each target is indexed under the
    class-qualified tail (when the owner is a class, which is what the leading capital
    marks) and under the ``bt.``-prefixed form when it hangs directly off the root
    package. A bare method name is deliberately not indexed: ``select`` alone is a word.
    """
    parts = target.split(".")
    out = [target]
    if len(parts) >= 2 and parts[-2][:1].isupper():
        out.append(".".join(parts[-2:]))  # Dataset.select, Expr.cast
        if len(parts) >= 3 and parts[-3][:1].isupper():
            out.append(".".join(parts[-3:]))  # _StrNamespace.replace stays qualified
    if parts[0] == "batcher" and len(parts) == 2:
        out.append(f"bt.{parts[1]}")  # bt.col, bt.read
        out.append(parts[1] if parts[1][:1].isupper() else f"bt.{parts[1]}")
    return out


@lru_cache(maxsize=1)
def index(docs: Path = DOCS) -> dict[str, str]:
    """Map every unambiguous API spelling to the target Sphinx registers for it.

    A spelling two different targets both answer to is dropped rather than guessed at.
    """
    targets: set[str] = set()
    for page in sorted(docs.rglob("*.md")):
        if "_build" in page.parts:
            continue
        targets |= _targets_in(page.read_text(encoding="utf-8"))
    table: dict[str, set[str]] = {}
    for target in targets:
        for spelling in _spellings(target):
            table.setdefault(spelling, set()).add(target)
    return {k: next(iter(v)) for k, v in table.items() if len(v) == 1}


_PATH = re.compile(r"^[\w.]+(?:/[\w.-]+)+$")


def source_path(spelling: str) -> str | None:
    """`spelling` as a repo-relative path to a file that exists, or None.

    A directory is deliberately included: ``tests/differential/`` is as worth clicking as
    one file in it. A path that does not exist is left alone rather than linked, so a
    renamed file degrades to plain text instead of to a 404.
    """
    name = spelling.strip().rstrip("/")
    if not _PATH.match(name) or not name.startswith(_SOURCE_ROOTS):
        return None
    return name if (REPO / name).exists() else None


#: A call spelling carries its arguments: ``bt.from_pydict(mapping)``. The name is what
#: resolves, the arguments are display.
_CALL = re.compile(r"^([A-Za-z_][\w.]*)\((.*)\)$", re.DOTALL)


def _single_call(spelling: str) -> str | None:
    """`spelling` as one name plus its argument list, or None if it is anything else.

    The distinction matters more than it looks. ``bt.col("price") * bt.col("qty")`` and
    ``bt.numeric().name.prefix("n_").round(2)`` both start with a resolvable name and end
    in a paren, so a regex that takes everything between the first ``(`` and the last
    ``)`` as arguments calls them both a reference to their first name. They are neither:
    they are illustrative code, and wrapping the whole expression in a link to ``col`` or
    ``numeric`` would be worse than leaving them plain. Only a span that is exactly one
    call is a lookup, so the parentheses have to balance to zero exactly once, at the end.
    """
    if (m := _CALL.match(spelling)) is None:
        return None
    depth = 0
    for ch in m.group(2):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return None
    return m.group(1) if depth == 0 else None


def resolve(spelling: str) -> str | None:
    """The target for `spelling`, or None when nothing published it."""
    name = spelling.strip()
    if name.endswith(")"):
        if (call := _single_call(name)) is None:
            return None
        name = call
    name = name.lstrip(".")
    return index().get(name)


def source(spelling: str) -> str:
    """`spelling` as a Markdown link to the file on GitHub, or as a plain code span."""
    path = source_path(spelling)
    if path is None:
        return f"`{spelling}`"
    base = _SOURCE_DIR_URL if (REPO / path).is_dir() else SOURCE_URL
    return f"[`{spelling}`]({base}{path})"


def link(spelling: str, *, display: str | None = None) -> str:
    """`spelling` as a MyST cross-reference, or as a code span when it resolves to nothing.

    The return value is always valid Markdown for a table cell or a sentence, so a caller
    never has to branch on whether a name happens to be published.
    """
    shown = display if display is not None else spelling
    target = resolve(spelling)
    if target is None:
        return f"`{shown}`"
    return f"{{{ROLE}}}`{shown} <{target}>`"


# --- the linter half -------------------------------------------------------------

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_ROLE_SPAN = re.compile(r"\{[a-z:]+\}`[^`]*`")
_MYST_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
_SPAN = re.compile(r"`([^`\n]+)`")

#: Pages whose links are the generator's decision, not an author's. The migration tables
#: put the *other* engine's names in one column, and Polars, Daft and Ray Data each have a
#: class called ``Dataset`` or ``Expr``, so a scan of the rendered page cannot tell a
#: Batcher ``Expr.cast`` from a Polars one. Linking those would point a reader at the wrong
#: engine's reference, which is worse than not linking them.
_GENERATED = (
    "docs/getting-started/migration/spark/",
    "docs/getting-started/migration/polars/",
    "docs/getting-started/migration/daft/",
    "docs/getting-started/migration/ray-data/",
)


def is_generated(page: Path) -> bool:
    """Whether `page` is written by a generator rather than by hand."""
    rel = page.resolve().relative_to(REPO).as_posix()
    return rel.startswith(_GENERATED)


@lru_cache(maxsize=1)
def _excluded() -> tuple[str, ...]:
    """The ``exclude_patterns`` prefixes in ``docs/conf.py``, as posix path prefixes.

    A contributor working record is not a page a reader clicks through, so it owes no
    cross-references. Reading the list from ``conf.py`` rather than restating it here is
    the same reason ``tests/docs/test_docs_structure.py`` does: a record added to an
    excluded directory should not become a lint failure nobody expected.
    """
    conf = (DOCS / "conf.py").read_text(encoding="utf-8")
    tree = ast.parse(conf)
    out: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(x, ast.Name) and x.id == "exclude_patterns" for x in node.targets)
            and isinstance(node.value, ast.List)
        ):
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    out.append(f"docs/{elt.value.removesuffix('*')}")
    return tuple(out)


def in_scope(page: Path) -> bool:
    """Whether `page` is a published page whose links are an author's to get right."""
    rel = page.resolve().relative_to(REPO).as_posix()
    if is_generated(page) or "_build" in page.parts:
        return False
    return not rel.startswith(_excluded())


def unlinked(page: Path) -> list[str]:
    """Every inline code span on `page` that names a published symbol and isn't a link.

    Only *dotted* spellings count. A bare class name such as ``Config`` is a word in a
    sentence as often as it is a lookup, and linking all nine mentions on a page is worse
    reading than linking none; the style rule is to link on first use, which is a judgment
    a scanner does not get to make. ``Dataset.collect`` is never anything but a lookup.
    """
    return [s for _, _, s in _spans(page.read_text(encoding="utf-8"))]


def _spans(text: str) -> list[tuple[int, int, str]]:
    """Each linkable span in `text` as ``(start, end, spelling)``, in source order.

    Fenced code, existing roles and Markdown links are blanked rather than removed so the
    offsets still index the original text, which is what lets ``--fix`` rewrite in place.
    """
    masked = list(text)
    for pattern in (_FENCE, _ROLE_SPAN, _MYST_LINK):
        for m in pattern.finditer(text):
            masked[m.start() : m.end()] = " " * (m.end() - m.start())
    blanked = "".join(masked)
    out = []
    for m in _SPAN.finditer(blanked):
        spelling = m.group(1)
        dotted = "." in spelling.split("(")[0]
        if (dotted and resolve(spelling) is not None) or source_path(spelling) is not None:
            out.append((m.start(), m.end(), spelling))
    return out


_DOC_ROLE = re.compile(r"\{doc\}`([^`]*)`")
_CARD_LINK = re.compile(r"^\s*:link:\s*(\S+)\s*$", re.MULTILINE)


def _page_names() -> set[str]:
    """Every doc name Sphinx will know, posix and suffix-free, relative to ``docs/``."""
    return {
        p.relative_to(DOCS).with_suffix("").as_posix()
        for p in DOCS.rglob("*.md")
        if "_build" not in p.parts
    }


def _target_of(page: Path, ref: str) -> str:
    """Resolve a ``{doc}`` target the way Sphinx does: absolute from ``docs/``, else relative."""
    ref = ref.strip()
    base = ref.lstrip("/") if ref.startswith("/") else f"{page.parent.relative_to(DOCS)}/{ref}"
    parts: list[str] = []
    for piece in base.split("/"):
        if piece == "..":
            if parts:
                parts.pop()
        elif piece not in (".", ""):
            parts.append(piece)
    return "/".join(parts)


def dead_doc_refs(page: Path, names: set[str]) -> list[str]:
    """Every ``{doc}`` or card ``:link:`` on `page` pointing at a document that isn't there.

    Sphinx catches these under ``-W``, but only at the end of a full build, which on this
    site is twenty minutes. Moving a page and missing one of its relative references is the
    single most common way to break the build, and it costs two seconds to check here.
    """
    text = page.read_text(encoding="utf-8")
    refs = [m.group(1) for m in _DOC_ROLE.finditer(text)]
    refs += [m.group(1) for m in _CARD_LINK.finditer(text)]
    dead = []
    for ref in refs:
        target = ref.split("<")[-1].rstrip(">") if "<" in ref else ref
        target = target.strip()
        if not target or target.startswith(("http://", "https://", "#")):
            continue
        if _target_of(page, target) not in names:
            dead.append(target)
    return dead


def fix(page: Path) -> int:
    """Rewrite every linkable span on `page` as a cross-reference. Returns how many."""
    text = page.read_text(encoding="utf-8")
    spans = _spans(text)
    for start, end, spelling in reversed(spans):
        markup = source(spelling) if source_path(spelling) is not None else link(spelling)
        text = text[:start] + markup + text[end:]
    if spans:
        page.write_text(text, encoding="utf-8")
    return len(spans)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print every resolvable spelling")
    parser.add_argument("--unlinked", action="store_true", help="print pages with dead spans")
    parser.add_argument("--fix", action="store_true", help="rewrite them as cross-references")
    parser.add_argument("--check", action="store_true", help="exit 1 if any page has one")
    parser.add_argument("--max", type=int, default=25, help="how many pages to print")
    args = parser.parse_args(argv)

    table = index()
    if args.list:
        for spelling, target in sorted(table.items()):
            print(f"{spelling:<48} {target}")
        return 0

    pages = [p for p in sorted(DOCS.rglob("*.md")) if in_scope(p)]
    names = _page_names()
    broken = {
        p.relative_to(REPO).as_posix(): dead
        for p in sorted(DOCS.rglob("*.md"))
        if "_build" not in p.parts and (dead := dead_doc_refs(p, names))
    }
    if args.fix:
        total = 0
        for page in pages:
            if (n := fix(page)) > 0:
                total += n
                print(f"{n:>5}  {page.relative_to(REPO).as_posix()}")
        print(f"\nlinked {total:,} span(s)")
        return 0
    if broken and not args.list:
        for rel, dead in broken.items():
            print(f"  {rel}: {', '.join(dead)}", file=sys.stderr)
        print(
            f"{sum(len(v) for v in broken.values())} dead `{{doc}}` reference(s) on "
            f"{len(broken)} page(s). Sphinx would fail on these at the end of a full build.",
            file=sys.stderr,
        )
        if args.check:
            return 1

    if args.unlinked or args.check:
        rows = [(n, p.relative_to(REPO).as_posix()) for p in pages if (n := len(unlinked(p))) > 0]
        rows.sort(reverse=True)
        for n, rel in rows[: args.max]:
            print(f"{n:>5}  {rel}", file=sys.stderr if args.check else sys.stdout)
        total = sum(n for n, _ in rows)
        if args.check:
            if rows:
                print(
                    f"\n{total:,} code span(s) on {len(rows)} page(s) name a published API "
                    "symbol or a repo file and don't link to it. Run "
                    "`python tools/doc_api_links.py --fix`.",
                    file=sys.stderr,
                )
                return 1
            print("doc_api_links: every API name in the published pages is a cross-reference")
            return 0
        print(f"\n{total:,} unlinked spans across {len(rows)} pages")
        return 0
    print(f"{len(table):,} spellings resolve to {len(set(table.values())):,} targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
