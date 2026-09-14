"""Structural anti-drift gate: every docs page is reachable, and none is oversized.

Sphinx's ``-W`` build already fails on an orphan page (``toc.not_included``), but that
signal only arrives at the end of a full HTML build. This test catches the same
problem in seconds so a page added without a toctree entry fails fast — the recurring
failure mode when a new design note or guide lands under ``docs/`` but nobody wires it
into a ``{toctree}``.

The contract: every Markdown file under ``docs/`` is either
- the root document (``index.md``),
- listed as an entry in some ``{toctree}`` block, or
- named in ``exclude_patterns`` in ``conf.py`` (a deliberate non-page: a build
  helper, a contributor RFC, a PDF-only paper).

Anything else is an orphan.

The second contract is size. A page that grows past ``_MAX_PAGE_LINES`` has stopped
being one topic: it needs a bullet preview to navigate, it buries its own sections, and
it is the state every one of these pages reached before being split. The limit is
mechanical for the same reason the Python and Rust size limits are — a reviewer's
patience is not a gate. Genuine exceptions go in ``OVERSIZED_ALLOW`` with a reason.

The third contract is *breadth*, and it is the one a page-by-page review never catches.
A directory holding thirty files is unnavigable however good each file is: the sidebar
runs past a screen, the reader scans instead of choosing, and nothing in a per-page gate
notices. So a directory carries at most ``_MAX_PAGES_PER_DIR`` Markdown files and
``_MAX_DIRS_PER_DIR`` subdirectories, and the tree goes no deeper than
``_MAX_DEPTH`` levels below ``docs/``. These mirror the ``≤12 files per directory`` and
``≤5 levels`` limits ``tools/lint_structure.py`` already enforces on ``python/batcher/``
and ``crates/*/src``, for the same reason and with the same remedy: at the ceiling, add a
subpackage grouped by responsibility rather than flattening names to dodge it.

The page count includes files ``conf.py`` excludes from the build. An excluded working
record is still a file a contributor has to scan past, and the directory that first hit
this limit — ``architecture/internals`` — was almost entirely excluded pages. Counting
only what Sphinx publishes would have declared it clean at 29 files.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_DOCS = Path(__file__).resolve().parents[2] / "docs"
_TOCTREE = re.compile(r"```\{toctree\}(.*?)```", re.DOTALL)
_LITERALINCLUDE = re.compile(r"(?m)^\s*```\{literalinclude\}\s+(\S+)\s*$")

# One topic per page, and a topic fits in this many lines. Pages that outgrew it were
# split into a section with an index rather than allowlisted.
_MAX_PAGE_LINES = 500

# Published pages allowed past the limit, each with the reason. Drain toward empty.
OVERSIZED_ALLOW: dict[str, str] = {}

# Breadth. A dozen entries is about what a reader chooses from rather than scans; past
# that the sidebar stops being a menu. `_MAX_DIRS_PER_DIR` is lower than the page limit
# on purpose: a subdirectory costs a reader a click and an index page to maintain, so
# a level should reach for one later than it reaches for another page.
_MAX_PAGES_PER_DIR = 12
_MAX_DIRS_PER_DIR = 10

# Levels below `docs/` itself, counting the file. `docs/user-guide/transform/columns/
# udfs.md` is 4. Five is the ceiling, matching `lint_structure.py`'s tree depth.
_MAX_DEPTH = 5

# Directories allowed past the breadth limits, each with the reason. Same rule as
# OVERSIZED_ALLOW: it may shrink, and an entry is an argument, not a parking space.
BREADTH_ALLOW: dict[str, str] = {}

#: Never content: build output, static assets, Jinja templates.
_NON_CONTENT = {"_build", "_static", "_templates"}


def _all_pages() -> set[str]:
    """Every doc name (posix, no suffix, relative to docs/) except the build tree."""
    return {
        p.relative_to(_DOCS).with_suffix("").as_posix()
        for p in _DOCS.rglob("*.md")
        if "_build" not in p.parts and "_static" not in p.parts
    }


def _toctree_entries() -> set[str]:
    """Every doc a ``{toctree}`` references, resolved relative to its own page."""
    entries: set[str] = set()
    for page in _DOCS.rglob("*.md"):
        if "_build" in page.parts:
            continue
        base = page.parent
        for block in _TOCTREE.findall(page.read_text(encoding="utf-8")):
            for line in block.splitlines():
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                # A caption entry may be `Title <path>`; keep the path.
                target = line.split("<")[-1].rstrip(">") if "<" in line else line
                entries.add((base / target).resolve().relative_to(_DOCS).as_posix())
    return entries


def _excluded_pages() -> set[str]:
    """Doc names that ``conf.py`` deliberately excludes from the build."""
    conf = (_DOCS / "conf.py").read_text(encoding="utf-8")
    tree = ast.parse(conf)
    patterns: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "exclude_patterns" for t in node.targets)
            and isinstance(node.value, ast.List)
        ):
            patterns = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    excluded: set[str] = set()
    for pat in patterns:
        if pat.endswith(".md"):
            excluded.add(pat[:-3])
            continue
        # A directory pattern (`architecture/internals/parity/*`) excludes a whole family
        # of working records at once, so adding one does not mean remembering to exclude
        # it. Resolve it against the tree rather than matching the literal string: this
        # helper answers "is this page published?", and a glob that matched nothing would
        # otherwise report a real orphan as deliberate.
        for page in _DOCS.glob(pat if pat.endswith(".md") else f"{pat}.md"):
            excluded.add(page.relative_to(_DOCS).with_suffix("").as_posix())
    return excluded


def test_no_orphan_pages():
    pages = _all_pages()
    reachable = _toctree_entries() | {"index"} | _excluded_pages()
    orphans = sorted(pages - reachable)
    assert not orphans, (
        f"{len(orphans)} docs page(s) are in no toctree and not excluded: {orphans}\n"
        "Add each to a `{toctree}` on its section index, or to `exclude_patterns` in "
        "docs/conf.py if it is a contributor note rather than a published page."
    )


def test_toctree_entries_all_exist():
    pages = _all_pages()
    # autosummary emits generated stubs under api/generated/ at build time; a toctree
    # may legitimately point at pages created by other directives, so only flag entries
    # that look like hand-written docs (no `generated/` segment) yet have no source file.
    missing = sorted(e for e in _toctree_entries() if "generated/" not in e and e not in pages)
    assert not missing, (
        f"{len(missing)} toctree entr(y/ies) point at a nonexistent page: {missing}\n"
        "Fix the path or remove the stale entry."
    )


def test_no_oversized_pages():
    excluded = _excluded_pages()
    oversized: list[tuple[str, int]] = []
    for page in sorted(_DOCS.rglob("*.md")):
        if "_build" in page.parts or "_static" in page.parts:
            continue
        name = page.relative_to(_DOCS).with_suffix("").as_posix()
        if name in excluded or name in OVERSIZED_ALLOW:
            continue
        lines = len(page.read_text(encoding="utf-8").splitlines())
        if lines > _MAX_PAGE_LINES:
            oversized.append((name, lines))

    assert not oversized, (
        f"{len(oversized)} docs page(s) over {_MAX_PAGE_LINES} lines: "
        + ", ".join(f"{n} ({c})" for n, c in oversized)
        + "\nSplit each into a section with its own index and toctree, or add it to "
        "OVERSIZED_ALLOW with a one-line reason."
    )

    stale = sorted(
        name
        for name in OVERSIZED_ALLOW
        if (_DOCS / f"{name}.md").exists()
        and len((_DOCS / f"{name}.md").read_text(encoding="utf-8").splitlines()) <= _MAX_PAGE_LINES
    )
    assert not stale, f"OVERSIZED_ALLOW lists pages that now fit (remove them): {stale}"


def _content_dirs() -> list[Path]:
    """Every directory under `docs/` that holds documentation, including `docs/` itself."""
    dirs = [_DOCS]
    dirs.extend(
        d
        for d in sorted(_DOCS.rglob("*"))
        if d.is_dir() and not (_NON_CONTENT & set(d.relative_to(_DOCS).parts))
    )
    return dirs


def test_no_overfull_directories() -> None:
    """No level holds more than a dozen pages. Past that a sidebar is scanned, not read."""
    overfull: list[tuple[str, int]] = []
    for d in _content_dirs():
        rel = d.relative_to(_DOCS).as_posix() or "."
        if rel in BREADTH_ALLOW:
            continue
        pages = list(d.glob("*.md"))
        if len(pages) > _MAX_PAGES_PER_DIR:
            overfull.append((rel, len(pages)))

    assert not overfull, (
        f"{len(overfull)} docs director(y/ies) over {_MAX_PAGES_PER_DIR} pages: "
        + ", ".join(f"{n} ({c})" for n, c in overfull)
        + "\nGroup the pages into subdirectories by responsibility, each with its own "
        "index.md and toctree, or merge the short ones. Do not flatten names to dodge it. "
        "A genuine exception goes in BREADTH_ALLOW with a one-line reason."
    )


def test_no_overbroad_directories() -> None:
    """No level holds more than ten subdirectories."""
    overbroad: list[tuple[str, int]] = []
    for d in _content_dirs():
        rel = d.relative_to(_DOCS).as_posix() or "."
        if rel in BREADTH_ALLOW:
            continue
        subdirs = [
            x
            for x in d.iterdir()
            if x.is_dir() and not (_NON_CONTENT & set(x.relative_to(_DOCS).parts))
        ]
        if len(subdirs) > _MAX_DIRS_PER_DIR:
            overbroad.append((rel, len(subdirs)))

    assert not overbroad, (
        f"{len(overbroad)} docs director(y/ies) over {_MAX_DIRS_PER_DIR} subdirectories: "
        + ", ".join(f"{n} ({c})" for n, c in overbroad)
        + "\nFold two sections whose readers are the same reader, or move one under the "
        "section it serves. A genuine exception goes in BREADTH_ALLOW with a reason."
    )


def test_tree_is_not_too_deep() -> None:
    """No page sits more than five levels below `docs/`."""
    deep = sorted(
        (p.relative_to(_DOCS).as_posix(), len(p.relative_to(_DOCS).parts))
        for p in _DOCS.rglob("*.md")
        if not (_NON_CONTENT & set(p.relative_to(_DOCS).parts))
        and len(p.relative_to(_DOCS).parts) > _MAX_DEPTH
    )
    assert not deep, (
        f"{len(deep)} docs page(s) deeper than {_MAX_DEPTH} levels: "
        + ", ".join(f"{n} ({c})" for n, c in deep)
        + "\nA reader who needs five clicks to reach a page will not reach it. Flatten "
        "the branch or move the content up."
    )


def test_every_content_directory_has_an_index() -> None:
    """A published directory without an `index.md` is a section with nothing introducing it.

    A directory whose pages are *all* excluded is not a published section — it is a shelf of
    contributor working records — so it is held to the breadth limits but not to this one.
    Requiring an index there would mean writing a page that then has to be excluded too.
    """
    excluded = _excluded_pages()
    missing: list[str] = []
    for d in _content_dirs():
        pages = list(d.glob("*.md"))
        if not pages or (d / "index.md").exists():
            continue
        published = [
            p for p in pages if p.relative_to(_DOCS).with_suffix("").as_posix() not in excluded
        ]
        if published:
            missing.append(d.relative_to(_DOCS).as_posix())
    missing.sort()
    assert not missing, (
        f"{len(missing)} docs director(y/ies) have pages but no index.md: {missing}\n"
        "Every directory carries an index that introduces its children and holds their "
        "toctree. If a directory does not warrant one, its pages should not be nested."
    )


def test_breadth_allowlist_is_not_stale() -> None:
    """An allowlisted directory that now fits is an entry to delete."""
    stale: list[str] = []
    for rel in BREADTH_ALLOW:
        d = _DOCS if rel == "." else _DOCS / rel
        if not d.exists():
            stale.append(f"{rel} (gone)")
            continue
        pages = len(list(d.glob("*.md")))
        subdirs = len([x for x in d.iterdir() if x.is_dir() and x.name not in _NON_CONTENT])
        if pages <= _MAX_PAGES_PER_DIR and subdirs <= _MAX_DIRS_PER_DIR:
            stale.append(f"{rel} ({pages} pages, {subdirs} dirs)")
    assert not stale, f"BREADTH_ALLOW lists directories that now fit (remove them): {stale}"


def test_literalinclude_paths_resolve() -> None:
    """Every `{literalinclude}` points at a file that exists.

    105 cookbook pages embed their script this way rather than duplicating it, which is
    what keeps the page and the executed example in step. The path is relative to the page
    holding it, so a page one directory deeper needs one more `../` — and getting that
    wrong fails only at the end of a full `-W` Sphinx build, twenty minutes later, with a
    message about a missing include rather than about the move that caused it. Six were
    wrong at once (`../../../` where the file needed `../../../../`) before this existed.
    """
    broken: list[str] = []
    for page in sorted(_DOCS.rglob("*.md")):
        if _NON_CONTENT & set(page.relative_to(_DOCS).parts):
            continue
        for target in _LITERALINCLUDE.findall(page.read_text(encoding="utf-8")):
            if not (page.parent / target).resolve().exists():
                broken.append(f"{page.relative_to(_DOCS).as_posix()} -> {target}")

    assert not broken, (
        f"{len(broken)} `literalinclude` path(s) point at a file that does not exist:\n  "
        + "\n  ".join(broken)
        + "\nThe path is relative to the page holding it, so check the `../` depth first."
    )
