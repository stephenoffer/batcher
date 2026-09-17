#!/usr/bin/env python3
"""Fail when a built documentation page renders longer than a reader can hold.

``tests/docs/test_docs_structure.py`` caps a page's *source* at 500 lines, and that cap
cannot see the pages most likely to break it. An ``.. autoclass::`` with ``:members:`` is
three source lines and renders the whole class: ``docs/api/complete/expressions.md`` was
139 lines of Markdown and 90,000 rendered words, one page holding 763 signatures. So this
check reads the HTML Sphinx wrote and counts the words in each page's article body.

The budget is words rather than bytes, because markup and highlighted code inflate bytes
without adding reading. ``_modules/`` (the viewcode source listings), the general index and
the search page are not pages a reader scrolls, so they are skipped.

Usage::

    python tools/check_page_size.py docs/_build/html
    python tools/check_page_size.py docs/_build/html --max-words 12000
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

#: The rendered budget. The longest hand-written page renders about 5,600 words; an API
#: page split into per-member tables renders a few thousand. Past this, split the page.
MAX_WORDS = 10_000

_SKIP_PREFIXES = ("_modules/", "_static/", "_sources/")
_SKIP_PAGES = frozenset({"genindex.html", "search.html", "py-modindex.html"})
_ARTICLE = re.compile(r"<article[^>]*>(.*?)</article>", re.DOTALL)
_TAG = re.compile(r"<[^>]+>")


def rendered_words(page: Path) -> int:
    """Count the words in a built page's article body, or its whole body if it has none."""
    text = page.read_text(encoding="utf-8", errors="ignore")
    match = _ARTICLE.search(text)
    body = match.group(1) if match else text
    return len(html.unescape(_TAG.sub(" ", body)).split())


def oversized(root: Path, max_words: int) -> list[tuple[int, str]]:
    """Every page under `root` over `max_words`, largest first."""
    found = []
    for page in root.rglob("*.html"):
        rel = page.relative_to(root).as_posix()
        if rel.startswith(_SKIP_PREFIXES) or rel in _SKIP_PAGES:
            continue
        words = rendered_words(page)
        if words > max_words:
            found.append((words, rel))
    return sorted(found, reverse=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("html_dir", type=Path)
    parser.add_argument("--max-words", type=int, default=MAX_WORDS)
    args = parser.parse_args(argv)
    if not (args.html_dir / "index.html").exists():
        print(f"error: {args.html_dir} holds no built site (no index.html)", file=sys.stderr)
        return 2
    found = oversized(args.html_dir, args.max_words)
    for words, rel in found:
        print(f"{words:>8,} words  {rel}", file=sys.stderr)
    if found:
        print(
            f"{len(found)} page(s) render over {args.max_words:,} words. Split them: turn "
            "`autoclass :members:` into grouped `autosummary` tables with `:toctree:`.",
            file=sys.stderr,
        )
        return 1
    print(f"check_page_size: every page renders at most {args.max_words:,} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
