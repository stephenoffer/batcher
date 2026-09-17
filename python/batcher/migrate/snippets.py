"""Apply the canonical-name rewrite to code that lives inside text: doctests and Markdown blocks.

Two kinds of Batcher code are executed without being Python source a parser sees directly. The
`.. doctest::` examples in every public docstring run under `just docs`, and every fenced
`python` block under `docs/` runs under `tests/docs/test_doc_examples.py`. Both would keep
calling a removed spelling after the rewrite touched only the surrounding module, and fail
the build the moment the spelling is deleted.

Receivers carry across examples the way a reader's session does: every `>>>` example in one
docstring is one program, and every block of one Markdown page is one program. Each is rewritten
as a whole and split back. A rewrite that would change a snippet's line count (none of the rule
kinds does) is refused and reported rather than applied, because the split back would misalign.
"""

from __future__ import annotations

import ast
import re

from batcher._internal.migration import KwargRename, Rename
from batcher.migrate.canonical import Edit, RenameReport, canonicalize

__all__ = ["rewrite_markdown", "rewrite_python"]

_DOCTEST = re.compile(r"^(?P<indent>[ \t]*)(?P<prompt>>>> |\.\.\. |>>>$|\.\.\.$)(?P<code>.*)$")
_FENCE = re.compile(r"^```python[^\n]*\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
_SEPARATOR = "# batcher-migrate: snippet boundary"


def _rewrite_program(
    pieces: list[str],
    tables: tuple[
        dict[str, dict[str, Rename]], dict[str, dict[str, str]], dict[str, dict[str, KwargRename]]
    ],
    report: RenameReport,
    line_of: list[int],
) -> list[str] | None:
    """Rewrite snippets as one program; `None` when any piece's line count would change."""
    joined = f"\n{_SEPARATOR}\n".join(pieces)
    try:
        rewritten, sites = canonicalize(joined, *tables)
    except Exception:  # a snippet that is not valid Python on its own is left alone
        return None
    out = rewritten.split(f"\n{_SEPARATOR}\n")
    if len(out) != len(pieces) or any(
        a.count("\n") != b.count("\n") for a, b in zip(out, pieces, strict=True)
    ):
        return None
    offsets = []
    line = 1
    for piece in pieces:
        offsets.append(line)
        line += piece.count("\n") + 2
    for kind, edits in (("renamed", sites.renamed), ("unresolved", sites.unresolved)):
        for edit in edits:
            index = max(i for i, start in enumerate(offsets) if start <= edit.line)
            mapped = Edit(
                line_of[index] + edit.line - offsets[index], edit.old, edit.new, edit.receiver
            )
            getattr(report, kind).append(mapped)
    return out


def rewrite_python(
    source: str,
    *tables: object,
    assume_accessors: bool = False,
    imported: dict[str, str] | None = None,
) -> tuple[str, RenameReport]:
    """Rewrite a Python module and the doctest examples in its docstrings.

    Args:
        source: The module source.
        *tables: `renames`, `returns` and `kwargs`, as `canonicalize` takes them.
        assume_accessors: Passed to `canonicalize` for the module body.
        imported: Passed to `canonicalize` for the module body.

    Returns:
        The rewritten source and one report covering code and doctests.
    """
    code, report = canonicalize(  # type: ignore[arg-type]
        source, *tables, assume_accessors=assume_accessors, imported=imported
    )
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, report
    lines = code.split("\n")
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Constant)
            or not isinstance(node.value, str)
            or ">>>" not in node.value
        ):
            continue
        start, end = node.lineno - 1, (node.end_lineno or node.lineno) - 1
        example_rows = [i for i in range(start, end + 1) if _DOCTEST.match(lines[i])]
        if not example_rows:
            continue
        codes = [_DOCTEST.match(lines[i]).group("code") for i in example_rows]  # type: ignore[union-attr]
        rewritten = _rewrite_program(["\n".join(codes)], tables, report, [example_rows[0] + 1])  # type: ignore[arg-type]
        if rewritten is None:
            continue
        for i, new in zip(example_rows, rewritten[0].split("\n"), strict=True):
            m = _DOCTEST.match(lines[i])
            assert m is not None
            prompt = m.group("prompt")
            lines[i] = (
                f"{m.group('indent')}{prompt}{new}" if new or prompt.endswith(" ") else lines[i]
            )
    return "\n".join(lines), report


def rewrite_markdown(
    text: str,
    *tables: object,
    assume_accessors: bool = False,
    imported: dict[str, str] | None = None,
) -> tuple[str, RenameReport]:
    """Rewrite every fenced `python` block of one Markdown page as one program.

    Args:
        text: The page.
        *tables: `renames`, `returns` and `kwargs`, as `canonicalize` takes them.
        assume_accessors: Unused for Markdown, whose examples import Batcher explicitly.
        imported: Unused for Markdown.

    Returns:
        The rewritten page, and a report whose line numbers are page lines.
    """
    del assume_accessors, imported
    report = RenameReport()
    matches = list(_FENCE.finditer(text))
    if not matches:
        return text, report
    bodies = [m.group("body").rstrip("\n") for m in matches]
    first_lines = [text.count("\n", 0, m.start("body")) + 1 for m in matches]
    rewritten = _rewrite_program(bodies, tables, report, first_lines)  # type: ignore[arg-type]
    if rewritten is None:
        return text, report
    out = []
    cursor = 0
    for match, body, new in zip(matches, bodies, rewritten, strict=True):
        out.append(text[cursor : match.start("body")])
        out.append(new + match.group("body")[len(body) :])
        cursor = match.end("body")
    out.append(text[cursor:])
    return "".join(out), report
