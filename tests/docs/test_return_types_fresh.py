"""Anti-drift gate: `docs/api/return-types.md` still matches the live return annotations.

The page is rendered from each public signature's return annotation by
`tools/gen_return_types_doc.py`. A signature whose return type changed without
regenerating would publish a type the engine no longer returns, and nothing executes a
table cell. If this fails, run `just return-types-doc` and commit the result.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools import gen_return_types_doc as gen  # noqa: E402


def test_return_types_page_is_current() -> None:
    expected = gen.render()
    committed = gen.TARGET.read_text()
    diff = "".join(
        difflib.unified_diff(
            committed.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            "committed",
            "generated",
            n=1,
        )
    )
    assert committed == expected, f"run `just return-types-doc`:\n{diff[:4000]}"


def test_every_terminal_is_listed() -> None:
    """The page names the terminals a reader looks up, so the freshness gate is not vacuous."""
    page = gen.TARGET.read_text()
    for name in ("`collect`", "`to_arrow`", "`iter_batches`", "`parquet`", "`count`", "`sum`"):
        assert name in page, name


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        ("Dataset", "Dataset"),
        ("pa.Table", "pa.Table"),
        ("int | None", "scalar"),
        ("Any", "scalar"),
        ("None", "nothing"),
        ("Iterator[tuple[Any, ...] | dict[str, Any]]", "iterator"),
        ("list[Dataset]", "collection"),
        ("AggExpr | Expr", "AggExpr or Expr"),
        ("MathExpr", "Expr"),
        ("WriteManifest | StreamingQuery", "WriteManifest or StreamingQuery"),
        ("_StrNamespace", "accessor"),
    ],
)
def test_category(annotation: str, expected: str) -> None:
    assert gen.category(annotation) == expected
