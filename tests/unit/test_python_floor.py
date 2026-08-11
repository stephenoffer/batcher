"""The declared `requires-python` must cover the language the control plane actually uses.

This is a hole CI cannot see. The gate runs on one interpreter, so nothing ever executes the
oldest Python the package claims to support, and a single newer-than-the-floor name is enough
to make `import batcher` fail there. That is exactly what had happened: the floor said 3.10,
the release workflow built the abi3 wheel on 3.10 so pip installed on 3.10 without complaint,
and `plan.expr_ir.fn_names` — reached by every expression import — used `enum.StrEnum`, which
is 3.11. Every 3.10 install was broken from the first import.

The check is a grep, deliberately. It cannot know about a *semantic* change between versions,
only about names that did not exist, and that is the class of failure that shipped. Adding a
newer name is fine — add it to `_INTRODUCED_IN` and raise the floor in the same commit.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE = _ROOT / "python" / "batcher"

#: Standard-library names against the version that introduced them. Matched as source text
#: rather than by import, because the point is what a *older* interpreter would refuse to
#: parse or resolve — which is a property of the file, not of the running process.
_INTRODUCED_IN: dict[str, tuple[int, int]] = {
    r"\bStrEnum\b": (3, 11),
    r"\bReprEnum\b": (3, 11),
    r"\bdatetime\.UTC\b": (3, 11),
    r"\btomllib\b": (3, 11),
    r"\.add_note\(": (3, 11),
    # Anchored to the start of a statement, because `*except*` is also how a comment
    # emphasises the word — and three comments in this package do exactly that. An unanchored
    # pattern reported all three as `except*` syntax nobody had written.
    r"(?m)^[ \t]*except[ \t]*\*[ \t]": (3, 11),
    r"\basyncio\.TaskGroup\b": (3, 11),
    r"\bExceptionGroup\b": (3, 11),
    r"\bitertools\.batched\b": (3, 12),
    r"\btyping\.override\b": (3, 12),
    r"\bhashlib\.file_digest\b": (3, 11),
}


def _declared_floor() -> tuple[int, int]:
    spec = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]["requires-python"]
    match = re.search(r">=\s*(\d+)\.(\d+)", spec)
    assert match, f"requires-python {spec!r} has no >= floor to check"
    return int(match.group(1)), int(match.group(2))


def _sources() -> list[Path]:
    return sorted(_PACKAGE.rglob("*.py"))


def _required_floor() -> tuple[tuple[int, int], list[str]]:
    """The highest version any source file needs, with the evidence that set it."""
    needed = (3, 0)
    evidence: list[str] = []
    for path in _sources():
        text = path.read_text(encoding="utf-8")
        for pattern, version in _INTRODUCED_IN.items():
            if re.search(pattern, text):
                if version > needed:
                    needed, evidence = version, []
                if version == needed:
                    evidence.append(f"{path.relative_to(_ROOT)}: {pattern}")
    return needed, evidence


def test_the_declared_floor_covers_every_feature_the_package_uses():
    required, evidence = _required_floor()
    declared = _declared_floor()
    assert declared >= required, (
        f"requires-python declares {declared[0]}.{declared[1]} but the package needs "
        f"{required[0]}.{required[1]}:\n  " + "\n  ".join(sorted(evidence)[:10])
    )


def test_the_release_workflow_builds_on_the_declared_floor():
    """The abi3 wheel's build interpreter *is* the floor it advertises to pip.

    Building on a Python older than `requires-python` makes the wheel installable where the
    code cannot run, which is the half of the bug that let pip say yes.
    """
    workflow = (_ROOT / ".github" / "workflows" / "release.yml").read_text()
    declared = _declared_floor()
    pinned = re.findall(r'python-version:\s*"(\d+\.\d+)"', workflow)
    built = {tuple(map(int, v.split("."))) for v in pinned}
    assert built, "no python-version pinned in the release workflow"
    assert all(v >= declared for v in built), (
        f"release builds on {sorted(built)} but requires-python declares {declared}"
    )


def test_the_check_would_catch_a_regression():
    """A guard that cannot fail is not a guard: prove the matcher sees a known marker."""
    assert re.search(r"\bStrEnum\b", "from enum import StrEnum")
    assert (_PACKAGE / "plan" / "expr_ir" / "fn_names.py").exists()
