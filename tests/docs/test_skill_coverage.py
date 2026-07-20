"""Mechanical anti-drift gate: every agent skill is catalogued in the docs.

The skills under ``.claude/skills/`` are how a coding agent learns to drive this
engine, and a skill nobody can find is a skill nobody uses. ``docs/agents/index.md``
is the catalog, and this module keeps it honest in both directions: a skill added to
the tree must be listed, and a skill listed must exist.

It also enforces the frontmatter contract every skill depends on. An agent selects a
skill by matching a task against its ``description``, so a file missing ``name`` or
``description`` is invisible to the mechanism that would have loaded it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_SKILLS = _ROOT / ".claude" / "skills"
_CATALOG = _ROOT / "docs" / "agents" / "index.md"

_NAME = re.compile(r"^name:\s*(\S+)\s*$", re.MULTILINE)
_DESCRIPTION = re.compile(r"^description:\s*(.+)$", re.MULTILINE)


def _skill_files() -> list[Path]:
    return sorted(_SKILLS.glob("*/SKILL.md"))


def _catalog_text() -> str:
    return _CATALOG.read_text()


def test_skills_exist() -> None:
    """The skill tree is non-empty — a silent glob failure would pass everything else."""
    assert _skill_files(), f"no SKILL.md files found under {_SKILLS}"


@pytest.mark.parametrize("path", _skill_files(), ids=lambda p: p.parent.name)
def test_skill_has_frontmatter(path: Path) -> None:
    """Every skill declares a `name` and a `description` an agent can match against."""
    text = path.read_text()
    name = _NAME.search(text)
    description = _DESCRIPTION.search(text)
    assert name, f"{path} is missing a `name:` frontmatter field"
    assert description, f"{path} is missing a `description:` frontmatter field"
    assert name.group(1) == path.parent.name, (
        f"{path} declares name `{name.group(1)}` but lives in `{path.parent.name}/`; "
        "the directory name is how the skill is invoked, so the two must agree"
    )
    # The description is the whole selection signal: it must say when to invoke.
    assert "Invoke" in description.group(1), (
        f"{path} description must say when to invoke the skill (an `Invoke when ...` "
        "clause), not only what it covers"
    )


@pytest.mark.parametrize("path", _skill_files(), ids=lambda p: p.parent.name)
def test_skill_is_catalogued(path: Path) -> None:
    """Every skill is listed in the docs catalog, so it is discoverable."""
    assert path.parent.name in _catalog_text(), (
        f"skill `{path.parent.name}` is not listed in docs/agents/index.md — add it to "
        "the relevant table so readers and agents can find it"
    )


def test_catalog_lists_no_missing_skills() -> None:
    """Every skill named in the catalog exists on disk (catches renames and typos)."""
    known = {p.parent.name for p in _skill_files()}
    # Skill names in the catalog are written as inline code: `some-skill-name`.
    cited = {
        token
        for token in re.findall(r"`([a-z][a-z0-9-]+)`", _catalog_text())
        if token.startswith(("add-", "migrate-", "run-", "write-", "build-", "debug-"))
        or token in {"optimize-a-slow-query", "read-and-write-data"}
        or token.startswith(("apply-", "manage-", "validate-"))
    }
    missing = sorted(cited - known)
    assert not missing, (
        f"docs/agents/index.md cites skills that do not exist: {missing}. "
        "Fix the name or add the skill."
    )
