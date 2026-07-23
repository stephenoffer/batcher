"""Guard the agent-facing context: a token budget, and the rules that fail silently.

Most of this codebase is written by agents, so ``CLAUDE.md`` and its sibling context files
are load-bearing infrastructure — and they have two failure modes that no other gate sees.

**Regrowth.** Every token in the always-loaded contract is paid by every agent and every
subagent on every session, before any work happens. That cost is invisible at edit time: a
paragraph added here is never obviously too expensive, and the file had grown to ~15,000
tokens (with its ``@import``s) one reasonable addition at a time. The budgets below make the
cost visible at the moment it is incurred.

**Silent loss.** The contract was deliberately restructured so layer detail loads only when
relevant, which means the always-loaded core is now the *only* place some rules are
guaranteed to be read. A specific set of them share a property: an agent that never sees
them writes a change that **passes every mechanical gate and is still wrong** — a phantom
import cycle, an unrun pre-commit hook, a sort bug an order-independent assertion cannot
see, a size-limit "fix" that breaks a layer, a weakened differential oracle. Losing one of
those in a routine edit would be undetectable until it caused a bug, so each is asserted
here by the phrase that carries it.

Editing the contract is expected. Silently dropping a guard, or drifting past the budget
without deciding to, is not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]

#: Rough chars-per-token. Deliberately crude — the budgets are order-of-magnitude
#: guardrails against regrowth, not an accounting system.
_CHARS_PER_TOKEN = 4

#: Files whose cost every agent pays on every session, and their ceilings. `CLAUDE.md` is
#: always loaded; the two nested files load only when editing under their directory, so the
#: real per-task cost is the root plus at most one of them. Headroom is intentional — these
#: are meant to catch drift, not to block a justified paragraph.
_BUDGETS = {
    "CLAUDE.md": 3_000,
    "python/batcher/CLAUDE.md": 2_200,
    "crates/CLAUDE.md": 2_200,
}

#: The always-loaded total a single-language task pays. Rebuilt from `_BUDGETS` so the two
#: cannot disagree.
_TASK_BUDGET = _BUDGETS["CLAUDE.md"] + max(
    _BUDGETS["python/batcher/CLAUDE.md"], _BUDGETS["crates/CLAUDE.md"]
)

#: Rules whose absence fails *silently*. Key = what it protects, value = a phrase that must
#: survive in `CLAUDE.md`. Matched case-insensitively on a whitespace-normalized file, so
#: rewording around the phrase is fine; removing the rule is not.
#:
#: Each phrase must be **unique to its rule**. A phrase that also appears in nearby prose
#: makes the assertion vacuous — deleting the rule still leaves the phrase behind, and the
#: test passes while the guard is gone. `test_guard_phrases_are_specific` enforces this.
_SILENT_FAILURE_GUARDS = {
    "never import _native directly (forges a phantom layer cycle)": "import batcher._native",
    "install the pre-commit hook (bootstraps every other gate)": "just install-hooks",
    "a green gate is not a green light": "green gate is not a green light",
    "a sort bug is invisible to an order-independent comparison": "never assert a sort",
    "size limits are subordinate to the invariants": "subordinate",
    "never weaken a differential test to make a change pass": "weaken or delete",
    "keep the subsystem verbs in their lanes": "Core measures, Kyber decides",
    "a non-mergeable stateful operator silently caps at one machine": "silently caps the operator",
    "the JSON IR is a two-sided change in one commit": "same commit",
    "the JIT must match the interpreter or fall back, never diverge": "fall back",
    "do not restore a retired competitive claim": "retires",
}


def _tokens(path: Path) -> int:
    """Estimate the token cost of a context file."""
    return len(path.read_text(encoding="utf-8")) // _CHARS_PER_TOKEN


@pytest.mark.parametrize(("rel", "budget"), sorted(_BUDGETS.items()))
def test_context_file_within_budget(rel: str, budget: int) -> None:
    """Each agent-context file stays under its token ceiling."""
    actual = _tokens(_REPO / rel)
    assert actual <= budget, (
        f"{rel} is ~{actual} tokens, over its ~{budget} budget.\n"
        "Every agent pays this on every session. Either tighten the file, move layer-specific "
        "detail into .claude/rules/ (loaded on demand), or raise the budget deliberately here."
    )


def test_single_language_task_budget() -> None:
    """The root contract plus one nested file stays within the per-task budget."""
    root = _tokens(_REPO / "CLAUDE.md")
    worst_nested = max(_tokens(_REPO / rel) for rel in _BUDGETS if rel != "CLAUDE.md")
    assert root + worst_nested <= _TASK_BUDGET, (
        f"A single-language task loads ~{root + worst_nested} tokens of contract, over the "
        f"~{_TASK_BUDGET} budget."
    )


@pytest.mark.parametrize(("protects", "phrase"), sorted(_SILENT_FAILURE_GUARDS.items()))
def test_silent_failure_guard_survives(protects: str, phrase: str) -> None:
    """Every rule that fails silently is still stated in the always-loaded contract."""
    text = " ".join((_REPO / "CLAUDE.md").read_text(encoding="utf-8").split())
    assert phrase.lower() in text.lower(), (
        f"CLAUDE.md no longer states the rule protecting: {protects}\n"
        f"(looked for {phrase!r})\n"
        "This rule fails SILENTLY — an agent that never reads it writes a change that passes "
        "every gate and is still wrong. It must stay in the always-loaded contract, not move "
        "to a file that loads only sometimes. If you reworded it, update the phrase here."
    )


@pytest.mark.parametrize(("protects", "phrase"), sorted(_SILENT_FAILURE_GUARDS.items()))
def test_guard_phrases_are_specific(protects: str, phrase: str) -> None:
    """Each guard phrase appears exactly once, so its assertion cannot be vacuous.

    A phrase occurring twice makes the guard untestable in practice: deleting the rule leaves
    the other occurrence behind and the assertion still passes. This was a real defect in the
    first version of this file — the sort guard matched ``order-independent``, which also
    appeared in the sentence beside it, so removing the rule did not fail the test.
    """
    text = " ".join((_REPO / "CLAUDE.md").read_text(encoding="utf-8").split()).lower()
    occurrences = text.count(phrase.lower())
    assert occurrences == 1, (
        f"The guard phrase {phrase!r} (protecting: {protects}) appears {occurrences} times in "
        "CLAUDE.md. It must appear exactly once, or deleting the rule would still leave the "
        "phrase behind and this guard would silently stop guarding. Pick a phrase unique to "
        "the rule."
    )


def test_nested_contracts_exist() -> None:
    """The layer contracts exist, since the root file delegates layer detail to them."""
    for rel in _BUDGETS:
        assert (_REPO / rel).is_file(), f"{rel} is missing — the root contract points at it."


def test_root_contract_does_not_reimport_the_rule_files() -> None:
    """The rule files stay on-demand: `@import`ing them restores the always-loaded cost.

    They are the deep reference, pointed to by topic. Re-adding `@.claude/rules/...` would
    silently put ~11,500 tokens back into every session's context.
    """
    text = (_REPO / "CLAUDE.md").read_text(encoding="utf-8")
    offenders = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("@.claude/rules/")]
    assert not offenders, (
        "CLAUDE.md @imports rule files again, making them always-loaded:\n"
        + "\n".join(f"  {o}" for o in offenders)
        + "\nReference them by path instead so they load on demand."
    )
