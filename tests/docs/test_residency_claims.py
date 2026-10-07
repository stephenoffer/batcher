"""Anti-drift gate: the docs may not claim the scheduler enforces data residency until it does.

The residency catalog is real and unit-tested, but the scheduler never hands it a dataset:
`scheduling.py` calls `plan_collective` without `datasets=`, so the one filter that reads the
catalog never runs. The docs once said a strict catalog "raises" and that "a rule applies to
every stage", which is a compliance claim a reader would act on. This module ties the wording
to the call site: while the call passes no datasets, the retired phrases may not come back,
and the pages must say plainly that placement does not consult the catalog. Wiring the
datasets through is the one change that should make this test need editing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_SCHEDULING = _ROOT / "python/batcher/dist/executors/ray_runtime/scheduling.py"
_PAGES = (
    _ROOT / "docs/api/operations/governance.md",
    _ROOT / "docs/user-guide/operate/running/gpu-fleets.md",
    _ROOT / "docs/architecture/deep-dives/distribution/gpu-fabric.md",
)
# Each phrase is one the pages used while the claim was false.
_RETIRED = (
    "so a rule applies to every stage",
    "the scheduler consults it",
    "catalog reaches the scheduler",
    "and `strict` raises.",
    "a node excluded by a data-residency rule or a power-zone budget is skipped before "
    "placement rather than after.",
)


def _scheduler_passes_datasets() -> bool:
    calls = re.findall(r"plan_collective\((.*?)\)\n", _SCHEDULING.read_text(), flags=re.S)
    assert calls, "scheduling.py no longer calls plan_collective; re-check the residency docs"
    return any("datasets" in call for call in calls)


def _page_text(path: Path) -> str:
    # Pages hard-wrap prose, so a phrase can straddle a newline.
    return " ".join(path.read_text().split())


def test_docs_do_not_claim_residency_is_enforced_by_the_scheduler() -> None:
    if _scheduler_passes_datasets():
        pytest.fail(
            "scheduling.py now passes datasets to plan_collective: update the residency docs "
            "to describe the enforcement, then rewrite this test"
        )
    for page in _PAGES:
        text = _page_text(page)
        for phrase in _RETIRED:
            assert phrase not in text, f"{page.relative_to(_ROOT)} claims: {phrase!r}"


def test_docs_say_the_scheduler_does_not_consult_the_catalog() -> None:
    # The positive control: removing the false claim is not enough if the caveat is dropped too.
    for page in _PAGES[:2]:
        assert "scheduler does not" in _page_text(page), page.relative_to(_ROOT)
    assert "The scheduler passes neither" in _page_text(_PAGES[2])
