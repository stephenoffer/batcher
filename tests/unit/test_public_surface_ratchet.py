"""A public name may be added freely, but removed only through the migration registry.

`tools/public_surface_baseline.txt` records every public spelling. A spelling that stops
resolving fails here unless `_internal/migration/data/renames.toml` records its removal,
which is what gives the `batcher.migrate` codemod and the `AttributeError` guidance
something to say about it. See `tools/public_surface_baseline.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import public_surface_baseline as baseline  # noqa: E402


def test_no_public_name_vanishes_without_a_recorded_removal() -> None:
    committed = baseline.read_baseline()
    assert len(committed) > 1000  # the baseline was actually read
    gone = committed - baseline.spellings() - baseline.recorded_removals()
    assert not gone, (
        f"{len(gone)} released public name(s) no longer resolve: {sorted(gone)[:20]}. "
        "Restore them, or record each removal in "
        "python/batcher/_internal/migration/data/renames.toml under its receiver."
    )


def test_baseline_is_current() -> None:
    """New public names are recorded, so a later removal of one is caught too."""
    added = baseline.spellings() - baseline.read_baseline()
    assert not added, f"run `just surface-baseline` to record {sorted(added)[:20]}"


def test_regeneration_keeps_an_unrecorded_removal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rewriting the baseline cannot launder a removal that renames.toml does not record."""
    current = baseline.spellings()
    monkeypatch.setattr(baseline, "read_baseline", lambda: current | {"Dataset.vanished_verb"})
    assert "Dataset.vanished_verb" in baseline.next_baseline()
    monkeypatch.setattr(baseline, "recorded_removals", lambda: {"Dataset.vanished_verb"})
    assert "Dataset.vanished_verb" not in baseline.next_baseline()


def test_spellings_follow_the_user_not_the_module() -> None:
    names = baseline.spellings()
    for spelling in ("bt.col", "bt.Dataset", "Dataset.filter", "Expr.str.upper", "GroupBy.agg"):
        assert spelling in names, spelling
    assert not any(n.startswith("batcher.plan.") for n in names)


def test_a_recorded_removal_is_spelled_like_the_baseline() -> None:
    """`renames.toml` keys line up with baseline spellings, so a recorded removal can match."""
    removals = baseline.recorded_removals()
    assert "Dataset.groupby" in removals
    assert "Dataset.group_by" in baseline.spellings()
