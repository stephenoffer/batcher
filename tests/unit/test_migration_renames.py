"""Every second spelling the alias lint finds has a decision, and every decision is usable.

`tools/lint_aliases.py` finds pairs; it cannot say which name survives. That is a naming
decision, recorded in `python/batcher/_internal/migration/data/renames.toml`, and the codemod
and the migration-error guidance both apply it. So a pair the lint finds with no decision is
a rename nobody can perform, and a decision naming a spelling that does not exist rewrites
user code into an `AttributeError`. These tests hold both ends.
"""

from __future__ import annotations

import pytest
from tools.lint_aliases import ALLOW, BLOCKING, find_all
from tools.parity.batcher_targets import resolve

from batcher._internal.migration import RegistryError, load_renames


@pytest.fixture(scope="module")
def renames() -> dict[str, dict[str, str]]:
    return load_renames()


def test_every_blocking_pair_has_a_decision(renames) -> None:
    findings = [f for f in find_all() if f.kind in BLOCKING and f.key not in ALLOW]
    assert findings, "the alias lint found nothing, so this test would pass vacuously"
    undecided = []
    for f in findings:
        table = renames.get(f.receiver, {})
        if f.name in table or table.get(f.target) == f.name:
            continue
        # Both names of a same-IR group can be removed in favour of a third spelling.
        if f.target in table and f.name in table.values():
            continue
        undecided.append(f"{f.key} ~ {f.target}")
    assert not undecided, f"{len(undecided)} second spellings with no decision: {undecided}"


def test_every_kept_spelling_exists(renames) -> None:
    missing = [
        f"{receiver}.{kept}"
        for receiver, table in renames.items()
        for kept in set(table.values())
        if resolve(f"{receiver}.{kept}") is None
    ]
    assert not missing, f"renames point at spellings that do not exist: {missing}"


def test_a_rename_chain_is_rejected(tmp_path, monkeypatch) -> None:
    from batcher._internal.migration import loader

    (tmp_path / "renames.toml").write_text('[Expr]\nisna = "isnull"\nisnull = "is_null"\n')
    monkeypatch.setattr(loader, "DATA_DIR", tmp_path)
    loader.load_renames.cache_clear()
    try:
        with pytest.raises(RegistryError, match="removed too"):
            loader.load_renames()
    finally:
        loader.load_renames.cache_clear()
