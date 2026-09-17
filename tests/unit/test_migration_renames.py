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

from batcher._internal.migration import RegistryError, Rename, load_kwarg_renames, load_renames


@pytest.fixture(scope="module")
def renames() -> dict[str, dict[str, Rename]]:
    return load_renames()


def test_no_second_spelling_is_left_on_the_surface() -> None:
    # The alias removal is done: the lint must find nothing blocking. Its ability to find a
    # second spelling at all is proven on planted classes in tests/unit/test_lint_aliases.py,
    # which is what keeps this from passing vacuously.
    findings = [f for f in find_all() if f.kind in BLOCKING and f.key not in ALLOW]
    assert not findings, f"{len(findings)} second spellings on the surface: {findings}"


def test_no_removed_spelling_still_resolves(renames) -> None:
    alive = [
        f"{rule.receiver}.{rule.removed}"
        for table in renames.values()
        for rule in table.values()
        if resolve(f"{rule.receiver}.{rule.removed}") is not None
    ]
    assert not alive, f"removed spellings still resolve: {alive}"


def _kept_target(rule: Rename) -> str:
    if rule.transform == "identity":
        return rule.receiver  # the call collapses to the receiver itself
    if rule.kind == "operator":
        from batcher._internal.migration.renames import OPERATORS

        return f"op:{rule.operator}" if rule.operator in OPERATORS else ""
    return f"{rule.receiver}.{rule.to}"


def test_every_kept_spelling_exists(renames) -> None:
    missing = sorted(
        {
            f"{rule.receiver}.{rule.removed} -> {_kept_target(rule)}"
            for table in renames.values()
            for rule in table.values()
            if _resolves(_kept_target(rule)) is None
        }
    )
    assert not missing, f"renames point at spellings that do not exist: {missing}"


def _resolves(target: str):
    # A dotted path under a receiver (`Dataset.write.csv`) resolves through the accessor.
    if target.startswith("op:"):
        return resolve(target)
    parts = target.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        head = ".".join(parts[:cut])
        if resolve(head) is not None or head in ("bt", "Dataset", "Expr"):
            tail = parts[cut:]
            if len(tail) == 1:
                return resolve(target)
            return resolve(f"{head}.{tail[0]}.{tail[1]}") if len(tail) == 2 else None
    return resolve(target)


def test_every_keyword_rule_names_a_real_parameter(renames) -> None:
    import inspect

    stale = []
    for method, rules in load_kwarg_renames().items():
        receiver, _, name = method.rpartition(".")
        # A table for a removed method (`Dataset.melt`) describes the call before its rename,
        # so its kept keywords must exist on the method it is renamed to.
        rule = renames.get(receiver, {}).get(name)
        target = f"{receiver}.{rule.to}" if rule is not None else method
        fn = resolve(target)
        if fn is None:
            stale.append(f"{method} -> {target} does not resolve")
            continue
        params = inspect.signature(fn).parameters
        for keyword, kw in rules.items():
            if kw.action in ("rename", "negate") and kw.to not in params:
                stale.append(f"{method}({keyword}=) -> {target}({kw.to}=)")
    assert not stale, f"keyword rules naming parameters that do not exist: {stale}"


def test_a_rename_chain_is_rejected(tmp_path, monkeypatch) -> None:
    from batcher._internal.migration import renames as renames_module

    (tmp_path / "renames.toml").write_text('[Expr]\nisna = "isnull"\nisnull = "is_null"\n')
    monkeypatch.setattr(renames_module, "_DATA", tmp_path)
    renames_module.load_renames.cache_clear()
    try:
        with pytest.raises(RegistryError, match="removed too"):
            renames_module.load_renames()
    finally:
        renames_module.load_renames.cache_clear()
