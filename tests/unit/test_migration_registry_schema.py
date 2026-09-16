"""A migration-registry row that cannot be acted on fails at load time, not as a blank docs cell.

The registry is read by the migration-error guidance, the `batcher.migrate` codemod and the
docs generator. Each of them needs a specific field for each kind of row: a mismatch is useless
without the note saying what differs, a gap without the wave that closes it, a decline without
its reason. These tests load hand-built registry trees and check that the loader refuses each
malformed shape, and accepts the well-formed one, so the rules in `schema.py` are exercised
rather than merely declared.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from batcher._internal.migration import RegistryError, Status, load_registry


def _tree(tmp_path: Path, body: str, engine: str = "polars", name: str = "rows.toml") -> Path:
    (tmp_path / engine).mkdir(parents=True, exist_ok=True)
    (tmp_path / engine / name).write_text(body)
    return tmp_path


def test_a_well_formed_tree_loads_every_status(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        """
[LazyFrame]
with_columns = { status = "canonical", batcher = "Dataset.with_columns" }
melt = { status = "alias", batcher = "Dataset.unpivot" }
join_asof = { status = "param", batcher = "Dataset.join_asof", need = "strategy", wave = "W2" }
join_where = { status = "gap", need = "inequality join", wave = "W5" }
sort = { status = "mismatch", batcher = "Dataset.sort", note = "nulls first", wave = "W0" }
rechunk = { status = "out_of_scope", reason = "engine-owned layout" }

["Expr.str"]
concat = { status = "canonical", batcher = ["Expr.str.join", "op:add"] }
""",
    )
    registry = load_registry(root)
    assert len(registry.rows) == 7
    assert {r.status for r in registry.rows.values()} == set(Status)
    concat = registry.get("polars", "Expr.str", "concat")
    assert concat is not None and concat.batcher == ("Expr.str.join", "op:add")
    assert registry.get("polars", "LazyFrame", "nope") is None


@pytest.mark.parametrize(
    ("row", "complaint"),
    [
        ('{ status = "mismatch", batcher = "Dataset.sort", wave = "W0" }', "requires 'note'"),
        ('{ status = "gap", need = "a kernel" }', "requires 'wave'"),
        ('{ status = "out_of_scope" }', "requires 'reason'"),
        ('{ status = "canonical" }', "requires 'batcher'"),
        ('{ status = "param", batcher = "Dataset.x", need = "n", wave = "W99" }', "wave 'W99'"),
        ('{ status = "sorta" }', "is not a Status"),
        ('{ status = "canonical", batcher = "Dataset.x", colour = "red" }', "unknown field"),
        ('{ status = "canonical", batcher = "Dataset..x" }', "not a dotted path"),
    ],
)
def test_a_row_missing_what_makes_it_actionable_is_rejected(
    tmp_path: Path, row: str, complaint: str
) -> None:
    root = _tree(tmp_path, f"[LazyFrame]\nsort = {row}\n")
    with pytest.raises(RegistryError, match=complaint):
        load_registry(root)


def test_the_same_name_classified_in_two_files_is_rejected(tmp_path: Path) -> None:
    row = '[Expr]\nabs = { status = "canonical", batcher = "Expr.abs" }\n'
    _tree(tmp_path, row, name="a.toml")
    root = _tree(tmp_path, row, name="b.toml")
    with pytest.raises(RegistryError, match="classified twice"):
        load_registry(root)
