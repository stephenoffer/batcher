"""The codemod's receiver table matches the live annotations it was generated from.

`batcher.migrate` decides which names in a script are Batcher objects by following
`returns.toml`, which `tools/parity/gen_migration_returns.py` generates from the engine's own
return annotations. A method added or retyped without regenerating the table leaves the codemod
blind past that call, and the symptom is a rename that silently does not happen. So the table is
regenerated in memory here and compared with the committed file.
"""

from __future__ import annotations

from tools.parity.gen_migration_returns import generate, render

from batcher._internal.migration.loader import DATA_DIR


def test_returns_table_is_fresh() -> None:
    expected = render(generate())
    committed = (DATA_DIR / "returns.toml").read_text()
    assert committed == expected, "run `python tools/parity/gen_migration_returns.py`"


def test_the_table_carries_the_chains_the_codemod_relies_on() -> None:
    # Positive control: a stale-but-empty table would compare equal to an empty regeneration.
    import tomllib

    table = tomllib.loads((DATA_DIR / "returns.toml").read_text())
    assert table["bt"]["from_pydict"] == "Dataset"
    assert table["Dataset"]["group_by"] == "GroupBy"
    assert table["Expr"]["str"] == "Expr.str"
    assert table["bt.read"]["__call__"] == "Dataset"
