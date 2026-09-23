"""The foreign codemod's receiver and signature tables match the libraries they were read from.

`batcher.migrate` never imports PySpark, Polars, Daft or Ray Data: it follows
`data/codemod/<engine>.toml`, generated from the installed libraries by
`tools/parity/gen_codemod_tables.py`. A library upgrade (or a Batcher signature change) that is
not regenerated leaves the codemod blind past a changed call, and the symptom is a rewrite that
silently stops happening. So each table is regenerated in memory and compared with the committed
file, for every engine installed here.
"""

from __future__ import annotations

import importlib.util
import tomllib

import pytest
from tools.parity.gen_codemod_tables import generate_batcher, generate_engine, render

from batcher._internal.migration.loader import DATA_DIR

_MODULES = {"pyspark": "pyspark", "polars": "polars", "daft": "daft", "ray_data": "ray.data"}


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _missing_surface(engine: str) -> str | None:
    """The first surface `engine` declares that cannot be resolved here, or None.

    The top-level module being importable is not the same as the table being regenerable.
    `ray.data` ships with `ray[default]`; `ray.data.llm`, which `surfaces.py` declares and
    the committed table carries a `params."llm"` section for, needs the `ray[llm]` extra.
    Guarding on `ray.data` alone therefore ran the regeneration on a box that could not
    perform it and reported a missing optional extra as a stale table.
    """
    from tools.parity.surfaces import SURFACES, _resolve

    for _label, target, _kind in SURFACES.get(engine, ()):
        try:
            _resolve(target)
        except Exception:  # any resolution failure means "not available on this box"
            return target
    return None


@pytest.mark.parametrize("engine", sorted(_MODULES))
def test_engine_table_is_fresh(engine: str) -> None:
    if not _installed(_MODULES[engine]):
        pytest.skip(f"{engine} is not installed, so its table cannot be regenerated here")
    missing = _missing_surface(engine)
    if missing is not None:
        pytest.skip(f"{engine}: {missing} is not importable here, so its table is not regenerable")
    committed = (DATA_DIR / "codemod" / f"{engine}.toml").read_text()
    assert committed == render(generate_engine(engine), engine), (
        f"run `python tools/parity/gen_codemod_tables.py {engine}`"
    )


def test_batcher_table_is_fresh() -> None:
    committed = (DATA_DIR / "codemod" / "batcher.toml").read_text()
    assert committed == render(generate_batcher(), "batcher"), (
        "run `python tools/parity/gen_codemod_tables.py batcher`"
    )


def test_the_tables_carry_the_chains_the_rewrites_rely_on() -> None:
    # Positive control: an empty regeneration compared with an empty file would pass above.
    tables = {e: tomllib.loads((DATA_DIR / "codemod" / f"{e}.toml").read_text()) for e in _MODULES}
    assert tables["pyspark"]["returns"]["SparkSession"]["read"] == "DataFrameReader"
    assert tables["pyspark"]["returns"]["DataFrame"]["groupBy"] == "GroupedData"
    assert tables["pyspark"]["params"]["functions"]["upper"] == ["col~"]
    assert tables["polars"]["returns"]["polars"]["col"] == "Expr"
    assert tables["polars"]["returns"]["Expr"]["str"] == "Expr.str"
    assert tables["polars"]["returns"]["When"]["then"] == "Then"
    assert tables["daft"]["returns"]["DataFrame"]["groupby"] == "GroupedDataFrame"
    assert tables["ray_data"]["returns"]["ray.data"]["from_items"] == "Dataset"
    batcher = tomllib.loads((DATA_DIR / "codemod" / "batcher.toml").read_text())
    assert "descending=" in batcher["params"]["Dataset"]["sort"]
