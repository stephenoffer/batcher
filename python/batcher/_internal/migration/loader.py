"""Load the migration registry from its TOML files.

The files live beside this module, one directory per competitor
(`data/<engine>/*.toml`). Inside a file each top-level table is one surface and each key
one name on it:

.. code-block:: toml

    [DataFrame]
    withColumn = { status = "canonical", batcher = "Dataset.with_columns" }

Keying rows by name gives the one-row-per-name rule for free, since TOML rejects a
duplicate key. A surface may still be split across files (a large function module by
family), so the loader also rejects the same `(surface, name)` arriving from two files.

The registry is parsed once per process and cached: the guidance tables consult it on
every failed attribute lookup, and re-reading several thousand rows there would turn a
typo into a noticeable pause.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from batcher._internal.migration.schema import ENGINES, Mapping, RegistryError, Status, validate

__all__ = ["DATA_DIR", "Registry", "load_codemod_tables", "load_registry", "load_returns"]

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class Registry:
    """Every registry row, indexed by `(engine, surface, name)`."""

    rows: dict[tuple[str, str, str], Mapping] = field(default_factory=dict)

    def get(self, engine: str, surface: str, name: str) -> Mapping | None:
        """Return the row for one name on one surface, or `None` when unclassified.

        Args:
            engine: The competitor.
            surface: The receiver the name is typed on.
            name: The competitor's spelling.

        Returns:
            The row, or `None`.
        """
        return self.rows.get((engine, surface, name))

    def for_engine(self, engine: str) -> Iterator[Mapping]:
        """Yield every row for one competitor.

        Args:
            engine: The competitor.

        Returns:
            An iterator over that engine's rows, in file order.
        """
        return (row for key, row in self.rows.items() if key[0] == engine)

    def with_status(self, status: Status) -> Iterator[Mapping]:
        """Yield every row, across engines, with one status.

        Args:
            status: The status to select.

        Returns:
            An iterator over the matching rows.
        """
        return (row for row in self.rows.values() if row.status is status)


def _parse(engine: str, path: Path, rows: dict[tuple[str, str, str], Mapping]) -> None:
    try:
        doc = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise RegistryError(f"{path}: {exc}") from exc
    for surface, entries in doc.items():
        if not isinstance(entries, dict):
            raise RegistryError(f"{path}: top-level key {surface!r} must be a surface table")
        for name, raw in entries.items():
            if not isinstance(raw, dict):
                raise RegistryError(f"{path}: {surface}.{name} must be an inline table")
            key = (engine, surface, name)
            if key in rows:
                raise RegistryError(f"{path}: {surface}.{name} is classified twice")
            rows[key] = validate(engine, surface, name, raw)


def load_registry(data_dir: Path = DATA_DIR) -> Registry:
    """Parse every registry file under `data_dir`.

    Args:
        data_dir: The directory holding one subdirectory per engine. Tests point this at a
            fixture tree; everything else uses the default.

    Returns:
        The parsed, validated registry.

    Raises:
        RegistryError: When any file or row is malformed.
    """
    if data_dir == DATA_DIR:
        return _cached()
    return _load(data_dir)


@lru_cache(maxsize=1)
def _cached() -> Registry:
    return _load(DATA_DIR)


def _load(data_dir: Path) -> Registry:
    rows: dict[tuple[str, str, str], Mapping] = {}
    for engine in ENGINES:
        for path in sorted((data_dir / engine).glob("*.toml")):
            _parse(engine, path, rows)
    return Registry(rows)


@lru_cache(maxsize=1)
def load_returns() -> dict[str, dict[str, str]]:
    """What each Batcher member returns, as `{receiver: {member: returned_receiver}}`.

    Generated from the live annotations by `tools/parity/gen_migration_returns.py`; the
    codemod follows it to decide which names in a script are Batcher objects.

    Returns:
        The table in `data/returns.toml`.
    """
    return tomllib.loads((DATA_DIR / "returns.toml").read_text())


@lru_cache(maxsize=8)
def load_codemod_tables(engine: str) -> dict[str, dict[str, dict[str, object]]]:
    """What each member of an engine returns, and its parameters, for the foreign codemod.

    Generated from the installed libraries by `tools/parity/gen_codemod_tables.py`. Each
    engine's file holds `returns` and `params` tables; `batcher.toml` holds only `params`.

    Args:
        engine: One of `ENGINES`, or `"batcher"`.

    Returns:
        `{"returns": {surface: {member: surface}}, "params": {surface: {member: tokens}}}`.

    Raises:
        RegistryError: When no table exists for `engine`.
    """
    path = DATA_DIR / "codemod" / f"{engine}.toml"
    if engine not in (*ENGINES, "batcher") or not path.is_file():
        raise RegistryError(f"no codemod tables for engine {engine!r}")
    return tomllib.loads(path.read_text())
