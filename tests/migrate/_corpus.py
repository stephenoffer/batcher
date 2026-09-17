"""The golden corpus the codemod tests share, and the one command that regenerates it.

Each case is a directory `corpus/<engine>/<case>/` holding three files:

* `src.py`, a script written against the foreign engine;
* `batcher.py`, what `translate` makes of it (the foreign-to-Batcher golden);
* `back.py`, what `export` makes of `batcher.py` (the Batcher-to-foreign golden).

A golden is a reviewed file, not a snapshot to refresh on red: regenerate with
`python tests/migrate/_corpus.py` and read the diff before committing it.
"""

from __future__ import annotations

import sys
from pathlib import Path

CORPUS = Path(__file__).resolve().parent / "corpus"
ENGINES = ("pyspark", "polars", "daft", "ray_data")

# Cases whose `src.py` and `batcher.py` both run on in-memory data and bind `result` to a list of
# row dicts, so the executed-equivalence tests can compare the two engines' answers. A case whose
# program sorts before collecting compares in order; the rest compare as multisets.
EXECUTED: dict[str, dict[str, bool]] = {
    "polars": {
        "verbs": True,
        "expressions": True,
        "aggregation": True,
        "sort_join": True,
        "windows": True,
        "lazy": True,
    },
    "daft": {"verbs": True, "expressions": True, "aggregation": True, "sort_join": True},
    "ray_data": {"verbs": True, "map_batches": False, "aggregation": True, "sort": True},
    "pyspark": {"verbs": True, "aggregation": True, "sort_join": True, "windows": True},
}


def cases(engine: str) -> list[Path]:
    """Every case directory of one engine, sorted.

    Args:
        engine: A registry engine.

    Returns:
        The case directories.
    """
    return sorted(p for p in (CORPUS / engine).iterdir() if (p / "src.py").is_file())


def _rows(value: object) -> list[dict[str, object]]:
    """Rows as plain dicts: a PySpark `Row` answers `asDict`, everything else already is one."""
    out = []
    for row in value:  # type: ignore[attr-defined]
        found = row.asDict() if hasattr(row, "asDict") else dict(row)
        out.append({k: (round(v, 9) if isinstance(v, float) else v) for k, v in found.items()})
    return out


def run(path: Path) -> list[dict[str, object]]:
    """Execute one corpus program and return the rows it binds to `result`.

    Args:
        path: A `src.py` or `batcher.py`.

    Returns:
        The rows, with floats rounded so two engines' last bits cannot differ.
    """
    namespace: dict[str, object] = {"__name__": "__corpus__"}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return _rows(namespace["result"])


def equivalent(engine: str, case: str) -> tuple[object, object]:
    """Both sides of one executed case, ready for an equality assertion.

    Args:
        engine: A registry engine.
        case: A case name in `EXECUTED[engine]`.

    Returns:
        `(source_rows, batcher_rows)`: lists when the case sorts, else sorted multisets.
    """
    source = run(CORPUS / engine / case / "src.py")
    translated = run(CORPUS / engine / case / "batcher.py")
    if EXECUTED[engine][case]:
        return source, translated
    return sorted(map(repr, source)), sorted(map(repr, translated))


def regenerate() -> int:
    """Rewrite every `batcher.py` and `back.py` from its `src.py`."""
    from batcher.migrate.outbound import export
    from batcher.migrate.translate import translate

    for engine in ENGINES:
        for case in cases(engine):
            forward, _ = translate((case / "src.py").read_text(), engine)
            (case / "batcher.py").write_text(forward)
            backward, _ = export(forward, engine)
            (case / "back.py").write_text(backward)
            print(f"{engine}/{case.name}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    raise SystemExit(regenerate())
