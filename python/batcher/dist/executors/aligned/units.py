"""Cut the key domain into units from the files' footer ranges, and prove the layout holds.

A *unit* is one half-open key range plus, per aligned source, the files whose footer range
overlaps it. It is the whole of what one engine call on a worker reads for that source.
Units are built from the largest aligned source (its files, sorted by footer minimum, are
grouped until a unit holds about `unit_bytes` of projected data), and every other aligned
source is then matched against those ranges by overlap.

Correctness never rests on this layout. Every aligned scan is filtered to its unit's range
before anything reads it (`run`), so a file that straddles two units is read by both and
contributes each row to exactly one. What the layout decides is *cost*, and a source whose
files do not follow the key would be read by every unit; `plan_units` measures exactly that
(the mean number of units each file lands in) and declines past `_MAX_OVERLAP`.
"""

from __future__ import annotations

import dataclasses
import datetime
import itertools
import operator
from typing import Any

from batcher._internal.logging import note_suppressed
from batcher.io.source import Source

__all__ = [
    "Unit",
    "clustered",
    "footer_columns",
    "implied_by_bounds",
    "plan_units",
    "projected_bytes",
    "projected_share",
    "source_key_bounds",
    "splits_by_file",
    "splits_by_piece",
]

# Mean units a file of an aligned source may overlap before its layout is judged not to
# follow the key. A clustered table's boundary files straddle two units, so a little above
# one is the norm; a randomly laid out one lands near the unit count.
_MAX_OVERLAP = 1.5

# Share of a source's files, sorted by footer minimum, that may overlap the next one before its
# layout is judged not to follow the key. Zero for a table written in key order.
_MAX_NEIGHBOR_OVERLAP = 0.1

_ORDERED = (int, datetime.date, datetime.datetime)


@dataclasses.dataclass(frozen=True)
class Unit:
    """One key range and the files of each aligned source that may hold keys in it.

    `lo` is inclusive and `hi` exclusive; either is None for an open end, so the first and
    last units together cover every key, including ones no footer anticipated.
    """

    lo: Any
    hi: Any
    files: dict[int, tuple[str, ...]]
    nbytes: int


def source_key_bounds(source: Source, column: str) -> list[Any] | None:
    """Per-file footer bounds of `column` for a Parquet source, or None when unavailable."""
    bounds = getattr(source, "key_bounds", None)
    if bounds is None:
        return None
    try:
        out = bounds(column)
    except Exception as exc:
        note_suppressed("dist", "read aligned key bounds", exc)
        return None
    if not out or not all(isinstance(b.lo, _ORDERED) and isinstance(b.hi, _ORDERED) for b in out):
        return None
    # A NULL key falls in no range, yet an outer or anti join on the aligned side, or a group
    # on the key, must still see its row once; and NULLs sit in every file, not in the files
    # of one range. So a key column that may hold NULLs is not a key to align on.
    if any(b.nulls != 0 for b in out):
        return None
    return out


#: Split target for aligned reads: below any real row group, so a split never spans files.
#: A coarser target packs small files into one multi-file split (TPC-H SF1000 `partsupp`,
#: two files a split at 64 MB), and a split that is no single file's cannot follow a range.
UNIT_SPLIT_BYTES = 1 << 20


def split_files(split: Any) -> tuple[str, ...] | None:
    """The files `split` reads, in order, or None when it names none (a whole-source read)."""
    paths = getattr(split, "paths", None)
    if paths:
        return tuple(paths)
    path = getattr(split, "path", None)
    return (path,) if path else None


def piece_key(split: Any) -> str | None:
    """The row group `split` starts at, as `path#index`: what a keyless unit is cut from.

    None for a split naming no single file. A split whose row groups carry no index is keyed
    by its file alone.
    """
    files = split_files(split)
    if files is None or len(files) != 1:
        return None
    row_groups = getattr(split, "row_groups", None)
    return f"{files[0]}#{row_groups[0]}" if row_groups else files[0]


def file_bounds(source: Source) -> list[Any] | None:
    """Pseudo key bounds for splitting `source` by row group: each its own one-point range.

    The key is the row group's position, so `plan_units` groups whole row groups into units
    and none lands in two; the unit filter is skipped for a keyless class (`run.unit_plan`).
    Row groups rather than files: TPC-H SF10 `lineitem` is 10 files of 6 row groups, and cut
    by file its 10 units left 6 of 16 streams idle.
    """
    from batcher.io.source import plan_splits
    from batcher.io.splits.parquet import FileKeyBounds

    try:
        splits = plan_splits(source, target_size=UNIT_SPLIT_BYTES)
    except Exception as exc:
        note_suppressed("dist", "plan splits for a file-split source", exc)
        return None
    pieces: dict[str, list[int]] = {}
    for split in splits:
        nbytes = getattr(split, "nbytes", None)
        # Split by piece means each split in exactly one unit, which a split over several
        # files cannot promise without a range filter to drop the rest.
        if piece_key(split) is None or nbytes is None:
            return None
        # Keyed per row group, so a split the pushed predicate later regroups still starts
        # at a key some unit owns (`splits_by_piece`).
        groups = getattr(split, "row_groups", None) or (None,)
        rows = getattr(split, "rows", 0) or 0
        for group in groups:
            key = piece_key(split) if group is None else f"{split_files(split)[0]}#{group}"
            pieces[key] = [rows // len(groups), nbytes // len(groups)]
    return [
        FileKeyBounds(k, i, i, rows, nbytes) for i, (k, (rows, nbytes)) in enumerate(pieces.items())
    ]


def clustered(bounds: list[Any]) -> bool:
    """Whether files with these footer ranges store the key in order (nearly disjoint ranges)."""
    ordered = sorted(bounds, key=lambda b: b.lo)
    # Files that merely touch (one's last key is the next one's first) are in order; the
    # half-open unit filter reads the shared key once.
    overlaps = sum(1 for a, b in itertools.pairwise(ordered) if b.lo < a.hi)
    return overlaps <= _MAX_NEIGHBOR_OVERLAP * max(1, len(ordered) - 1)


def projected_bytes(source: Source, projection: list[str] | None) -> int | None:
    """Decoded bytes of `source` restricted to `projection`, or None when unknown."""
    sizes_of = getattr(source, "column_byte_sizes", None)
    if sizes_of is not None:
        sizes = sizes_of()
        if sizes:
            cols = sizes if projection is None else [c for c in projection if c in sizes]
            return int(sum(sizes[c] for c in cols))
    try:
        stats = source.statistics()
    except Exception as exc:
        note_suppressed("dist", "read source statistics for sizing", exc)
        return None
    size = getattr(stats, "byte_size", None) if stats is not None else None
    if size is None:
        return None
    try:
        width = len(source.schema())
    except Exception:
        width = 0
    share = 1.0 if projection is None or not width else min(1.0, len(projection) / width)
    return int(size * share)


#: Projected bytes a round of units must give each stream before a small input is cut into
#: another round. A unit's read costs ~0.2 s whatever its size -- one 28 MB row group of SF10
#: `lineitem` took 0.2 s, a 113 MB unit 0.3 s -- so a small input runs fastest as whole rounds
#: of one unit per stream: TPC-H q22's 0.5 GB anti join at SF100, cut into 100 units for 16
#: streams, took 2.7 s where one round is its reads, and SF10 `lineitem` cut into 60 row
#: groups for 24 streams paid three rounds of that latency where 24 units pay one.
SMALL_UNIT_BYTES = 64 << 20


def plan_units(
    bounds: dict[int, list[Any]],
    shares: dict[int, float],
    unit_bytes: int,
    min_units: int,
    streams: int | None = None,
) -> list[Unit] | None:
    """The units for aligned sources with the given per-file `bounds`, or None to decline.

    Args:
        bounds: Per aligned source id, its files' `FileKeyBounds`.
        shares: Per aligned source id, the fraction of a file's bytes the query reads.
        unit_bytes: Target projected bytes of the driving source per unit.
        min_units: Units to cut a large input into, several per stream, which evens out a
            key range denser than the rest.
        streams: Units that run at once. Given, a small input is cut into whole rounds of
            one unit per stream, as many rounds as it fills with `SMALL_UNIT_BYTES` each.

    Returns:
        The units in key order, or None when some source's files do not follow the key.
    """
    driving = max(bounds, key=lambda s: sum(b.nbytes for b in bounds[s]) * shares.get(s, 1.0))
    files = sorted(bounds[driving], key=lambda b: b.lo)
    share = shares.get(driving, 1.0)
    total = sum(b.nbytes for b in files) * share
    per_unit = total / max(1, min_units)
    if streams:
        rounds = max(1, int(total // (streams * SMALL_UNIT_BYTES)))
        per_unit = max(per_unit, total / (streams * rounds))
    per_unit = max(1.0, min(float(unit_bytes), per_unit))
    groups: list[list[Any]] = [[]]
    acc = 0.0
    for f in files:
        size = f.nbytes * share
        if groups[-1] and acc + size > per_unit * 1.25:
            groups.append([])
            acc = 0.0
        groups[-1].append(f)
        acc += size
    starts = [g[0].lo for g in groups]
    ranges = [
        (None if i == 0 else starts[i], None if i == len(starts) - 1 else starts[i + 1])
        for i in range(len(starts))
    ]
    units = []
    for (lo, hi), group in zip(ranges, groups, strict=True):
        per_source = {
            sid: tuple(b.path for b in bs if _overlaps(b, lo, hi)) for sid, bs in bounds.items()
        }
        units.append(Unit(lo, hi, per_source, int(sum(f.nbytes for f in group) * share)))
    for sid, bs in bounds.items():
        # A table with fewer, wider files than there are units spans several units per file
        # by construction (the unit's range filter prunes the rest of each file's row
        # groups); only spread past that means the files do not follow the key.
        hits = sum(len(u.files[sid]) for u in units)
        expected = max(_MAX_OVERLAP, 2.0 * len(units) / len(bs)) if bs else 0.0
        if bs and hits / len(bs) > expected:
            return None
    return units


def _overlaps(b: Any, lo: Any, hi: Any) -> bool:
    """Whether the closed file range `[b.lo, b.hi]` meets the half-open `[lo, hi)`."""
    return (lo is None or b.hi >= lo) and (hi is None or b.lo < hi)


def projected_share(source: Source, projection: list[str] | None) -> float:
    full = projected_bytes(source, None)
    part = projected_bytes(source, projection)
    return 1.0 if not full or part is None else min(1.0, part / full)


def splits_by_piece(source: Source, projection, predicate) -> dict[str, list] | None:
    """Every split of `source` under the row group it starts at (`piece_key`); None if one
    names no single file."""
    from batcher.io.source import plan_splits

    out: dict[str, list] = {}
    for split in plan_splits(
        source, target_size=UNIT_SPLIT_BYTES, predicate=predicate, projection=projection
    ):
        key = piece_key(split)
        if key is None:
            return None
        out.setdefault(key, []).append(split)
    return out


def splits_by_file(source: Source, projection, predicate) -> dict[str, list] | None:
    """Every split of `source`, under each file it reads; None if one names no file.

    A split over several files is listed under each, so a unit needing any of them reads it
    whole; the unit's range filter drops the rows outside its range, which keeps the result
    exact. A split naming no file (a whole-source read) cannot be placed at all, and reading
    nothing for it would silently lose its rows, so that declines the cut.
    """
    from batcher.io.source import plan_splits

    out: dict[str, list] = {}
    for split in plan_splits(
        source, target_size=UNIT_SPLIT_BYTES, predicate=predicate, projection=projection
    ):
        files = split_files(split)
        if files is None:
            return None
        for path in files:
            out.setdefault(path, []).append(split)
    return out


def footer_columns(source: Source | None) -> dict:
    """The per-column statistics `source` declares, or an empty mapping when it has none."""
    if source is None:
        return {}
    try:
        stats = source.statistics()
    except Exception as exc:
        note_suppressed("dist", "read source statistics for a broadcast filter", exc)
        return {}
    return dict(getattr(stats, "columns", None) or {})


#: `column <op> literal` holds for every row when the column's bound on that side does:
#: (which bound, the comparison it must pass).
_IMPLIED_BY = {
    "ge": ("min", operator.ge),
    "gt": ("min", operator.gt),
    "le": ("max", operator.le),
    "lt": ("max", operator.lt),
}


def implied_by_bounds(ir: dict, columns: dict) -> bool:
    """Whether a `column <op> literal` range conjunct holds for every row of the source.

    Only an integer literal against an integer bound is judged, so no type coercion is
    guessed at; anything else is taken to filter, which is the old, conservative answer.
    """
    if ir.get("e") != "binary" or ir.get("op") not in _IMPLIED_BY:
        return False
    left, right = ir.get("left", {}), ir.get("right", {})
    if left.get("e") != "col" or right.get("e") != "lit":
        return False
    side, passes = _IMPLIED_BY[ir["op"]]
    bound = getattr(columns.get(left.get("name")), side, None)
    value = (right.get("value") or {}).get("int")
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (value, bound)):
        return False
    return passes(bound, value)
