"""Whether a file's compression is one the device can undo, or one that lands back on the CPU.

A device Parquet read is worth taking on because the decode happens where the compute is. That
argument has a precondition nobody states: the *decompression* has to happen there too. Parquet
pages are compressed, and a device reader handed a codec it has no kernel for does not fail —
it copies the pages to the host, decompresses them there, and copies them back.

That path is strictly worse than the host reader it replaced. It crosses PCIe twice instead of
once, it uses the host cores anyway, and it does it while holding device memory for the result.
And it looks, in every log and every plan, exactly like a successful device read.

**This is a performance veto, not a correctness one, and the difference decides its default.**
The sibling type check in `device` refuses a device read whenever it cannot *prove* the two
readers agree, because being wrong there produces wrong rows. Being wrong here produces a slow
read, so the default runs the other way: a codec is disqualifying only on evidence that it is
one, and a footer nobody could read leaves the decision exactly where it was. Vetoing on doubt
would silently disable device reads for every corpus whose metadata is momentarily
unavailable — which, before this module read through the shared footer cache, was every object
store.

**A short allowlist**, for the same reason `device` keeps one for types. Snappy and Zstd are the
codecs the device reader has decompression kernels for, and a Parquet corpus is overwhelmingly
written in one of them or in neither. An unlisted codec is not one the device *cannot*
eventually undo; it is one nobody has checked.

**The footer read is shared, not repeated.** It goes through the same identity-keyed cache the
row-group splitter uses, so on the path that matters — where the splits came from row groups,
and the footer was therefore already read — this check costs a dictionary lookup.
"""

from __future__ import annotations

__all__ = [
    "DEVICE_CODECS",
    "device_hostile_codec",
    "split_codecs",
]

#: Parquet compression codecs the device reader undoes on the device, lowercased as pyarrow
#: reports them. `"none"` is how pyarrow spells an uncompressed chunk.
#:
#: Deliberately short. Adding a codec here without a device kernel behind it does not produce a
#: slower read that reports itself — it produces a slower read that reports success, which is
#: the failure this whole module exists to name.
DEVICE_CODECS: frozenset[str] = frozenset({"none", "uncompressed", "snappy", "zstd"})


def split_codecs(path: str, row_groups: tuple[int, ...] | None = None) -> frozenset[str]:
    """The compression codecs used by one Parquet file, lowercased.

    Reads the footer through the shared identity-keyed cache, so a caller that already split
    the file by row group pays nothing, and a caller on an object store gets the filesystem
    resolution rather than a bare-path open that would fail on every URI scheme.

    A file whose row groups use different codecs — which a multi-writer table routinely
    produces — reports all of them, because the read is only as device-native as its worst
    chunk.

    Args:
        path: The file's path or URI.
        row_groups: Row-group indices to inspect, or `None` for every one of them.

    Returns:
        The codec names, or an empty set when the footer could not be read. Empty means
        *unknown*, never "uncompressed", and `device_hostile_codec` is written so the two
        cannot be confused.
    """
    try:
        from batcher.io.splits.parquet import _parquet_footer

        metadata = _parquet_footer(path)
    except Exception:
        return frozenset()
    if metadata is None:
        return frozenset()
    wanted = range(metadata.num_row_groups) if row_groups is None else row_groups
    found: set[str] = set()
    try:
        for index in wanted:
            group = metadata.row_group(index)
            for column in range(group.num_columns):
                found.add(str(group.column(column).compression or "").lower())
    except Exception:
        # A malformed footer, or a row-group index past the end of this file, is unknown
        # rather than uncompressed, and the empty set is what carries that upward.
        return frozenset()
    return frozenset(found)


def device_hostile_codec(path: str, row_groups: tuple[int, ...] | None = None) -> bool:
    """Whether the file is known to use compression the device would send back to the host.

    The veto a device Parquet read should honor. True only on evidence: the footer was read
    *and* it named a codec outside `DEVICE_CODECS`. An unreadable footer reports False, which
    leaves the read where it was rather than disabling an optimization on a guess.

    That asymmetry is deliberate and is the opposite of the type check's. A wrong answer here
    costs a slower read; a wrong answer there costs wrong rows.

    Args:
        path: The file's path or URI.
        row_groups: Row-group indices to inspect, or `None` for every one of them.

    Returns:
        True when a disqualifying codec was positively found, False otherwise — including when
        nothing could be read.
    """
    codecs = split_codecs(path, row_groups)
    return bool(codecs) and not codecs <= DEVICE_CODECS
