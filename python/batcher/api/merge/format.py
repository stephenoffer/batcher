"""Which format is the table at this path? — the one question a merge can answer by looking.

`io.detect.detect_format` reads the *path*: an extension, or a ``delta://``-style scheme.
That is the right rule for a **new** path, and it fails on the one layout a warehouse table
almost always has — a directory, which carries no extension of its own.

A merge is where that is fixable, because unlike a fresh read or write it always has an
existing target in front of it. So when the path says nothing, list it and take the format
from the data files inside, rather than making the layout every warehouse uses the one shape
that has to spell ``format=`` out loud.
"""

from __future__ import annotations

__all__ = ["target_format"]


def target_format(target: str, explicit: str | None = None) -> str:
    """The format of the table at `target` — inferred from its data files if need be.

    `detect_format` reads the path: an extension, or a ``delta://``-style scheme. That is
    the right rule for a *new* path, but it fails on the one layout a warehouse table
    almost always has — a **directory**, which carries no extension of its own.

    A merge is the case where that is fixable, because unlike a fresh read or write it
    always has an existing target to look at. So when the path says nothing, list it and
    take the format from the data files inside. One listing, one pass (`expand` accepts the
    whole suffix tuple), and only on the path where detection already failed.

    Args:
        target: Path/URI of the table being merged into.
        explicit: A caller-supplied format, which always wins.

    Returns:
        The format name.

    Raises:
        FormatError: If the format cannot be inferred from the path or its contents, or if
            `target` is a database connection URI rather than a table in storage.
    """
    import posixpath

    from batcher._internal.errors import FormatError
    from batcher.io.detect import DATA_SUFFIXES, detect_format, format_for_extension
    from batcher.io.filesystem import resolve_filesystem

    _refuse_a_connection_uri(target)
    try:
        return detect_format(target, explicit)
    except FormatError:
        pass  # nothing at the target to sniff yet -> fall through to the extension probe

    fs = resolve_filesystem(target)
    try:
        files = fs.expand(target, suffix=DATA_SUFFIXES)
    except (OSError, ValueError):
        files = []
    for path in files:
        fmt = format_for_extension(posixpath.splitext(path)[1])
        if fmt is not None:
            return fmt
    raise FormatError(
        f"could not infer a format for the merge target {target!r} — it has no extension "
        "and no recognizable data files. Pass format=... (e.g. format='parquet')."
    )


def _refuse_a_connection_uri(target: str) -> None:
    """Point a database `MERGE` at the call that performs one, instead of at a filesystem.

    A merge target is a table in storage, and `resolve_filesystem` says so — but it says it
    as ``unsupported storage scheme postgresql://: Protocol not known``, which reads as
    "Batcher cannot reach PostgreSQL". It can, and the merge a user wants there has a name:
    ``ds.write.sql(table, uri=..., mode="upsert", key_columns=...)``, which is a real
    ``MERGE``/``ON CONFLICT`` executed by the database inside one transaction rather than a
    copy-on-write rewrite of data files.

    Raises:
        FormatError: If `target` names a database Batcher can route a connection URI to.
    """
    from batcher._internal.errors import FormatError

    scheme, separator, _rest = target.partition("://")
    if not separator:
        return
    from batcher.io.formats.sql.uri import known_schemes

    if scheme.split("+")[0].lower() not in known_schemes():
        return
    raise FormatError(
        f"{target!r} is a database connection URI, not a table in storage, so there are "
        "no data files to merge into. A database performs the merge itself: "
        "ds.write.sql(table, uri=..., mode='upsert', key_columns=[...]), which runs one "
        "ON CONFLICT / ON DUPLICATE KEY / MERGE statement inside a transaction. "
        "ds.write.merge is the copy-on-write merge for a lakehouse or file table."
    )
