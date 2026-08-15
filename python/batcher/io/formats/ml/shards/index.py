"""The shard manifest: what a corpus contains, and where — without holding a list of it.

`ShardIndex` answers "which shard holds global row *i*" and "where is shard *k*". At the
scale this format exists for, *how* it answers matters as much as what it answers: a
petabyte corpus in 256 MB shards is four million shards, and a manifest that names each one
is a 170 MB JSON document parsed on every rank at startup and four million Python strings
resident for the life of the job.

So a corpus written by `write_shards` is **uniform** — every shard but the last holds exactly
`rows_per_shard` rows, under a generated name — and a uniform index stores none of that per
shard. It stores a count. Locating a row is then integer division rather than a binary search
over a materialized prefix sum, and a four-million-shard corpus costs exactly what a
four-shard one does.

An index that names its shards explicitly is still read (that is what older corpora look
like, and what a hand-assembled one may look like), and those keep the per-shard tuples and
the bisect.
"""

from __future__ import annotations

import base64
import json
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from batcher.io.filesystem import resolve_filesystem

__all__ = ["INDEX_NAME", "ShardIndex", "read_shard_index", "shard_name", "write_index"]

INDEX_NAME = "index.json"

#: Width of the shard number in a generated file name. Eight digits, not five, so the names
#: sort **lexicographically into numeric order** up to a hundred million shards. At five they
#: stopped agreeing at 100,000: ``shard-100000.arrow`` sorts before ``shard-99999.arrow``, so
#: the documented fallback of reading a corpus with ``bt.read.arrow(dir + "/*.arrow")`` —
#: which orders by name — silently returned the rows in the wrong order, on exactly the
#: corpora large enough that nobody would notice by looking.
_NAME_DIGITS = 8


def shard_name(shard: int) -> str:
    """The file name of shard `shard` in a corpus written by `write_shards`.

    Args:
        shard: The shard's index in the corpus.

    Returns:
        The file name, zero-padded so names sort into numeric order.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.ml.shards import shard_name
            >>> shard_name(7)
            'shard-00000007.arrow'
    """
    return f"shard-{shard:0{_NAME_DIGITS}d}.arrow"


def encode_schema(schema: pa.Schema) -> str:
    """The schema as base64 Arrow IPC, so the index carries it without a shard read.

    `Schema.to_string` is for humans and cannot be parsed back, so an empty gather (and
    `ShardReader.schema`) had to open a data file to answer a metadata question — and on a
    directory with zero shards it could not answer at all. The IPC encoding round-trips
    exactly, including extension types and field metadata, which a field-list encoding
    would quietly drop from a tensor column.
    """
    return base64.b64encode(schema.serialize().to_pybytes()).decode("ascii")


def decode_schema(encoded: str) -> pa.Schema:
    """Invert `encode_schema`."""
    return pa.ipc.read_schema(pa.BufferReader(base64.b64decode(encoded)))


@dataclass(frozen=True, slots=True)
class ShardIndex:
    """Where every row of a shard corpus lives, in O(1) space for a uniform corpus.

    `explicit_paths`/`explicit_rows` are empty for a corpus this package wrote: its shards
    are `rows_per_shard` rows each under `shard_name` names, so the count is the whole
    manifest. They are populated only when reading an index that names its shards.
    """

    rows_per_shard: int
    total_rows: int
    shard_count: int
    directory: str = ""
    schema: pa.Schema | None = None
    explicit_paths: tuple[str, ...] = field(default_factory=tuple)
    explicit_rows: tuple[int, ...] = field(default_factory=tuple)

    @property
    def uniform(self) -> bool:
        """Whether the shards are generated-and-equal, so nothing is stored per shard."""
        return not self.explicit_paths

    def shard_path(self, shard: int) -> str:
        """The full path of one shard — O(1), and building no list of the others."""
        name = shard_name(shard) if self.uniform else self.explicit_paths[shard]
        return f"{self.directory}/{name}" if self.directory else name

    def rows_in(self, shard: int) -> int:
        """How many rows shard `shard` holds — O(1)."""
        if not self.uniform:
            return self.explicit_rows[shard]
        start = shard * self.rows_per_shard
        return max(0, min(self.rows_per_shard, self.total_rows - start))

    def start_of(self, shard: int) -> int:
        """The global row index shard `shard` begins at — O(1) for a uniform corpus."""
        if self.uniform:
            return shard * self.rows_per_shard
        return sum(self.explicit_rows[:shard])

    def locate(self, global_index: int) -> tuple[int, int]:
        """Map a global row index to ``(shard_idx, local_idx)``.

        Args:
            global_index: A row index into the whole corpus.

        Returns:
            The shard holding it and the row's offset inside that shard.

        Raises:
            IndexError: If `global_index` is outside ``[0, total_rows)``.
        """
        if not 0 <= global_index < self.total_rows:
            raise IndexError(f"row {global_index} out of range [0, {self.total_rows})")
        if self.uniform:
            return divmod(global_index, self.rows_per_shard)
        shard = bisect_right(self.starts, global_index) - 1
        return shard, global_index - self.starts[shard]

    @property
    def shard_paths(self) -> tuple[str, ...]:
        """Every shard's path, materialized.

        O(`shard_count`) in time and memory, so it is for inspection and for a caller that
        genuinely wants the whole list. The read path uses `shard_path`, which is O(1).
        """
        return tuple(self.shard_path(i) for i in range(self.shard_count))

    @property
    def shard_rows(self) -> tuple[int, ...]:
        """Every shard's row count, materialized (see `shard_paths` on the cost)."""
        return self.explicit_rows or tuple(self.rows_in(i) for i in range(self.shard_count))

    @property
    def starts(self) -> tuple[int, ...]:
        """Each shard's global start offset, materialized (see `shard_paths` on the cost)."""
        offsets: list[int] = []
        running = 0
        for rows in self.shard_rows:
            offsets.append(running)
            running += rows
        return tuple(offsets)

    def to_document(self) -> dict[str, Any]:
        """The index as the JSON document written to ``index.json``."""
        doc: dict[str, Any] = {
            "rows_per_shard": self.rows_per_shard,
            "total_rows": self.total_rows,
            "shard_count": self.shard_count,
            "uniform": self.uniform,
        }
        if self.schema is not None:
            doc["schema"] = self.schema.to_string()  # human-readable, kept for inspection
            doc["schema_ipc"] = encode_schema(self.schema)
        if not self.uniform:
            doc["shards"] = [
                {"path": p, "rows": r}
                for p, r in zip(self.explicit_paths, self.explicit_rows, strict=True)
            ]
        return doc


def build_index(directory: str, doc: dict[str, Any]) -> ShardIndex:
    """A `ShardIndex` from a manifest document, in either representation."""
    encoded = doc.get("schema_ipc")
    schema = decode_schema(encoded) if encoded else None
    shards = doc.get("shards")
    if shards is None:  # uniform: the count is the whole manifest
        return ShardIndex(
            rows_per_shard=int(doc["rows_per_shard"]),
            total_rows=int(doc["total_rows"]),
            shard_count=int(doc["shard_count"]),
            directory=directory,
            schema=schema,
        )
    rows = tuple(int(s["rows"]) for s in shards)
    return ShardIndex(
        rows_per_shard=int(doc["rows_per_shard"]),
        total_rows=int(doc["total_rows"]),
        shard_count=len(rows),
        directory=directory,
        schema=schema,
        explicit_paths=tuple(str(s["path"]) for s in shards),
        explicit_rows=rows,
    )


def write_index(directory: str, index: ShardIndex, *, filesystem: Any = None) -> None:
    """Publish `index` as ``index.json`` under `directory`, atomically."""
    fs = resolve_filesystem(directory) if filesystem is None else filesystem
    with fs.atomic_writer(f"{directory}/{INDEX_NAME}") as fh:
        fh.write(json.dumps(index.to_document(), indent=2).encode())


def read_shard_index(directory: str) -> ShardIndex:
    """Load the `ShardIndex` for a shard directory (reads only ``index.json``).

    Args:
        directory: The shard directory to describe.

    Returns:
        The directory's `ShardIndex`.

    Raises:
        FormatError: If `directory` holds no ``index.json``, or holds one that is not a
            shard manifest.
    """
    from batcher._internal.errors import FormatError

    fs = resolve_filesystem(directory)
    path = f"{directory}/{INDEX_NAME}"
    try:
        with fs.open(path) as fh:
            doc = json.loads(fh.read())
    except FileNotFoundError as exc:
        # Reached by anyone who points a loader at the directory *above* the corpus, or at
        # one that was never written. A bare `FileNotFoundError` naming `index.json` says
        # nothing about what wrote it or how to produce one.
        raise FormatError(
            f"{directory!r} is not a training-shard corpus: no {INDEX_NAME} there. "
            "Write one with `Dataset.ml.write_shards(directory)`."
        ) from exc
    except json.JSONDecodeError as exc:
        raise FormatError(f"{path!r} is not readable as a shard manifest: {exc}") from exc
    if not isinstance(doc, dict) or "total_rows" not in doc or "rows_per_shard" not in doc:
        raise FormatError(
            f"{path!r} is a JSON document but not a shard manifest. "
            "Write the corpus with `Dataset.ml.write_shards(directory)`."
        )
    return build_index(directory, doc)
