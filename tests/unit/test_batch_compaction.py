"""Compacting Arrow into one batch is done once, and it does not lose rows doing it.

`Table.combine_chunks().to_batches()[0]` reads as "the table as a single batch" and is not.
`to_batches` splits at Arrow's 32-bit offset limit, so a column holding more than 2 GiB of
`string` or `binary` data comes back as several batches and taking the first drops every row
after it — with no exception, no warning, and a result that is simply short.

The payloads that reach 2 GiB in one batch are the ones Batcher is built to carry: a protobuf
blob column, a decoded image or audio column, an embedding, a large text field. Six call sites
had worked this out independently and written a fix; three others still had the bug, in the
streaming distinct, in `iter_batches`' exact re-chunker, and in the keyed-state fold.

So there are two tests here and they guard different halves. The first pins what `one_batch`
does. The second is the one that keeps the bug from regrowing: it reads the source and fails
on the spelling itself, because a *new* call site is how this came back the last three times,
and no behavioural test can see a call site that does not exist yet.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from batcher.plan.types import one_batch

pytestmark = pytest.mark.unit

_PACKAGE = Path(__file__).resolve().parents[2] / "python" / "batcher"

#: The spelling that silently truncates. `to_batches()` used in full is fine, so the pattern
#: is deliberately anchored on the subscript.
_LOSSY = "combine_chunks().to_batches()[0]"


class TestOneBatch:
    def test_a_multi_chunk_table_keeps_every_row(self) -> None:
        table = pa.Table.from_batches(
            [pa.record_batch({"a": pa.array([i, i + 1])}) for i in range(0, 10, 2)]
        )
        compacted = one_batch(table)
        assert compacted.num_rows == table.num_rows
        assert compacted.column("a").to_pylist() == table.column("a").to_pylist()

    def test_a_sequence_of_batches_is_concatenated(self) -> None:
        batches = [pa.record_batch({"a": pa.array([1])}), pa.record_batch({"a": pa.array([2, 3])})]
        assert one_batch(batches).column("a").to_pylist() == [1, 2, 3]

    def test_a_lone_batch_is_returned_unchanged(self) -> None:
        """No copy for the overwhelmingly common case."""
        batch = pa.record_batch({"a": pa.array([1, 2])})
        assert one_batch(batch) is batch
        assert one_batch([batch]) is batch

    def test_an_empty_table_keeps_its_schema(self) -> None:
        """A caller handed `None` has to invent a schema to type whatever comes next."""
        schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
        compacted = one_batch(pa.Table.from_pylist([], schema=schema))
        assert compacted is not None
        assert compacted.num_rows == 0
        assert compacted.schema == schema

    def test_an_empty_sequence_is_none(self) -> None:
        """A sequence carries no schema, so there is nothing to return."""
        assert one_batch([]) is None


def _code_only(path: Path) -> str:
    """`path`'s source with comments and string literals removed.

    The spelling appears legitimately in prose: several modules explain in a comment why they
    do *not* use it. Tokenizing is what separates an explanation from a call.
    """
    import io
    import tokenize

    kept: list[str] = []
    with path.open("rb") as handle:
        for token in tokenize.tokenize(io.BytesIO(handle.read()).readline):
            if token.type not in (tokenize.COMMENT, tokenize.STRING):
                kept.append(token.string)
    return "".join(kept)


def test_no_call_site_uses_the_truncating_spelling() -> None:
    """The regression guard. A new *call* writing `...to_batches()[0]` fails the build.

    This is the half that matters. A behavioural test cannot see a call site that does not
    exist yet, and a new call site is how this came back the last three times.
    """
    offenders = [
        path.relative_to(_PACKAGE.parent)
        for path in _PACKAGE.rglob("*.py")
        if _LOSSY.replace(" ", "") in _code_only(path).replace(" ", "")
    ]
    assert not offenders, (
        f"{len(offenders)} module(s) take the first batch of a split table and drop the rest: "
        f"{[str(p) for p in offenders]}. Use `plan.types.one_batch`, which returns one batch "
        "or raises, never a prefix."
    )
