"""Published shuffle output, charged to the same envelope everything else reserves from.

A mapper's output bucket used to be invisible to Carbonite. Nobody reserves it: the mapper
produces it, hands it to the local Flight store, and it stays resident until a reducer
collects it. The buffer pool's `used` read zero while the process held the node's whole
share of the shuffle, which is why `PressureMonitor` falls back to process RSS there.

Now the store holds a pool reservation equal to what it retains, and yields it back by
spilling to local disk when the pool cannot cover a growth. Two consequences are worth
pinning, and they are the two halves of the same design:

- **It bounds.** Publishing more than the envelope leaves less than the envelope resident.
- **It costs nothing but a re-read.** Every bucket comes back byte-identical, so this is a
  memory strategy and not a semantics.

Each test runs in a fresh interpreter. The pool is process-global and its limit only ever
grows, so a test sharing a process with an earlier large-envelope query would silently
measure that query's envelope instead of its own — the test would still pass and would be
asserting nothing.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.integration

# Enough buckets that spilling has something to choose between, and enough bytes each that
# the footprint is well clear of allocator noise.
_PROLOGUE = """
import pyarrow as pa
import batcher as bt
from batcher._internal.native import engine

nat = engine()
nat.set_flight_transport_config(0, 0, 0, None, 0)  # no store cap: pool pressure only

ENVELOPE = {envelope}
BUCKETS = 12
ROWS = 100_000

# Size the process pool by running one trivial query at the envelope under test.
with bt.config_context(bt.Config().replace(memory=bt.MemoryConfig(max_memory_bytes=ENVELOPE))):
    bt.from_pydict({{"a": [1]}}).collect()

server = nat.FlightShuffleServer()

def bucket(i):
    return pa.record_batch({{"k": pa.array([i] * ROWS), "v": pa.array(range(ROWS))}})

published = 0
for i in range(BUCKETS):
    b = bucket(i)
    published += b.get_total_buffer_size()
    server.publish("91/0/%d/0/0" % i, [b])
"""


def _run(envelope: int, body: str) -> str:
    """Run `body` after the prologue in a fresh interpreter, returning its stdout."""
    script = textwrap.dedent(_PROLOGUE.format(envelope=envelope)) + textwrap.dedent(body)
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, f"child failed:\n{done.stderr[-3000:]}"
    return done.stdout.strip()


def test_a_tight_envelope_bounds_what_the_shuffle_keeps_resident() -> None:
    """The failure this exists to prevent: a node holding its whole share of the shuffle."""
    out = _run(
        8 << 20,
        """
        print(server.retained_bytes, published)
        """,
    )
    retained, published = (int(x) for x in out.split())
    assert published > 8 << 20, "the fixture did not publish more than the envelope"
    assert retained < published, "nothing was bounded — the store kept everything"
    assert retained <= 8 << 20, f"retained {retained} exceeds the {8 << 20}-byte envelope"


def test_every_bucket_reads_back_identically_after_being_bounded() -> None:
    """A memory strategy that changed an answer would be a correctness bug in disguise."""
    out = _run(
        8 << 20,
        """
        ok = True
        for i in range(BUCKETS):
            got = server.local_fetch("91/0/%d/0/0" % i)
            ok = ok and got is not None and len(got) == 1
            ok = ok and got[0].num_rows == ROWS
            ok = ok and got[0].column(0)[0].as_py() == i
            ok = ok and got[0].column(1).to_pylist() == list(range(ROWS))
        print("OK" if ok else "MISMATCH")
        """,
    )
    assert out == "OK"


def test_a_roomy_envelope_keeps_everything_in_memory() -> None:
    """The bound must not cost the common case a disk round trip it does not need.

    A shuffle that fits should never touch the disk; if it does, every shuffle pays for the
    protection that only the large ones need.
    """
    out = _run(
        2 << 30,
        """
        print(server.retained_bytes, published)
        """,
    )
    retained, published = (int(x) for x in out.split())
    # `>=` rather than `==`: the store measures `get_array_memory_size`, which counts the
    # allocator's buffer padding that the caller-side `get_total_buffer_size` does not. The
    # gap is ~2 KB over 19 MB and is not what this test is about; anything spilled would
    # take the figure *down* by a bucket, which is what the comparison catches.
    assert retained >= published, "a shuffle that fits the envelope was spilled anyway"


def test_clearing_returns_every_byte() -> None:
    """Bounded or not, the store must account its way back to zero.

    A store that leaks accounting holds pool credit no query can use, which starves the
    next shuffle rather than the current one — a failure that shows up nowhere near its
    cause.
    """
    out = _run(
        8 << 20,
        """
        server.clear()
        print(server.retained_bytes)
        """,
    )
    assert int(out) == 0


def test_an_unsized_pool_does_not_spill_anything() -> None:
    """A Flight server built before the first query must not spill on every publish.

    This is the normal startup order for a worker, not an edge case: the process pool has
    no envelope until an `execute_plan` sets one, and charging against an unsized pool
    refuses *everything*. Measured before the guard existed: eight buckets published, none
    retained.
    """
    script = textwrap.dedent(
        """
        import pyarrow as pa
        from batcher._internal.native import engine

        nat = engine()
        nat.set_flight_transport_config(0, 0, 0, None, 0)
        server = nat.FlightShuffleServer()  # no query has run: the pool is unsized
        for i in range(8):
            server.publish(
                "92/0/%d/0/0" % i,
                [pa.record_batch({"v": pa.array(range(100_000))})],
            )
        print(server.retained_bytes)
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, done.stderr[-3000:]
    assert int(done.stdout.strip()) > 0, "an unsized pool spilled the whole shuffle to disk"
