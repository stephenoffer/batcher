"""`StorageLevel`: the vocabulary, its coercion, and what it refuses.

The parse is a public API edge — a user types a Spark name from memory — so what matters
here is that a wrong name produces a message naming the right ones rather than an
`AttributeError` from somewhere inside the cache.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import PlanError
from batcher.plan.resource import StorageLevel

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("memory_only", StorageLevel.MEMORY_ONLY),
        ("MEMORY_ONLY", StorageLevel.MEMORY_ONLY),
        ("Memory Only", StorageLevel.MEMORY_ONLY),
        ("memory-and-disk", StorageLevel.MEMORY_AND_DISK),
        ("  DISK_ONLY  ", StorageLevel.DISK_ONLY),
        (StorageLevel.DISK_ONLY, StorageLevel.DISK_ONLY),
    ],
)
def test_parse_accepts_every_reasonable_spelling(spelling, expected):
    assert StorageLevel.parse(spelling) is expected


def test_parse_defaults_to_memory_and_disk():
    # The default matters: a cache whose only answer to a full budget is to forget cannot
    # help a working set larger than RAM, which is the working set that asked for a cache.
    assert StorageLevel.parse(None) is StorageLevel.MEMORY_AND_DISK


@pytest.mark.parametrize("bad", ["MEMORY_ONLY_SER", "MEMORY_AND_DISK_2", "OFF_HEAP", "", 3])
def test_parse_refuses_an_unknown_level_and_names_the_real_ones(bad):
    with pytest.raises(PlanError) as excinfo:
        StorageLevel.parse(bad)
    message = str(excinfo.value)
    assert "unknown storage level" in message
    # The Spark spellings with no counterpart here are the mistake this actually catches,
    # so the message has to carry the alternatives rather than only rejecting the input.
    for name in ("memory_only", "memory_and_disk", "disk_only"):
        assert name in message


@pytest.mark.parametrize(
    ("level", "memory", "disk"),
    [
        (StorageLevel.MEMORY_ONLY, True, False),
        (StorageLevel.MEMORY_AND_DISK, True, True),
        (StorageLevel.DISK_ONLY, False, True),
    ],
)
def test_the_media_a_level_allows(level, memory, disk):
    assert level.uses_memory is memory
    assert level.uses_disk is disk


def test_every_level_allows_at_least_one_medium():
    # A level that permits neither would silently make `cache()` a no-op.
    assert all(level.uses_memory or level.uses_disk for level in StorageLevel)
