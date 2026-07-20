"""The spill store must react to a disk filled by someone *other* than itself.

The local budget accounts only for the bytes this store wrote, and it is clamped once, at
construction, against a single free-space sample. Neither can see a co-tenant process,
another query's scratch, or a log filling the same volume *during* the query — so the
local tier kept writing until the filesystem returned ENOSPC, which surfaced as a bare
`OSError: [Errno 28]` from inside the Arrow writer, naming neither the spill tier nor the
way out.

Two behaviors close that. A bucket opened while the volume is genuinely low routes to the
remote tier even though the budget says there is room (the tier is fixed at open, so this
is the last moment the choice is free). And an ENOSPC that still happens becomes a typed
`ResourceError` carrying the three actual remedies.
"""

from __future__ import annotations

import errno

import pyarrow as pa
import pytest

from batcher._internal.errors import ResourceError
from batcher.carbonite import spill as spill_mod
from batcher.carbonite.spill import SpillTier, TieredSpillStore

pytestmark = pytest.mark.unit


def _batch(n: int = 4) -> pa.RecordBatch:
    return pa.record_batch({"v": pa.array(list(range(n)), type=pa.int64())})


def test_a_low_volume_routes_a_new_bucket_to_remote(tmp_path, monkeypatch) -> None:
    """The regression: budget says there is room, the actual disk says there is not."""
    store = TieredSpillStore(
        str(tmp_path / "scratch"),
        remote_uri=f"memory://{tmp_path.name}",
        local_budget_bytes=1 << 40,  # enormous — the budget alone would stay LOCAL
    )
    monkeypatch.setattr(spill_mod, "_free_disk_bytes", lambda _p: 1 << 20)  # 1 MiB left

    writer = store.writer("b0")
    writer.write(_batch())
    handle = writer.close()

    assert handle is not None
    assert handle.tier is SpillTier.REMOTE


def test_an_ample_volume_still_uses_the_local_tier(tmp_path, monkeypatch) -> None:
    """The fast path must be untouched — this only fires when the disk is really low."""
    store = TieredSpillStore(
        str(tmp_path / "scratch"),
        remote_uri=f"memory://{tmp_path.name}",
        local_budget_bytes=1 << 40,
    )
    monkeypatch.setattr(spill_mod, "_free_disk_bytes", lambda _p: 1 << 40)

    writer = store.writer("b0")
    writer.write(_batch())
    handle = writer.close()

    assert handle is not None
    assert handle.tier is SpillTier.LOCAL


def test_a_low_volume_with_no_remote_tier_stays_local(tmp_path, monkeypatch) -> None:
    """With nowhere to overflow to, the write must still be attempted, not pre-emptively failed.

    There is no remote URI configured, so routing away is not an option; the store has to
    try the local write and let `ENOSPC` (if it comes) produce the actionable error.
    """
    store = TieredSpillStore(str(tmp_path / "scratch"), local_budget_bytes=1 << 40)
    monkeypatch.setattr(spill_mod, "_free_disk_bytes", lambda _p: 1 << 10)

    writer = store.writer("b0")
    writer.write(_batch())
    handle = writer.close()

    assert handle is not None
    assert handle.tier is SpillTier.LOCAL


def test_enospc_becomes_an_actionable_resource_error(tmp_path) -> None:
    """A full disk must name the spill tier and the way out, not leak `[Errno 28]`."""
    store = TieredSpillStore(str(tmp_path / "scratch"))
    writer = store.writer("b0")
    writer.write(_batch())  # opens the LOCAL writer

    def _full(_batch_arg):
        raise OSError(errno.ENOSPC, "No space left on device")

    writer._writer.write_batch = _full

    with pytest.raises(ResourceError, match="spill disk is full"):
        writer.write(_batch())


def test_a_non_enospc_oserror_is_not_relabelled(tmp_path) -> None:
    """Only a full disk gets the disk-full message; other IO errors must stay themselves."""
    store = TieredSpillStore(str(tmp_path / "scratch"))
    writer = store.writer("b0")
    writer.write(_batch())

    def _denied(_batch_arg):
        raise OSError(errno.EACCES, "Permission denied")

    writer._writer.write_batch = _denied

    with pytest.raises(OSError) as excinfo:
        writer.write(_batch())
    assert not isinstance(excinfo.value, ResourceError)
    assert excinfo.value.errno == errno.EACCES
