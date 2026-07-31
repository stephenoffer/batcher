"""Reading and decoding on the device, and the two ways that silently does not happen.

Both paths here exist to save the same thing — bytes across PCIe and host cores that are
feeding seven other devices — and both have a fallback that engages without raising:

* A Parquet codec the device has no kernel for is decompressed on the *host*, so the pages
  cross the bus twice and the CPU does the work anyway. Strictly worse than the host reader,
  and identical to a successful device read in every log line.
* KvikIO's compat mode reads through a host bounce buffer with the GPUDirect API in front of
  it. Also strictly worse than the plain host read, also silent.

Both are therefore checked before the fast path is chosen. What each check does with an
*unknown* answer is the interesting part, and the two go opposite ways on purpose:

* KvikIO defaults to compat. Being wrong there costs the host read, which works.
* A codec nobody could read defaults to allowed. Vetoing on doubt would disable device reads
  for every corpus whose footer was momentarily unavailable, which is a far larger loss than
  the occasional slow read it would prevent.

The asymmetry is not an inconsistency. The first is a claim about what a library will do and is
knowable up front; the second is a claim about a file, and files are unreadable for reasons that
have nothing to do with their compression. Neither check protects correctness — the type gate in
`device` does that, and it refuses whatever it cannot prove.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.io.splits import codecs, device, kvikio
from batcher.ml.decode import accelerated

pytestmark = pytest.mark.unit


@pytest.fixture
def parquet_file(tmp_path):
    """A writer for one small Parquet file at a chosen compression."""

    def _write(compression: str, name: str = "t") -> str:
        path = str(tmp_path / f"{name}.parquet")
        pq.write_table(
            pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}), path, compression=compression
        )
        return path

    return _write


# --- compression codecs -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("compression", "hostile"),
    [("snappy", False), ("zstd", False), ("none", False), ("gzip", True), ("brotli", True)],
)
def test_only_a_codec_without_a_device_kernel_vetoes_the_read(parquet_file, compression, hostile):
    assert codecs.device_hostile_codec(parquet_file(compression)) is hostile


def test_an_unreadable_footer_is_unknown_and_does_not_veto(tmp_path):
    missing = str(tmp_path / "absent.parquet")
    assert codecs.split_codecs(missing) == frozenset()
    # Unknown leaves the read where it was. This gate protects throughput rather than
    # correctness, so vetoing on doubt would disable device reads for every corpus whose
    # metadata was momentarily unavailable — which is the whole of an object store on a
    # transient error.
    assert codecs.device_hostile_codec(missing) is False


def test_a_truncated_footer_is_unknown_and_does_not_veto(tmp_path, parquet_file):
    path = parquet_file("snappy")
    truncated = str(tmp_path / "truncated.parquet")
    with open(path, "rb") as src, open(truncated, "wb") as dst:
        dst.write(src.read()[:64])
    assert codecs.split_codecs(truncated) == frozenset()
    assert codecs.device_hostile_codec(truncated) is False


def test_the_codecs_of_every_row_group_are_collected(tmp_path):
    # The read is only as device-native as its worst chunk, so the answer is the union across
    # the groups that will actually be read rather than the first one's codec.
    path = str(tmp_path / "many.parquet")
    table = pa.table({"a": [1, 2, 3]})
    writer = pq.ParquetWriter(path, table.schema, compression="snappy")
    for _ in range(3):
        writer.write_table(table)
    writer.close()
    assert pq.ParquetFile(path).metadata.num_row_groups == 3
    assert codecs.split_codecs(path) == frozenset({"snappy"})
    assert codecs.device_hostile_codec(path) is False


def test_one_codec_outside_the_allowlist_rejects_the_whole_read(tmp_path):
    # All-or-nothing: a descriptor read half on the device and half on the host concatenates
    # two readers' output, which is the thing the device path cannot promise to be identical.
    path = str(tmp_path / "gz.parquet")
    pq.write_table(pa.table({"a": [1, 2, 3]}), path, compression="gzip")
    assert codecs.split_codecs(path) == frozenset({"gzip"})
    assert not codecs.split_codecs(path) <= codecs.DEVICE_CODECS
    assert codecs.device_hostile_codec(path) is True


def test_the_allowlist_is_short_on_purpose():
    # An unlisted codec is one nobody has checked, not one the device cannot undo. Growing
    # this set without a kernel behind it produces a slower read that reports success.
    assert frozenset({"none", "uncompressed", "snappy", "zstd"}) == codecs.DEVICE_CODECS


def test_row_groups_narrow_the_footer_read(parquet_file):
    path = parquet_file("snappy")
    assert codecs.split_codecs(path, row_groups=(0,)) == frozenset({"snappy"})


# --- the split decision -------------------------------------------------------------------


def test_a_gzip_corpus_keeps_the_host_reader(monkeypatch, parquet_file):
    from batcher.io.splits.file import FileSplit

    path = parquet_file("gzip")
    split = FileSplit(path=path, format_name="parquet")
    monkeypatch.setattr(
        FileSplit, "schema", lambda self: pa.schema([("a", pa.int64()), ("b", pa.string())])
    )
    assert device.device_read_specs([split], None) is None


def test_a_snappy_corpus_is_read_on_the_device(monkeypatch, parquet_file):
    from batcher.io.splits.file import FileSplit

    path = parquet_file("snappy")
    split = FileSplit(path=path, format_name="parquet")
    monkeypatch.setattr(
        FileSplit, "schema", lambda self: pa.schema([("a", pa.int64()), ("b", pa.string())])
    )
    specs = device.device_read_specs([split], None)
    assert specs is not None
    assert [s.path for s in specs] == [path]


# --- GPUDirect Storage --------------------------------------------------------------------


class _Defaults:
    def __init__(self, compat: bool, threads: int = 8) -> None:
        self._compat = compat
        self._threads = threads

    def is_compat_mode_preferred(self) -> bool:
        return self._compat

    def num_threads(self) -> int:
        return self._threads


class _Kvikio:
    def __init__(self, defaults) -> None:
        self.defaults = defaults


def _install(monkeypatch, module) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "kvikio", module)
    kvikio.reset_kvikio_probe()


def test_an_absent_library_is_not_direct(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "kvikio", None)
    kvikio.reset_kvikio_probe()
    status = kvikio.kvikio_status()
    assert status.available is False
    # Compat is the conservative default in every unavailable case, so a caller that tests
    # one field never concludes it has DMA it does not have.
    assert status.compat_mode is True
    assert status.direct is False
    kvikio.reset_kvikio_probe()


def test_compat_mode_is_reported_as_not_direct(monkeypatch):
    _install(monkeypatch, _Kvikio(_Defaults(compat=True)))
    try:
        status = kvikio.kvikio_status()
        assert status.available is True
        assert status.direct is False
        assert status.reason
    finally:
        kvikio.reset_kvikio_probe()


def test_direct_mode_is_the_only_state_worth_preferring(monkeypatch):
    _install(monkeypatch, _Kvikio(_Defaults(compat=False)))
    try:
        status = kvikio.kvikio_status()
        assert status.direct is True
        assert status.threads == 8
        assert status.reason == ""
    finally:
        kvikio.reset_kvikio_probe()


def test_the_environment_override_is_named_in_the_reason(monkeypatch):
    monkeypatch.setenv("KVIKIO_COMPAT_MODE", "on")
    _install(monkeypatch, _Kvikio(_Defaults(compat=False)))
    try:
        status = kvikio.kvikio_status()
        assert status.direct is False
        # An operator can act on "someone set this" and cannot act on "the library says so".
        assert "KVIKIO_COMPAT_MODE" in status.reason
    finally:
        kvikio.reset_kvikio_probe()


def test_a_library_with_no_compat_predicate_defaults_to_compat(monkeypatch):
    class _Bare:
        pass

    _install(monkeypatch, _Kvikio(_Bare()))
    try:
        assert kvikio.kvikio_status().direct is False
    finally:
        kvikio.reset_kvikio_probe()


def test_a_predicate_that_raises_defaults_to_compat(monkeypatch):
    class _Angry:
        def is_compat_mode_preferred(self):
            raise RuntimeError("cuFile is unhappy")

    _install(monkeypatch, _Kvikio(_Angry()))
    try:
        assert kvikio.kvikio_status().direct is False
    finally:
        kvikio.reset_kvikio_probe()


# --- device decode ------------------------------------------------------------------------


def test_no_device_means_no_backend(monkeypatch):
    monkeypatch.setattr(accelerated, "_cuda_device", lambda: "")
    accelerated.reset_decode_backend_probe()
    try:
        assert bool(accelerated.image_decode_backend()) is False
        assert bool(accelerated.video_decode_backend()) is False
        # And the decode declines rather than raising, so a caller keeps the host path.
        assert accelerated.decode_jpeg_batch([b"\xff\xd8"]) is None
    finally:
        accelerated.reset_decode_backend_probe()


def test_a_resolved_backend_needs_both_a_device_and_a_library(monkeypatch):
    monkeypatch.setattr(accelerated, "_cuda_device", lambda: "cuda")
    monkeypatch.setattr(accelerated, "_importable", lambda path: False)
    accelerated.reset_decode_backend_probe()
    try:
        assert accelerated.image_decode_backend().available is False
    finally:
        accelerated.reset_decode_backend_probe()


def test_a_resolved_image_backend_decodes_batched(monkeypatch):
    monkeypatch.setattr(accelerated, "_cuda_device", lambda: "cuda")
    monkeypatch.setattr(accelerated, "_importable", lambda path: True)
    accelerated.reset_decode_backend_probe()
    try:
        backend = accelerated.image_decode_backend()
        assert backend.name == "torchvision"
        # A per-image call into a device decoder loses to the host decoder it replaced, so
        # batched is a property of the plan rather than a performance footnote.
        assert backend.batched is True
    finally:
        accelerated.reset_decode_backend_probe()


def test_an_empty_batch_decodes_to_an_empty_batch(monkeypatch):
    monkeypatch.setattr(accelerated, "_cuda_device", lambda: "cuda")
    monkeypatch.setattr(accelerated, "_importable", lambda path: True)
    accelerated.reset_decode_backend_probe()
    try:
        # Distinct from `None`, which means "use the host path".
        assert accelerated.decode_jpeg_batch([None, b""]) == []
    finally:
        accelerated.reset_decode_backend_probe()


def test_a_failing_decoder_returns_none_rather_than_a_partial_batch(monkeypatch):
    monkeypatch.setattr(accelerated, "_cuda_device", lambda: "cuda")
    monkeypatch.setattr(accelerated, "_importable", lambda path: True)
    accelerated.reset_decode_backend_probe()
    try:
        # A batch half on the device and half on the host is worse than either, so a build of
        # torchvision without nvJPEG sends the whole batch back to the host path.
        assert accelerated.decode_jpeg_batch([b"not a jpeg"]) is None
    finally:
        accelerated.reset_decode_backend_probe()


def test_the_saving_is_computed_from_geometry_rather_than_asserted():
    # A 1440x1440 RGB frame is 6.2 MB; a 500 KB JPEG of it is twelve times smaller, and that
    # ratio is what the bus carries on the host path.
    assert accelerated.transfer_saving_ratio(1440, 1440, 500_000) == pytest.approx(12.4, abs=0.1)
    # Missing geometry reports nothing rather than a conventional guess.
    assert accelerated.transfer_saving_ratio(0, 0, 0) == 0.0
    assert accelerated.transfer_saving_ratio(1440, 1440, 0) == 0.0


def test_hardware_decode_is_unknown_where_the_counters_are_absent(monkeypatch):
    monkeypatch.setattr("batcher._internal.hardware.telemetry.engines.device_engines", lambda: ())
    # `None`, not `False`: a part that publishes no engine counters is not evidence the
    # decode landed on the SMs.
    assert accelerated.hardware_decode_confirmed() is None


def test_hardware_decode_is_false_when_the_counters_are_readable_and_idle(monkeypatch):
    from batcher._internal.hardware.telemetry.engines import EngineUtilization

    monkeypatch.setattr(
        "batcher._internal.hardware.telemetry.engines.device_engines",
        lambda: (EngineUtilization(index=0, supported=("NVDEC",), readable=True),),
    )
    assert accelerated.hardware_decode_confirmed() is False


def test_hardware_decode_is_confirmed_from_the_engine_counters(monkeypatch):
    from batcher._internal.hardware.telemetry.engines import EngineUtilization

    monkeypatch.setattr(
        "batcher._internal.hardware.telemetry.engines.device_engines",
        lambda: (EngineUtilization(index=0, jpeg=0.7, supported=("NVJPG",), readable=True),),
    )
    # Asking for a device decode and getting one are different events, and this is the only
    # source that can tell them apart.
    assert accelerated.hardware_decode_confirmed() is True
