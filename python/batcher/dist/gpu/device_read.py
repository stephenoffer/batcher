"""Read a shard onto the device, instead of onto the host and then across the bus.

A GPU shard's read is the half of the query nobody accelerated. The worker asked pyarrow for
the rows, which decoded Parquet on the CPU into host memory, and only then did the frame cross
PCIe onto the device. For a scan-heavy query — the shape a GPU is worth using for — that decode
is most of the wall clock, and the device spends it idle waiting for a core to hand it
something. cuDF reads Parquet on the device: the compressed bytes cross, and the decode happens
on the thing that was going to compute anyway.

The gain is only real if the two readers produce the same rows, so the conditions are narrow
and every one of them is checked before the read rather than hoped for afterwards:

* the shard's splits must be plain Parquet locators of types both readers agree on, which is
  `io.splits.device`'s question, not this module's;
* **nothing may have been pushed into the read.** A predicate the host reader would have used
  to skip row groups is not carried here, so a device read that ignored it would move far more
  bytes than the read it replaced. A selective query keeps the selective reader;
* the frame that comes back must present the schema the host path would have. It is compared,
  not assumed — a device Parquet reader is a second implementation, and the one failure this
  cannot tolerate is a shard whose schema differs from its neighbours' in a concatenation.

Any of those failing returns `None`, which the caller reads as "use the host reader". That is
the same contract every other fallback in the GPU backend follows, and it is why turning this
on cannot change an answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed

if TYPE_CHECKING:
    from batcher.core.gpu_plan import DfBackend

__all__ = ["read_descriptor_on_device"]


def read_descriptor_on_device(descriptor: dict, be: DfBackend):
    """The shard as a device frame read straight from storage, or `None` to use the host path.

    Args:
        descriptor: A descriptor from `partition_descriptors`.
        be: The dataframe backend that will compute on the result. A host backend declines
            immediately: reading "on the device" through pandas would be a second Parquet
            reader with none of the benefit, and the verification path must exercise the same
            code the engine's own tests cover.

    Returns:
        A device dataframe holding the shard's rows, or `None` when the shard is not
        device-readable, when a predicate was pushed into its read, or when the device reader
        produced a schema the host reader would not have.

    Examples:
        .. doctest::

            >>> import pandas as pd
            >>> from batcher.core.gpu_plan import DfBackend
            >>> from batcher.dist.gpu.device_read import read_descriptor_on_device
            >>> read_descriptor_on_device({"batches": []}, DfBackend(pd)) is None
            True
    """
    if not be.is_gpu:
        return None
    specs = _specs(descriptor)
    if specs is None:
        return None
    projection = descriptor.get("projection")
    _publish_transfer_path(specs)
    try:
        frame = _widen(_read_parquet(specs, projection), descriptor, projection, be)
    except Exception as exc:
        # A device read that fails is a slow shard, never a failed query: the host reader is
        # still there and still correct.
        note_suppressed("dist", "read a gpu shard on the device", exc)
        return None
    return frame if _schema_agrees(frame, descriptor, projection) else None


def _widen(frame, descriptor: dict, projection: list[str] | None, be: DfBackend):
    """Widen the frame's narrow numeric columns, as the host path's `from_arrow` would.

    This reader never goes through `from_arrow`, so it does not get that widening for free —
    and without it a device-read shard contributes an `int32` column beside a host-read one's
    `int64`, which is exactly the concatenation a fan-out then has to make sense of. The source
    schema decides rather than the frame's own dtypes, so both readers reach the same answer
    from the same fact.
    """
    from batcher.core.gpu_plan.backend import widened_type

    schema = descriptor["splits"][0].schema()
    # This reader never goes through `from_arrow`, so the backend would not otherwise learn
    # which columns were calendar days — and a DATE that entered a frame comes back out of
    # `to_arrow` as a timestamp. Registering the source schema here is what keeps a device-read
    # shard's schema equal to a host-read one's.
    be.remember_dates(schema)
    names = list(projection) if projection is not None else list(frame.columns)
    for name in names:
        target = widened_type(schema.field(name).type)
        if target is not None:
            frame[name] = frame[name].astype(be.dtype(target))
    return frame


def _specs(descriptor: dict):
    """The device locators for this descriptor, or `None` when it must go through the host."""
    from batcher.io.splits.device import device_read_specs

    splits = descriptor.get("splits")
    if not splits or descriptor.get("predicate") is not None:
        return None
    return device_read_specs(splits, descriptor.get("projection"))


def _publish_transfer_path(specs: list) -> None:
    """Report whether these files reach the device by DMA or through a host bounce buffer.

    A device-native read is two wins stacked: the decode happens on the device, and — only
    when GPUDirect Storage applies — the bytes travel storage-to-device without the host
    touching them. The second one silently does not happen on a container overlay, on a FUSE
    -mounted object store, and in an image without the cuFile library, and the read then adds
    a host copy to a path whose whole argument was avoiding one. Nothing about the result
    changes either way, which is exactly why it needs to be visible: a scan that is half its
    expected rate looks identical to one that is not.

    Best-effort and skipped entirely when nothing is listening, so an unobserved run pays
    neither the mount-table read nor the library probe.
    """
    from batcher._internal import events

    if not events.listening():
        return
    try:
        from batcher.io.splits.gds import gds_summary

        events.publish(
            events.GPU,
            name="device_read",
            event="transfer_path",
            **gds_summary(tuple(spec.path for spec in specs)),
        )
    except Exception as exc:  # pragma: no cover - observability must never fail a read
        note_suppressed("dist", "report the gpu read's transfer path", exc)


def _read_parquet(specs: list, projection: list[str] | None):
    """Read every locator in one cuDF call, so the files are fetched concurrently."""
    import cudf

    paths = [spec.path for spec in specs]
    kwargs: dict = {}
    if projection is not None:
        kwargs["columns"] = list(projection)
    # cuDF takes row groups as one list per path, and only when every path names some. A
    # whole-file split alongside a row-group one has no such list, so the pair is read whole
    # and the operator chain above sees the same rows either way.
    if all(spec.row_groups is not None for spec in specs):
        kwargs["row_groups"] = [list(spec.row_groups) for spec in specs]
    return cudf.read_parquet(paths, **kwargs)


def _schema_agrees(frame, descriptor: dict, projection: list[str] | None) -> bool:
    """Whether the device read produced the columns, in the order, the host read would have.

    Compared rather than trusted. Two Parquet readers agreeing on a file's *types* is what
    `io.splits.device` gates on; agreeing on its *columns and their order* is separate, because
    a projection is applied by two different mechanisms on the two paths, and a shard whose
    column order differs from its neighbours' silently corrupts the concatenation that follows.
    """
    try:
        expected = _expected_names(descriptor, projection)
        return expected is None or list(frame.columns) == expected
    except Exception as exc:
        note_suppressed("dist", "compare the device read's schema", exc)
        return False


def _expected_names(descriptor: dict, projection: list[str] | None) -> list[str] | None:
    """The column names the host reader would have returned, or `None` when it cannot say."""
    if projection is not None:
        return list(projection)
    splits = descriptor.get("splits")
    return list(splits[0].schema().names) if splits else None
