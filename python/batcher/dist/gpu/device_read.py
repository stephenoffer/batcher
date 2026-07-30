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
    try:
        frame = _read_parquet(specs, projection)
    except Exception as exc:
        # A device read that fails is a slow shard, never a failed query: the host reader is
        # still there and still correct.
        note_suppressed("dist", "read a gpu shard on the device", exc)
        return None
    return frame if _schema_agrees(frame, descriptor, projection) else None


def _specs(descriptor: dict):
    """The device locators for this descriptor, or `None` when it must go through the host."""
    from batcher.io.splits.device import device_read_specs

    splits = descriptor.get("splits")
    if not splits or descriptor.get("predicate") is not None:
        return None
    return device_read_specs(splits, descriptor.get("projection"))


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

