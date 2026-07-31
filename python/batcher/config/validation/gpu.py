"""Range checks for the GPU packing and merge tunables.

Kept out of `sections.py` because that module is at its size limit, and because these four
knobs share one failure mode that is worth stating once: each of them is *clamped* at the point
of use, so an out-of-range value never crashes and never obviously misbehaves. It quietly
becomes a different number. A `gpu_task_fraction` of `1.5` asks Ray for one and a half devices
per shard and pends the fan-out forever; a `gpu_shard_expansion` of `0.5` claims a shard's
intermediate is smaller than the shard, which packs a device to twice what it can hold and
turns every run into a subdivision storm. Neither reads as a configuration error at the point
it bites, so they are refused at the point they are set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.device_share import MAX_COTENANTS
from batcher._internal.errors import ConfigError

if TYPE_CHECKING:
    from batcher.config.config import DistributedConfig

__all__ = ["check_gpu_packing"]


def check_gpu_packing(d: DistributedConfig) -> None:
    """Raise `ConfigError` when a GPU packing or merge tunable is out of range.

    Args:
        d: The distributed section to check.

    Raises:
        ConfigError: On the first out-of-range value, naming the field and its bound.
    """
    checks: tuple[tuple[bool, str], ...] = (
        (
            0.0 <= d.gpu_task_fraction <= 1.0,
            "distributed.gpu_task_fraction must be in [0, 1] (0 derives it); a shard task runs "
            f"on one device, so it cannot hold more than one, got {d.gpu_task_fraction}",
        ),
        (
            1 <= d.gpu_max_tasks_per_device <= MAX_COTENANTS,
            f"distributed.gpu_max_tasks_per_device must be in [1, {MAX_COTENANTS}] (the "
            f"reciprocal of the smallest packing quantum), got {d.gpu_max_tasks_per_device}",
        ),
        (
            d.gpu_shard_expansion >= 1.0,
            "distributed.gpu_shard_expansion must be >= 1.0: a shard's input is resident "
            f"alongside whatever the operator derives from it, got {d.gpu_shard_expansion}",
        ),
        (
            d.gpu_merge_wave >= 0,
            f"distributed.gpu_merge_wave must be >= 0 (0 folds at once), got {d.gpu_merge_wave}",
        ),
    )
    for ok, message in checks:
        if not ok:
            raise ConfigError(message)
