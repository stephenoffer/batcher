"""Making RMM the *only* device allocator in a RAPIDS worker, rather than one of three.

`accel.allocator` builds an RMM memory resource and points cuDF at it. That is where the story
used to end, and it left two other libraries allocating device memory out of the same board
through allocators RMM does not govern:

* **CuPy.** cuDF hands off to it for `.values` and `to_cupy`, for rolling windows, for several
  statistical reductions, and for every host/device array interop the translator touches. Left
  alone, CuPy builds its own device memory pool straight from the driver.
* **Numba.** cuDF compiles user-defined functions through Numba's CUDA target, which ships its
  own memory manager.

Two pools on one device is not a tuning detail. `pool_max_fraction` defaults to `1.0`, so an
RMM `pool` resource reserves everything the VRAM headroom leaves, and the first CuPy allocation
after it fails against a device RMM is holding and CuPy cannot see. The shipped `async` resource
returns freed blocks to the driver and so merely *competes* rather than deadlocking — which is
exactly why this survived: reproducing it needs the non-default allocator, or a device full
enough that the two pools' combined high-water mark matters, and a correctness run on a quiet
GPU is neither.

The other half of the same problem is *which* device the resource is installed for.
`set_current_device_resource` writes one entry in a per-device table, for whatever device is
current at that instant; a worker that later moves — `torch.cuda.set_device`, or a second
granted device — allocates through the default resource there while the state read-back still
reports the pool as applied. Both halves need more than one device, or more than one library,
to appear at all.

Everything here is best-effort per library. An absent, older, or unco-operative one leaves the
process exactly as it was: on its own allocator, still computing correct results.
"""

from __future__ import annotations

import sys

from batcher._internal.logging import note_suppressed

__all__ = ["adopt_rmm_everywhere", "install_rmm_resource"]


def install_rmm_resource(rmm, resource) -> None:
    """Point RMM at `resource` for the device this worker is bound to, and for "current".

    Both entries are written rather than only the per-device one, because a library that does
    not consult the table reads the current-device resource, and the two must not disagree.

    Args:
        rmm: The imported `rmm` module.
        resource: The device memory resource to install.
    """
    rmm.mr.set_current_device_resource(resource)
    per_device = getattr(rmm.mr, "set_per_device_resource", None)
    if per_device is None:  # pragma: no cover - only on an RMM without the per-device table
        return
    from batcher._internal.hardware.devices import current_ordinal

    ordinal = current_ordinal()
    if ordinal is not None:
        per_device(ordinal, resource)


def adopt_rmm_everywhere() -> None:
    """Route CuPy's and Numba's device allocations through RMM, where each is present."""
    for library, adopt in (("cupy", _adopt_in_cupy), ("numba", _adopt_in_numba)):
        try:
            adopt()
        except Exception as exc:  # pragma: no cover - needs the library actually installed
            note_suppressed("carbonite", f"route {library} allocations through RMM", exc)


def _adopt_in_cupy() -> None:
    """Install RMM as CuPy's device allocator, if CuPy is loaded in this process.

    `sys.modules` rather than an import: a relational worker that never touches CuPy should not
    pay a multi-second import to be told so, and a worker that *does* use it has imported it
    (directly, or through cuDF) well before it allocates.
    """
    cupy = sys.modules.get("cupy")
    if cupy is None:
        return
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)


def _adopt_in_numba() -> None:
    """Install RMM as Numba's CUDA memory manager, if Numba is loaded in this process.

    Numba latches its memory manager the first time a CUDA context is used, so this can only
    take effect early — which is why it is attempted rather than asserted. A worker that has
    already run a UDF keeps the manager it has, which allocates correctly and merely outside
    the pool.
    """
    if sys.modules.get("numba") is None:
        return
    from numba import cuda
    from rmm.allocators.numba import RMMNumbaManager

    cuda.set_memory_manager(RMMNumbaManager)
