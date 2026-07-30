"""GPU-accelerated relational transform kernels (the compute core of a GPU backend).

Pure, self-contained GPU compute over Arrow: a group-by aggregate runs on the GPU and returns
Arrow. It uses **cuDF** (RAPIDS) when importable — a mature GPU dataframe, ~3x the hand-rolled
torch kernel and the engine behind Polars-GPU — and falls back to a torch scatter-reduce kernel
otherwise. No Ray, no scheduling — the *dispatch* (run locally when the process owns a GPU, else
on a GPU worker; ship cuDF via the task runtime_env) is a `dist`/`api` concern; this module is
the kernel both call, so the two execution backends (the native Rust CPU engine and this GPU
path) share one tested implementation. Every kernel is result-identical to the CPU engine (the
same relational math on the device), so a GPU backend is a *where*, not a *what* — the
mergeable-algebra spirit applied to accelerators.

`gpu_available()` gates use; the kernels raise `BackendError` if torch/CUDA is absent rather
than silently returning wrong results, so a caller falls back to the CPU engine explicitly.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from batcher._internal.hardware import accelerator_backend, gpu_devices_absent
from batcher._internal.logging import note_suppressed
from batcher.core.energy import measure_stage

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["gpu_available", "gpu_groupby_agg"]

# The reductions a GPU group-by supports; each maps to a scatter-based kernel below.
# Torch device string per detected backend. ROCm speaks the CUDA API, so it uses the
# ``cuda`` string; a backend with no torch device (or an unknown one) falls back to CPU
# rather than raising — the accelerated path is an optimization, never a requirement.
_TORCH_DEVICE = {"cuda": "cuda", "rocm": "cuda", "xpu": "xpu", "mps": "mps"}

_SUPPORTED_AGGS = ("sum", "count", "mean", "min", "max")


@functools.lru_cache(maxsize=1)
def gpu_available() -> bool:
    """Whether torch is importable *and* an accelerator this module can compute on is present.

    False on a GPU-less host (the driver), where a caller dispatches to a GPU worker or falls
    back to the CPU engine. Never raises.

    The probe follows the *detected backend*, not CUDA alone. `_TORCH_DEVICE` and the scatter
    kernel below already support `xpu` (Intel) and `mps` (Apple), but this gate asked
    `torch.cuda.is_available()`, which is False on both — so the one function deciding whether
    to use those devices reported "no accelerator" on hosts that had a working one, and the
    kernel supporting them was unreachable. A backend with no torch device still returns False:
    a TPU host has an accelerator, but not one these kernels can drive.

    Short-circuits on the cheap device-node check before touching torch, and memoizes the
    answer. Importing torch costs ~2 s and this is reached from the post-collect GPU-crossover
    probe, so *every* run on a GPU-less host paid that once — a 2 s first-query stall to
    discover there is no GPU to use, on the machines least able to profit from it. Device
    presence cannot change within a process, so caching loses nothing.
    """
    if gpu_devices_absent():
        return False
    device = _TORCH_DEVICE.get(accelerator_backend())
    if device is None:
        return False
    try:
        import torch

        if device == "cuda":  # also ROCm, which speaks the CUDA API
            return bool(torch.cuda.is_available())
        if device == "xpu":
            xpu = getattr(torch, "xpu", None)
            return bool(xpu is not None and xpu.is_available())
        mps = getattr(torch.backends, "mps", None)  # device == "mps"
        return bool(mps is not None and mps.is_available())
    except Exception as exc:
        note_suppressed("core", "probe gpu", exc)
        return False


def _validate_aggs(aggs: dict[str, tuple[str, str]]) -> None:
    """Reject a reduction this backend cannot compute, naming the offending output.

    Called from both entry points — the dispatcher and the device kernel, which tests
    drive directly on ``device="cpu"`` — so neither can be reached with an aggregate the
    kernel would silently mis-handle."""
    from batcher._internal.errors import BackendError

    for name, (_col, red) in aggs.items():
        if red not in _SUPPORTED_AGGS:
            raise BackendError(f"unsupported GPU reduction {red!r} for {name!r}")


def gpu_groupby_agg(table: pa.Table, key: str, aggs: dict[str, tuple[str, str]]) -> pa.Table:
    """Group `table` by `key` and compute `aggs` on the GPU, returning an Arrow table.

    `aggs` maps ``output_name -> (column, reduction)`` where reduction is one of
    `sum`/`count`/`mean`/`min`/`max`. The key column and all aggregated columns must be numeric.
    The result matches the CPU engine's `group_by(key).agg(...)` (up to float summation order),
    so this is a drop-in accelerated backend for the shape. Raises `BackendError` if no GPU is
    available (the caller then uses the CPU engine).

    Uses **cuDF** (RAPIDS) when it is importable — a mature GPU dataframe whose group-by is
    ~3x the hand-rolled torch kernel (measured) and the same engine Polars-GPU builds on — and
    falls back to a torch scatter-reduce kernel otherwise. Both are result-identical to the CPU
    engine (up to float summation order). Raises `BackendError` if no GPU is available.
    """
    from batcher._internal.errors import BackendError

    # Validate the request before probing hardware: an unsupported reduction is a caller
    # error whatever device is attached, and reporting "no accelerator" for a typo'd
    # aggregate would send the reader hunting for a GPU they do not need.
    _validate_aggs(aggs)
    if not gpu_available():
        raise BackendError("gpu_groupby_agg needs a CUDA-capable torch on a GPU device")
    # cuDF is CUDA-only by construction, so it is tried only on a CUDA/ROCm device. The
    # torch kernel is device-parameterized (see `_torch_groupby_agg`), so an Intel XPU or
    # Apple MPS host runs the accelerated path on *its* device instead of the CUDA string,
    # which would have raised and dropped the whole stage back to the CPU engine.
    backend = accelerator_backend()
    device = _TORCH_DEVICE.get(backend)
    # `_TORCH_DEVICE` maps only accelerator backends, so an unknown/absent one is `None`.
    # (A `device == "cpu"` disjunct used to sit here; no entry has that value, so it read as
    # covering a case it could never match.)
    if device is None:
        # No accelerator to run on. Raise rather than quietly computing on the CPU *inside*
        # the GPU kernel: the caller asked for the accelerated backend and owns the decision
        # to fall back, and a silent CPU computation here would report GPU acceleration that
        # never happened — the exact failure this backend is supposed to make visible.
        raise BackendError(
            f"gpu_groupby_agg found no usable accelerator (detected backend {backend!r}); "
            "the caller should use the CPU engine"
        )
    # Bracketed so the stage's energy is recorded wherever one is being collected: this is
    # the one place every GPU relational stage passes through, local dispatch and Ray worker
    # alike, and it is the only point that knows both the device and the rows it produced.
    # Outside an `energy_scope` the meter returns immediately, so the untraced path is
    # unchanged.
    with measure_stage(
        f"GpuGroupBy#{id(table) & 0xFFFF}",
        accelerator_type=_local_device_model(),
        device_count=1,
    ) as meter:
        out = _dispatch_groupby(table, key, aggs, backend=backend, device=device)
        meter.add_rows(out.num_rows)
        return out


def _dispatch_groupby(
    table: pa.Table,
    key: str,
    aggs: dict[str, tuple[str, str]],
    *,
    backend: str,
    device: str,
) -> pa.Table:
    """Run the group-by on the best kernel this backend has: cuDF where installed, else torch."""
    if backend in ("cuda", "rocm"):
        try:
            import cudf  # noqa: F401

            return _cudf_groupby_agg(table, key, aggs)
        except ImportError:
            pass  # cudf not installed -> fall through to the next backend, then the CPU
    return _torch_groupby_agg(table, key, aggs, device=device)


@functools.lru_cache(maxsize=1)
def _local_device_model() -> str:
    """This host's device model as a `device_specs` key, or `""` when it cannot be resolved.

    Memoized: the attached hardware does not change within a process, and the lookup would
    otherwise run per stage on a path that is meant to be free when nothing is measuring.
    """
    from batcher._internal.device_specs import resolve_device_name
    from batcher._internal.hardware import gpu_inventory

    for device in gpu_inventory():
        resolved = resolve_device_name(str(device.get("name") or ""))
        if resolved:
            return resolved
    return ""


def _cudf_groupby_agg(table: pa.Table, key: str, aggs: dict[str, tuple[str, str]]) -> pa.Table:
    """The cuDF (RAPIDS) group-by kernel — a mature GPU dataframe, ~3x the torch kernel and
    the engine behind Polars-GPU. Arrow in, Arrow out; result-identical to the CPU engine."""
    import cudf

    gdf = cudf.DataFrame.from_arrow(table)
    col_funcs: dict[str, list[str]] = {}
    for _alias, (col_name, func) in aggs.items():
        col_funcs.setdefault(col_name, [])
        if func not in col_funcs[col_name]:
            col_funcs[col_name].append(func)
    grouped = gdf.groupby(key, sort=False).agg(col_funcs)
    out = cudf.DataFrame({key: grouped.index.to_pandas()})
    for alias, (col_name, func) in aggs.items():
        out[alias] = grouped[(col_name, func)].reset_index(drop=True)
    return out.to_arrow()


def _torch_groupby_agg(
    table: pa.Table, key: str, aggs: dict[str, tuple[str, str]], device: str
) -> pa.Table:
    """The device-parameterized group-by kernel shared by the GPU path and its CPU-torch test.

    Identical code on ``device="cuda"`` (the accelerated backend) and ``device="cpu"`` (so the
    densify + scatter-reduce algorithm is verifiable against the CPU engine without a GPU — the
    GPU is only *where* it runs)."""
    import numpy as np
    import pyarrow as pa
    import torch

    _validate_aggs(aggs)

    dev = torch.device(device)
    keys_np = table.column(key).to_numpy(zero_copy_only=False)
    keys_t = torch.from_numpy(np.ascontiguousarray(keys_np)).to(dev)
    uniq, inv = torch.unique(keys_t, sorted=True, return_inverse=True)
    n_groups = int(uniq.numel())
    counts = torch.zeros(n_groups, device=dev, dtype=torch.float64).scatter_add_(
        0, inv, torch.ones_like(inv, dtype=torch.float64)
    )

    out_cols: dict[str, object] = {key: uniq.cpu().numpy()}
    for name, (col, red) in aggs.items():
        if red == "count":
            out_cols[name] = counts.to(torch.int64).cpu().numpy()
            continue
        vals = torch.from_numpy(
            np.ascontiguousarray(table.column(col).to_numpy(zero_copy_only=False))
        ).to(dev, dtype=torch.float64)
        if red in ("sum", "mean"):
            acc = torch.zeros(n_groups, device=dev, dtype=torch.float64).scatter_add_(0, inv, vals)
            res = acc / counts if red == "mean" else acc
        else:  # min / max via scatter_reduce
            init = float("inf") if red == "min" else float("-inf")
            acc = torch.full((n_groups,), init, device=dev, dtype=torch.float64)
            acc = acc.scatter_reduce(0, inv, vals, reduce="amin" if red == "min" else "amax")
            res = acc
        out_cols[name] = res.cpu().numpy()

    return pa.table({k: pa.array(v) for k, v in out_cols.items()})
