"""GPU-accelerated relational transform kernels (the compute core of a GPU backend).

Pure, self-contained GPU compute over Arrow: a group-by aggregate and a filtered reduction run
on the GPU via torch, returning Arrow. No Ray, no scheduling — the *dispatch* (run locally when
the process owns a GPU, else on a GPU worker) is a `dist`/`api` concern; this module is the
kernel both call, so the two execution backends (the native Rust CPU engine and this GPU path)
share one tested implementation. Torch is the CUDA-13 vehicle here; a cuDF backend can slot in
behind the same surface. Every kernel is result-identical to the CPU engine (a fused GPU
pipeline just does the same relational math on the device), so a GPU backend is a *where*, not
a *what* — the mergeable-algebra spirit applied to accelerators.

`gpu_available()` gates use; the kernels raise `BackendError` if torch/CUDA is absent rather
than silently returning wrong results, so a caller falls back to the CPU engine explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["gpu_available", "gpu_groupby_agg"]

# The reductions a GPU group-by supports; each maps to a scatter-based kernel below.
_SUPPORTED_AGGS = ("sum", "count", "mean", "min", "max")


def gpu_available() -> bool:
    """Whether a CUDA-capable torch is importable *and* a device is present in this process.

    False on a GPU-less host (the driver), where a caller dispatches to a GPU worker or falls
    back to the CPU engine. Never raises."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def gpu_groupby_agg(
    table: pa.Table, key: str, aggs: dict[str, tuple[str, str]]
) -> pa.Table:
    """Group `table` by `key` and compute `aggs` on the GPU, returning an Arrow table.

    `aggs` maps ``output_name -> (column, reduction)`` where reduction is one of
    `sum`/`count`/`mean`/`min`/`max`. The key column and all aggregated columns must be numeric.
    The result matches the CPU engine's `group_by(key).agg(...)` (up to float summation order),
    so this is a drop-in accelerated backend for the shape. Raises `BackendError` if no GPU is
    available (the caller then uses the CPU engine).

    The whole aggregate is one host->device transfer + scatter reductions on the device — the
    fused-pipeline win: transfer once, reduce on the GPU. Keys are densified with
    ``torch.unique`` so arbitrary integer keys map to contiguous buckets.
    """
    if not gpu_available():
        from batcher._internal.errors import BackendError

        raise BackendError("gpu_groupby_agg needs a CUDA-capable torch on a GPU device")
    return _torch_groupby_agg(table, key, aggs, device="cuda")


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

    for name, (_col, red) in aggs.items():
        if red not in _SUPPORTED_AGGS:
            from batcher._internal.errors import BackendError

            raise BackendError(f"unsupported GPU reduction {red!r} for {name!r}")

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
