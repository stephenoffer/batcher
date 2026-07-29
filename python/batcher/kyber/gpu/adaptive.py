"""Adaptive GPU crossover — learn where the GPU backend starts beating the CPU engine.

`gpu_min_rows` is a measured default for one cluster; this closes Kyber's learning loop for the
backend choice so it self-corrects to *this* hardware. Every GPU or CPU group-by run records its
(actual input rows, wall time) to the hub — the *actual* footer row count, not an estimate, so
the same source yields the same x on both backends and cold/warm estimate drift can't pollute
the fit. From the two point-clouds we fit one line per backend — `t ≈ a + b·rows` — and solve
for their crossover, the row count above which the GPU's lower per-row cost overtakes its higher
fixed overhead. **Core measures, Kyber consumes:** a faster GPU, a slower CPU, or a wider table
moves the threshold on its own. Until enough distinct samples exist for both backends the learned
value is `None` and the caller keeps the config default.

Storage is a per-backend bucket of ordinary least-squares sufficient statistics
(`n, Σx, Σy, Σxx, Σxy`) under one hub learned-params namespace, updated in place — O(1) per run,
no history scan. Everything here is best-effort: a malformed bucket or a degenerate fit yields
`None`, never an exception into execution or planning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed
from batcher.config import active_config
from batcher.kyber.ols import fit_ols, ols_update
from batcher.metadata.hardware_scope import scoped

if TYPE_CHECKING:
    from batcher.metadata import MetadataHub

__all__ = ["learned_gpu_min_rows", "record_backend_timing"]

_NS = "gpu_backend_xover"
# Clamp the learned crossover to a band around the config default, so one bad early fit can
# only nudge the threshold within a bounded range, never send it to an absurd value. (The
# sample-count and spread gates that decide whether a fit is usable at all live in `kyber.ols`.)
_BAND = 8.0  # learned ∈ [default / _BAND, default * _BAND]


def _bucket(backend: str, accelerator_type: str | None) -> str:
    """The storage bucket for a backend's OLS statistics, per device model where known.

    A GPU timing is only comparable to another timing from the *same* device: an H100 and a T4
    differ by roughly an order of magnitude in per-row cost, so pooling them fits one line
    through two unrelated point-clouds and converges on a crossover right for neither — while
    still *overriding* the config default, so the learned value is actively worse than no
    learning at all. Splitting the bucket per model is what makes more data help rather than hurt.

    The CPU bucket stays pooled. Heterogeneous CPUs have the same issue in principle, but the
    spread across a fleet's cores is far narrower than across GPU generations, and there is no
    equivalent per-node CPU model label to key on. Unkeyed GPU timings (a mixed fleet, an
    unlabelled node) also stay in the shared bucket, which is exactly the behavior before this.
    """
    return f"{backend}:{accelerator_type}" if backend == "gpu" and accelerator_type else backend


def record_backend_timing(
    hub: MetadataHub | None,
    backend: str,
    rows: int,
    wall_ms: float,
    accelerator_type: str | None = None,
) -> None:
    """Fold one (rows, wall_ms) observation for `backend` ("gpu"/"cpu") into its OLS statistics.

    `accelerator_type` buckets a GPU observation by device model so unlike devices don't share
    one regression; `None` keeps the pooled bucket."""
    if hub is None or rows <= 0 or wall_ms <= 0.0 or backend not in ("gpu", "cpu"):
        return
    try:
        key = _bucket(backend, accelerator_type)
        s = hub.get_keyed_param(scoped(_NS), key) or {}
        hub.put_keyed_param(scoped(_NS), key, ols_update(s, float(rows), float(wall_ms)))
    except Exception as exc:  # pragma: no cover - learning must never break a query
        note_suppressed("kyber", "record backend timing", exc)
        return


# `(intercept_ms, slope_ms_per_row)` for a backend, or None when the samples are too few or
# too clustered to identify a line. Shared with the broadcast/sort-merge crossovers, which
# fold the same statistics — the two copies of this fit had already diverged once.
_fit = fit_ols


def learned_gpu_min_rows(
    hub: MetadataHub | None, accelerator_type: str | None = None
) -> int | None:
    """The measured GPU/CPU crossover row count, or `None` when not yet learnable.

    Solves `a_gpu + b_gpu·x == a_cpu + b_cpu·x`. A crossover exists only when the GPU is cheaper
    per row (`b_gpu < b_cpu`) yet has higher fixed cost (`a_gpu > a_cpu`) — the expected regime;
    any other shape (GPU dominates everywhere, or never wins) means "no useful threshold from the
    data", so we defer to the config default. The result is clamped to a sane band.

    `accelerator_type` selects this device model's own samples, falling back to the pooled bucket
    when that model has none yet — so a fleet whose device is newly identified keeps using what it
    already learned instead of going cold."""
    if hub is None:
        return None
    try:
        gpu = _fit(hub.get_keyed_param(scoped(_NS), _bucket("gpu", accelerator_type)) or {})
        if gpu is None and accelerator_type:
            gpu = _fit(hub.get_keyed_param(scoped(_NS), "gpu") or {})
        cpu = _fit(hub.get_keyed_param(scoped(_NS), "cpu") or {})
    except Exception as exc:  # pragma: no cover
        note_suppressed("kyber", "fit gpu crossover", exc)
        return None
    if gpu is None or cpu is None:
        return None
    a_gpu, b_gpu = gpu
    a_cpu, b_cpu = cpu
    if not (b_cpu > b_gpu and a_gpu > a_cpu):
        return None
    xover = (a_gpu - a_cpu) / (b_cpu - b_gpu)
    if xover <= 0.0:
        return None
    default = float(active_config().distributed.gpu_min_rows)
    return int(min(max(xover, default / _BAND), default * _BAND))
