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

from batcher.config import active_config
from batcher.kyber.ols import fit_ols, ols_update

if TYPE_CHECKING:
    from batcher.metadata import MetadataHub

__all__ = ["learned_gpu_min_rows", "record_backend_timing"]

_NS = "gpu_backend_xover"
# Clamp the learned crossover to a band around the config default, so one bad early fit can
# only nudge the threshold within a bounded range, never send it to an absurd value. (The
# sample-count and spread gates that decide whether a fit is usable at all live in `kyber.ols`.)
_BAND = 8.0  # learned ∈ [default / _BAND, default * _BAND]


def record_backend_timing(hub: MetadataHub | None, backend: str, rows: int, wall_ms: float) -> None:
    """Fold one (rows, wall_ms) observation for `backend` ("gpu"/"cpu") into its OLS statistics."""
    if hub is None or rows <= 0 or wall_ms <= 0.0 or backend not in ("gpu", "cpu"):
        return
    try:
        s = hub.get_keyed_param(_NS, backend) or {}
        hub.put_keyed_param(_NS, backend, ols_update(s, float(rows), float(wall_ms)))
    except Exception:  # pragma: no cover - learning must never break a query
        return


# `(intercept_ms, slope_ms_per_row)` for a backend, or None when the samples are too few or
# too clustered to identify a line. Shared with the broadcast/sort-merge crossovers, which
# fold the same statistics — the two copies of this fit had already diverged once.
_fit = fit_ols


def learned_gpu_min_rows(hub: MetadataHub | None) -> int | None:
    """The measured GPU/CPU crossover row count, or `None` when not yet learnable.

    Solves `a_gpu + b_gpu·x == a_cpu + b_cpu·x`. A crossover exists only when the GPU is cheaper
    per row (`b_gpu < b_cpu`) yet has higher fixed cost (`a_gpu > a_cpu`) — the expected regime;
    any other shape (GPU dominates everywhere, or never wins) means "no useful threshold from the
    data", so we defer to the config default. The result is clamped to a sane band."""
    if hub is None:
        return None
    try:
        gpu = _fit(hub.get_keyed_param(_NS, "gpu") or {})
        cpu = _fit(hub.get_keyed_param(_NS, "cpu") or {})
    except Exception:  # pragma: no cover
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
