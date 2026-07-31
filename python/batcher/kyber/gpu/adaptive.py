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

__all__ = [
    "learned_device_throughput",
    "learned_gpu_min_rows",
    "record_backend_timing",
    "record_device_throughput",
    "shape_key",
]

_NS = "gpu_backend_xover"
#: Where per-device-model throughput is folded. Separate from the crossover namespace because
#: it answers a different question — how fast *this* model is, rather than where it overtakes
#: the CPU — and a stage dealing shards across a heterogeneous node needs the first without
#: having enough CPU samples to fit the second.
_TP_NS = "gpu_device_throughput"

#: Samples a throughput bucket needs before it is reported. One run of one shard is as likely
#: to be measuring a cold allocator or a contended host as the device, and a fleet that deals
#: its shards against that figure has made the imbalance worse rather than better.
_MIN_TP_SAMPLES = 3
# Clamp the learned crossover to a band around the config default, so one bad early fit can
# only nudge the threshold within a bounded range, never send it to an absurd value. (The
# sample-count and spread gates that decide whether a fit is usable at all live in `kyber.ols`.)
_BAND = 8.0  # learned ∈ [default / _BAND, default * _BAND]


def _bucket(backend: str, accelerator_type: str | None, shape: str | None = None) -> str:
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

    `shape` splits the bucket again, by the query's structure. Two pipelines on one device
    share a bucket and average toward each other otherwise: a wide projection is
    transfer-bound and a group-by over a narrow key is not, and their crossovers differ by
    more than the device generations do. The shape is a plan signature, so it is the same key
    the cardinality learner already indexes by, and both halves of the fit are per shape or
    neither is — a shape-keyed GPU line solved against a pooled CPU line is two different
    workloads' regressions crossed against each other.
    """
    base = f"{backend}:{accelerator_type}" if backend == "gpu" and accelerator_type else backend
    return f"{base}@{shape}" if shape else base


def shape_key(plan) -> str | None:
    """The bucket key for a plan's workload shape, or `None` when it cannot be taken.

    One helper for both halves of the loop — the recorder in the conductor and the reader in
    the policy — because a shape recorded under one key and read under another is a bucket that
    silently never fills. Best-effort by construction: the signature is a *key*, so a plan whose
    shape cannot be hashed falls back to the pooled bucket rather than failing to be planned.

    Args:
        plan: The logical plan whose shape is being keyed.

    Returns:
        The structural signature, or `None`.
    """
    try:
        from batcher.kyber.signature import plan_signature

        return plan_signature(plan)
    except Exception as exc:  # pragma: no cover - a learning key never fails a query
        note_suppressed("kyber", "take a plan signature for the crossover learner", exc)
        return None


def record_backend_timing(
    hub: MetadataHub | None,
    backend: str,
    rows: int,
    wall_ms: float,
    accelerator_type: str | None = None,
    shape: str | None = None,
) -> None:
    """Fold one (rows, wall_ms) observation for `backend` ("gpu"/"cpu") into its OLS statistics.

    `accelerator_type` buckets a GPU observation by device model so unlike devices don't share
    one regression; `None` keeps the pooled bucket. `shape` is the plan signature, which splits
    the bucket again by what the query does — recorded into *both* the shaped and the pooled
    bucket, so a shape seen once still contributes to the fleet-wide fit that every other query
    reads, and a shape seen often eventually earns a fit of its own."""
    if hub is None or rows <= 0 or wall_ms <= 0.0 or backend not in ("gpu", "cpu"):
        return
    keys = [_bucket(backend, accelerator_type)]
    if shape:
        keys.append(_bucket(backend, accelerator_type, shape))
    try:
        for key in keys:
            s = hub.get_keyed_param(scoped(_NS), key) or {}
            hub.put_keyed_param(scoped(_NS), key, ols_update(s, float(rows), float(wall_ms)))
    except Exception as exc:  # pragma: no cover - learning must never break a query
        note_suppressed("kyber", "record backend timing", exc)
        return


def record_device_throughput(
    hub: MetadataHub | None, accelerator_type: str | None, rows: int, seconds: float
) -> None:
    """Fold one device's measured rows-per-second into its model's throughput bucket.

    What a fleet needs to deal shards proportionally rather than round-robin
    (`dist.gpu.fabric.placement.device_shard_counts`). Kept as its own statistic rather than derived
    from the crossover fit: the fit needs both backends and enough spread to identify a line,
    and a node that has only ever run on GPUs still has to divide work between them.

    Args:
        hub: The metadata hub, or `None` to record nothing.
        accelerator_type: The device model, `None` for an unlabelled device (pooled).
        rows: Rows the device processed.
        seconds: How long it took.
    """
    if hub is None or rows <= 0 or seconds <= 0.0:
        return
    try:
        key = accelerator_type or "unknown"
        s = hub.get_keyed_param(scoped(_TP_NS), key) or {}
        n = int(s.get("n", 0)) + 1
        hub.put_keyed_param(
            scoped(_TP_NS),
            key,
            {
                "n": n,
                "rows": float(s.get("rows", 0.0)) + float(rows),
                "seconds": float(s.get("seconds", 0.0)) + float(seconds),
            },
        )
    except Exception as exc:  # pragma: no cover - learning must never break a query
        note_suppressed("kyber", "record device throughput", exc)


def learned_device_throughput(hub: MetadataHub | None, accelerator_type: str | None) -> float:
    """This device model's measured rows per second, or `0.0` when not yet learnable.

    Totals rather than a mean of means: a bucket accumulates rows and seconds, so a long shard
    weighs more than a short one, which is the correct weighting for a figure whose whole use
    is dividing a long stage's work.

    Args:
        hub: The metadata hub, or `None`.
        accelerator_type: The device model, `None` for the pooled bucket.

    Returns:
        Rows per second, `0.0` when the hub is absent, the bucket is empty, or it is
        under-sampled. Zero is "no opinion", and every consumer here reads it as "treat this
        device as average" rather than as "this device is slow".
    """
    if hub is None:
        return 0.0
    try:
        s = hub.get_keyed_param(scoped(_TP_NS), accelerator_type or "unknown") or {}
    except Exception as exc:  # pragma: no cover
        note_suppressed("kyber", "read device throughput", exc)
        return 0.0
    seconds = float(s.get("seconds", 0.0))
    if int(s.get("n", 0)) < _MIN_TP_SAMPLES or seconds <= 0.0:
        return 0.0
    return float(s.get("rows", 0.0)) / seconds


# `(intercept_ms, slope_ms_per_row)` for a backend, or None when the samples are too few or
# too clustered to identify a line. Shared with the broadcast/sort-merge crossovers, which
# fold the same statistics — the two copies of this fit had already diverged once.
_fit = fit_ols


def learned_gpu_min_rows(
    hub: MetadataHub | None, accelerator_type: str | None = None, shape: str | None = None
) -> int | None:
    """The measured GPU/CPU crossover row count, or `None` when not yet learnable.

    Solves `a_gpu + b_gpu·x == a_cpu + b_cpu·x`. A crossover exists only when the GPU is cheaper
    per row (`b_gpu < b_cpu`) yet has higher fixed cost (`a_gpu > a_cpu`) — the expected regime;
    any other shape (GPU dominates everywhere, or never wins) means "no useful threshold from the
    data", so we defer to the config default. The result is clamped to a sane band.

    `accelerator_type` selects this device model's own samples and `shape` this query shape's,
    each falling back when its bucket has none yet — so a fleet whose device is newly identified,
    or a query shape seen for the first time, keeps using what it already learned instead of
    going cold.

    **Both lines come from the same rung of the ladder.** A shaped GPU fit solved against a
    pooled CPU fit is two different workloads' regressions crossed against each other, and the
    crossover it produces belongs to neither; when the shaped CPU samples are missing, the
    shaped GPU fit is discarded and the pooled pair is used instead."""
    if hub is None:
        return None
    try:
        gpu = cpu = None
        if shape:
            gpu = _fit(
                hub.get_keyed_param(scoped(_NS), _bucket("gpu", accelerator_type, shape)) or {}
            )
            cpu = _fit(hub.get_keyed_param(scoped(_NS), _bucket("cpu", None, shape)) or {})
            if gpu is None or cpu is None:
                gpu = cpu = None
        if gpu is None:
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
