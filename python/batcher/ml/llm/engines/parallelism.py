"""How many GPUs one LLM engine replica needs, and what that choice costs.

Tensor parallelism splits a model's weights across GPUs so a model too large for one card
can run at all. It is not free: every forward does an all-reduce across the group, so the
interconnect decides whether TP is nearly free or ruinous. The field guides put numbers on
it — NVLink at 600-900 GB/s makes TP efficient, while PCIe Gen4/Gen5 at 32-64 GB/s is
10-50x slower and costs a **measured 30-50% throughput** at TP>=2 on Llama-70B.

That gives two questions worth answering before a run rather than after:

* **Is TP even needed?** Below the point where weights plus KV cache exceed one card, TP=1
  is always fastest, because there is no communication at all.
* **Is TP going to hurt here?** The same TP=2 is routine on an H100 and a serious tax on an
  L4 — the model is identical, the interconnect is not.

This module answers both as arithmetic over a model footprint and a card's VRAM, so it is
testable without a GPU. It deliberately does **not** choose `tensor_parallel_size` for the
user: the penalty above is hardware-specific and unmeasurable from this process, and a
wrong automatic choice would silently halve throughput on the exact hardware where the user
would never think to look. Advice, not a decision.
"""

from __future__ import annotations

__all__ = [
    "local_device_name",
    "measured_link_class",
    "minimum_tensor_parallel_size",
    "nvlink_class",
    "warn_about_tensor_parallelism",
]

#: Share of a card's VRAM a model's weights may occupy before it needs company. The rest is
#: KV cache, activations, and the CUDA context — vLLM's own `gpu_memory_utilization` default
#: is 0.90, and weights that fill that leave no cache, which is the same as not fitting.
_WEIGHT_BUDGET = 0.55
#: GPU families whose multi-GPU links are PCIe rather than NVLink. The guides name these
#: specifically as the cards where TP>=2 costs 30-50%: L4, A10G, L40S.
_PCIE_ONLY = ("l4", "a10", "l40", "t4", "rtx", "a2000", "a4000", "a5000", "a6000")
#: Families with NVLink/NVSwitch, where TP is efficient.
_NVLINK = ("a100", "h100", "h200", "h20", "b200", "v100", "gh200")


def nvlink_class(device_name: str | None) -> str:
    """``"nvlink"``, ``"pcie"``, or ``"unknown"`` for a GPU model name.

    Args:
        device_name: The card's reported name, e.g. ``"NVIDIA L4"``.

    Returns:
        Which interconnect family the card belongs to.

    Examples:
        .. doctest::

            >>> from batcher.ml.llm.engines.parallelism import nvlink_class
            >>> nvlink_class("NVIDIA A100-SXM4-80GB")
            'nvlink'
            >>> nvlink_class("NVIDIA L4")
            'pcie'
    """
    if not device_name:
        return "unknown"
    name = device_name.lower()
    if any(tag in name for tag in _NVLINK):
        return "nvlink"
    if any(tag in name for tag in _PCIE_ONLY):
        return "pcie"
    return "unknown"


def minimum_tensor_parallel_size(model_gb: float, vram_gb: float | None) -> int:
    """The smallest TP degree whose combined VRAM can hold `model_gb` of weights.

    Rounded up to a power of two, because a tensor-parallel group splits attention heads
    evenly and vLLM requires the head count to be divisible by the degree — 3 GPUs is not a
    configuration, 4 is.

    Args:
        model_gb: The model's weight footprint in GB.
        vram_gb: One card's VRAM in GB, or `None` when it cannot be measured.

    Returns:
        The minimum workable degree, or `1` when the model fits or nothing is known.

    Examples:
        .. doctest::

            >>> from batcher.ml.llm.engines.parallelism import minimum_tensor_parallel_size
            >>> minimum_tensor_parallel_size(14.0, 24.0)
            1
            >>> minimum_tensor_parallel_size(140.0, 80.0)
            4
    """
    if model_gb <= 0 or not vram_gb or vram_gb <= 0:
        return 1
    budget = vram_gb * _WEIGHT_BUDGET
    if model_gb <= budget:
        return 1
    degree = 1
    while degree < 64 and model_gb > budget * degree:
        degree *= 2
    return degree


#: Warned once per process; the engine is built per worker and would otherwise repeat.
_TP_WARNED = False


def warn_about_tensor_parallelism(
    declared: int, model_gb: float, vram_gb: float | None, device_name: str | None
) -> None:
    """Say once when the declared TP degree looks wrong for this model and this hardware.

    Three distinct mistakes, with different fixes:

    * **Too low** — the weights cannot fit the group at all, so the engine will OOM on
      load. Better said before the model download than after it.
    * **Too high for the interconnect** — TP>=2 on a PCIe-only card costs a measured
      30-50% throughput. Sometimes unavoidable (the model must fit); worth knowing either
      way, because the same setting on an NVLink card is nearly free.
    * **The interconnect is not what the card says** — an SXM board whose NVLink is down is
      a PCIe card that every nameplate check calls an NVLink one. It is the same throughput
      loss with none of the visibility, and unlike the two above it is a node fault rather
      than a setting, so the fix is to drain the node instead of changing the degree.

    Nothing is changed: the degree stays exactly what the caller asked for. The penalty is
    hardware-specific and unmeasurable from here, so this is advice, not a decision.

    Args:
        declared: The `tensor_parallel_size` the caller set (1 when unset).
        model_gb: The model's weight footprint, or 0 when unknown.
        vram_gb: One card's VRAM, or `None` when unmeasurable.
        device_name: The card's reported name, for the interconnect class.
    """
    global _TP_WARNED
    if _TP_WARNED:
        return
    needed = minimum_tensor_parallel_size(model_gb, vram_gb)
    link = nvlink_class(device_name)
    message = ""
    if needed > max(1, declared):
        message = (
            f"this model needs about {model_gb:.0f} GB of weights but "
            f"tensor_parallel_size={declared} gives it "
            f"{(vram_gb or 0) * max(1, declared):.0f} GB of VRAM to live in. It will very "
            f"likely fail to load; tensor_parallel_size={needed} is the smallest group that "
            f"fits."
        )
    elif declared >= 2 and link == "nvlink" and measured_link_class() == "pcie":
        message = (
            f"tensor_parallel_size={declared} on {device_name}, whose NVLink fabric is "
            f"reported DOWN on this node. The card supports NVLink, so the group looks free "
            f"on paper; with the links down every forward all-reduces over PCIe instead, at "
            f"a fraction of the rate. This is a node fault, not a setting: check "
            f"`bt.accelerators()['fabric']` and drain the node if the links do not come back."
        )
    elif declared >= 2 and link == "pcie":
        message = (
            f"tensor_parallel_size={declared} on {device_name} — a PCIe-only card. Every "
            f"forward all-reduces over PCIe rather than NVLink, which is measured at 30-50% "
            f"lower throughput. Unavoidable if the model does not otherwise fit; if it does, "
            f"tensor_parallel_size=1 will be faster."
        )
    if not message:
        return
    _TP_WARNED = True
    import warnings

    from batcher._internal.errors import PerformanceWarning

    warnings.warn(message, PerformanceWarning, stacklevel=3)


def local_device_name() -> str | None:
    """This worker's GPU model name, or `None` when there is no readable device.

    Read from torch rather than NVML because the LLM path already requires torch and this
    runs on the GPU worker, where the device is present by construction.

    Returns:
        The card's reported name, e.g. ``"NVIDIA L4"``.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return str(torch.cuda.get_device_name(0))
    except Exception:  # pragma: no cover - no driver, no device, or an older torch
        return None


def measured_link_class() -> str:
    """What this node's device fabric is *doing*, as opposed to what its cards support.

    `nvlink_class` reads a model name, which says what the hardware can do. This reads the
    links, which says what they are doing — and the two differ on exactly the node where it
    matters: a board whose NVLink has dropped still reports an NVLink-capable model, so every
    nameplate check clears it while its collectives run over PCIe.

    Returns:
        `"nvlink"` when every device with a fabric has all of its links up, `"pcie"` when
        devices report links and none are up, and `"unknown"` when the driver publishes
        nothing — including on a node whose cards have no NVLink at all, where "down" would
        be a misleading way to say "absent".
    """
    from batcher._internal.hardware.fabric import nvlink_status

    records = [status for status in nvlink_status() if status.links > 0]
    if not records:
        return "unknown"
    if all(status.active_links == status.links for status in records):
        return "nvlink"
    # Partially down counts as `"pcie"`, not as a third state: a collective is bounded by the
    # slowest pair in its group, so one device off the fabric costs the group the fabric.
    return "pcie"
