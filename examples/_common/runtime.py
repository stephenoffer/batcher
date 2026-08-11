"""Device selection for examples that can use a GPU but must not require one.

Every accelerator example in this suite runs on a machine with no GPU. That is the
point: the suite is a release check, and a check that skips itself on the hardware CI
actually has checks nothing. So these scripts ask for a device rather than assuming
one, and fall back to the CPU engine when there isn't one.

Resolution order, first match wins:

1. A command-line flag: ``--device gpu|cpu|auto``, or the ``--gpu`` / ``--cpu``
   shorthands.
2. The ``BATCHER_EXAMPLES_DEVICE`` environment variable, same three values.
3. Auto-detection, which reports ``gpu`` only when the engine can actually see a
   device.

Asking for ``--device gpu`` on a machine with no device is an error rather than a
silent downgrade, because the one time you type it deliberately is the time you need
to know it didn't happen.
"""

from __future__ import annotations

import os
import sys

import batcher as bt

__all__ = [
    "device_count",
    "has_gpu",
    "resolve_device",
    "resolve_distributed",
    "torch_device",
]

_VALID = ("auto", "cpu", "gpu")


def device_count() -> int:
    """Return how many accelerators the engine can see."""
    try:
        return len(bt.accelerators().get("devices", ()))
    except Exception:  # no driver, no NVML, container with no /dev/nvidia*
        return 0


def has_gpu() -> bool:
    """Report whether at least one usable accelerator is visible."""
    return device_count() > 0


def _requested(argv: list[str] | None) -> str:
    arguments = sys.argv[1:] if argv is None else argv
    for index, argument in enumerate(arguments):
        if argument == "--gpu":
            return "gpu"
        if argument == "--cpu":
            return "cpu"
        if argument == "--device" and index + 1 < len(arguments):
            return arguments[index + 1].strip().lower()
        if argument.startswith("--device="):
            return argument.split("=", 1)[1].strip().lower()
    return os.environ.get("BATCHER_EXAMPLES_DEVICE", "auto").strip().lower()


def resolve_device(argv: list[str] | None = None) -> str:
    """Return ``"gpu"`` or ``"cpu"`` for this run.

    Raises:
        SystemExit: if a GPU was explicitly requested and none is visible, or the
            requested value is not one of ``auto``, ``cpu``, ``gpu``.
    """
    request = _requested(argv)
    if request not in _VALID:
        raise SystemExit(f"--device must be one of {_VALID}, got {request!r}")
    if request == "cpu":
        return "cpu"
    if request == "gpu":
        if not has_gpu():
            raise SystemExit(
                "--device gpu was requested but the engine sees no accelerator. "
                "Drop the flag to run this example on the CPU engine."
            )
        return "gpu"
    return "gpu" if has_gpu() else "cpu"


def torch_device(argv: list[str] | None = None) -> str:
    """Return the torch device string matching :func:`resolve_device`.

    Returns ``"cuda"`` only when the engine sees a device *and* torch agrees it can
    use one; the two disagree often enough (a torch built without CUDA on a GPU host)
    that trusting either alone puts a confusing error in the middle of an example.
    """
    if resolve_device(argv) != "gpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_distributed(argv: list[str] | None = None) -> bool:
    """Report whether this run should execute on a Ray cluster.

    Distributed examples default to **off**, for the same reason the GPU ones default to
    auto: bringing up a cluster takes longer than the whole rest of the suite, so a
    release check that did it 12 times would stop being run. Opt in with ``--distributed``
    or ``BATCHER_EXAMPLES_DISTRIBUTED=1`` when you have a cluster to point at.

    With it off, the examples still run the identical query single-node and still assert
    the single-node-equals-distributed contract against the mergeable path — they just do
    not pay for the scheduler.
    """
    arguments = sys.argv[1:] if argv is None else argv
    if "--distributed" in arguments:
        return True
    if "--single-node" in arguments:
        return False
    return os.environ.get("BATCHER_EXAMPLES_DISTRIBUTED", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
