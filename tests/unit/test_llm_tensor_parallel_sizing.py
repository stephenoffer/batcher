"""Sizing a tensor-parallel group, and knowing when it will hurt.

Tensor parallelism lets a model too large for one card run at all, at the cost of an
all-reduce every forward. The interconnect decides whether that is nearly free or ruinous:
the field guides measure NVLink (600-900 GB/s) as efficient and PCIe Gen4/Gen5 (32-64 GB/s)
as a **30-50% throughput loss at TP>=2** on Llama-70B.

The arithmetic is testable without a GPU; the *choice* is deliberately left to the user,
because the penalty is hardware-specific and a wrong automatic pick would silently halve
throughput on exactly the hardware nobody would think to check.
"""

from __future__ import annotations

import pytest

from batcher.ml.llm.engines.parallelism import minimum_tensor_parallel_size, nvlink_class

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("NVIDIA A100-SXM4-80GB", "nvlink"),
        ("NVIDIA H100 80GB HBM3", "nvlink"),
        ("NVIDIA H200", "nvlink"),
        ("NVIDIA L4", "pcie"),
        ("NVIDIA A10G", "pcie"),
        ("NVIDIA L40S", "pcie"),
        ("NVIDIA T4", "pcie"),
    ],
)
def test_known_cards_are_classified(name: str, expected: str) -> None:
    assert nvlink_class(name) == expected


def test_an_unknown_card_is_not_guessed() -> None:
    """Claiming NVLink on an unrecognized card would advise TP on hardware that may pay the
    full PCIe penalty; claiming PCIe would discourage it wrongly. Neither is safe."""
    assert nvlink_class("Some Future GPU") == "unknown"
    assert nvlink_class(None) == "unknown"
    assert nvlink_class("") == "unknown"


def test_a_model_that_fits_needs_no_tensor_parallelism() -> None:
    """TP=1 is always fastest when the model fits — there is no communication at all."""
    assert minimum_tensor_parallel_size(model_gb=14.0, vram_gb=80.0) == 1


def test_a_model_too_large_for_one_card_gets_a_group() -> None:
    assert minimum_tensor_parallel_size(model_gb=140.0, vram_gb=80.0) == 4


@pytest.mark.parametrize("degree", [1, 2, 4, 8, 16])
def test_the_degree_is_always_a_power_of_two(degree: int) -> None:
    """A TP group splits attention heads evenly, so vLLM needs the head count divisible by
    the degree — 3 GPUs is not a configuration."""
    vram = 80.0
    budget = vram * 0.55
    model = budget * degree * 0.9  # just under what `degree` cards can hold
    got = minimum_tensor_parallel_size(model_gb=model, vram_gb=vram)
    assert got & (got - 1) == 0, f"{got} is not a power of two"


def test_weights_are_not_allowed_to_fill_the_card() -> None:
    """Weights that consume the whole card leave no KV cache, which is the same as not
    fitting — vLLM's own `gpu_memory_utilization` default is 0.90 for this reason."""
    # 70 GB of weights nominally "fits" in 80 GB, but leaves nothing to run with.
    assert minimum_tensor_parallel_size(model_gb=70.0, vram_gb=80.0) > 1


def test_an_unmeasurable_card_does_not_produce_a_confident_answer() -> None:
    """With no VRAM reading there is nothing to divide by, and inventing a group size would
    be worse than leaving the user's own setting alone."""
    assert minimum_tensor_parallel_size(model_gb=140.0, vram_gb=None) == 1
    assert minimum_tensor_parallel_size(model_gb=140.0, vram_gb=0.0) == 1
    assert minimum_tensor_parallel_size(model_gb=0.0, vram_gb=80.0) == 1


def test_the_degree_grows_monotonically_with_model_size() -> None:
    sizes = [10.0, 40.0, 100.0, 300.0, 700.0]
    degrees = [minimum_tensor_parallel_size(m, 80.0) for m in sizes]
    assert degrees == sorted(degrees)


def test_it_terminates_on_an_absurd_model() -> None:
    """A bounded search, so a nonsense footprint cannot hang the caller."""
    assert minimum_tensor_parallel_size(model_gb=1e9, vram_gb=80.0) <= 64


# --- the advisory ---------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_once_flag():
    from batcher.ml.llm.engines import parallelism

    parallelism._TP_WARNED = False
    yield
    parallelism._TP_WARNED = False


def _warn(**kw):
    from batcher.ml.llm.engines.parallelism import warn_about_tensor_parallelism

    defaults = {"declared": 1, "model_gb": 14.0, "vram_gb": 80.0, "device_name": "NVIDIA A100"}
    return warn_about_tensor_parallelism(**{**defaults, **kw})


def test_a_group_too_small_to_hold_the_model_is_called_out():
    """Better said before the weights are downloaded than as an OOM after."""
    from batcher._internal.errors import PerformanceWarning

    with pytest.warns(PerformanceWarning, match="smallest group that fits"):
        _warn(declared=1, model_gb=140.0, vram_gb=80.0)


def test_tp_on_a_pcie_card_reports_the_measured_penalty():
    from batcher._internal.errors import PerformanceWarning

    with pytest.warns(PerformanceWarning, match="30-50%"):
        _warn(declared=2, model_gb=14.0, vram_gb=24.0, device_name="NVIDIA L4")


def test_tp_on_an_nvlink_card_is_silent():
    """The same setting is nearly free here — advising against it would be wrong."""
    import warnings

    from batcher._internal.errors import PerformanceWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        _warn(declared=4, model_gb=140.0, vram_gb=80.0, device_name="NVIDIA A100-SXM4-80GB")


def test_a_fitting_model_at_tp1_is_silent():
    import warnings

    from batcher._internal.errors import PerformanceWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        _warn(declared=1, model_gb=14.0, vram_gb=80.0)


def test_an_unknown_card_is_not_accused():
    """Claiming a PCIe penalty on an unrecognized card would be a guess presented as a
    measurement."""
    import warnings

    from batcher._internal.errors import PerformanceWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        _warn(declared=2, model_gb=14.0, vram_gb=48.0, device_name="Some Future GPU")


def test_nothing_is_said_when_the_model_size_is_unknown():
    import warnings

    from batcher._internal.errors import PerformanceWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        _warn(declared=1, model_gb=0.0, vram_gb=80.0)
