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


# --- the replica shape Carbonite chose ------------------------------------------------------
#
# The advisory used to take a tensor degree and re-derive everything else from it, so the one
# shape a degree cannot express — a model that fits no admissible tensor group and has to be
# cut into pipeline stages — produced no advice at all. `carbonite.accel.plan_parallelism` is
# the decision, and these pin that its answer reaches the user rather than a report.


def _plan(**kw):
    from batcher.carbonite.accel.parallelism import ParallelPlan

    defaults = {
        "tensor_parallel": 2,
        "pipeline_parallel": 1,
        "replicas": 0,
        "weight_bytes_per_device": 1,
        "bytes_per_token_per_device": 1,
        "allreduce_bytes_per_token": 1,
    }
    return ParallelPlan(**{**defaults, **kw})


def _message(**kw) -> str:
    """The advisory's message, or `""`. Clears the once-per-process flag so a test may ask
    twice and compare the two answers, which is what makes a control possible here at all."""
    import warnings

    from batcher._internal.errors import PerformanceWarning
    from batcher.ml.llm.engines import parallelism

    parallelism._TP_WARNED = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PerformanceWarning)
        _warn(**kw)
    return str(caught[0].message) if caught else ""


def test_a_model_that_fits_no_tensor_group_is_told_what_the_pipeline_costs():
    """The case a degree cannot carry: there is no setting to change, only a floor to plan for."""
    message = _message(declared=1, plan=_plan(tensor_parallel=0, pipeline_parallel=4))
    assert "4 pipeline stages" in message
    assert "%" in message, "the bubble is quoted as a share of the stage's time"


def test_a_plan_that_needed_no_pipeline_says_nothing_about_one():
    """The positive control: the branch is keyed on the plan's shape, not on having a plan."""
    assert "pipeline" not in _message(declared=1, plan=_plan(pipeline_parallel=1))


def test_the_replica_count_is_divided_by_the_whole_replica_not_its_tensor_degree():
    """A replica spanning pipeline stages occupies `tensor x pipeline` devices.

    Dividing the budget by the tensor degree alone claims replicas the stage cannot place —
    eight devices over a 2x2 replica is two, not four.
    """
    crowded = _message(declared=8, needed=2, plan=_plan(tensor_parallel=2, pipeline_parallel=2))
    assert "would run 2 replicas" in crowded
    flat = _message(declared=8, needed=2, plan=_plan(tensor_parallel=2, pipeline_parallel=1))
    assert "would run 4 replicas" in flat


def test_the_advisory_without_a_plan_behaves_exactly_as_it_did():
    """Every existing caller passes no plan, and must see the message it saw before."""
    assert "would run 4 replicas" in _message(declared=8, needed=2)
    assert "pipeline" not in _message(declared=1)


# --- what the all-reduce actually costs on this node -----------------------------------------
#
# The PCIe advisory quoted "30-50%" from the field guides. That is a range across hardware, so
# on any particular node it stands in for a number rather than being one -- while every term
# needed to compute the real figure was already measured and sitting in three modules that
# nothing joined up.


def _wired(degree: int, cls: str) -> list[list[str]]:
    """A described node: every pair of `degree` devices connected by `cls`."""
    return [["self" if i == j else cls for j in range(degree)] for i in range(degree)]


def _cost(
    model: str | None,
    degree: int = 4,
    bytes_per_token: int = 10_000_000,
    cls: str = "nvlink",
) -> float:
    from batcher.ml.llm.engines.parallelism import allreduce_ms_per_token

    return allreduce_ms_per_token(
        _plan(allreduce_bytes_per_token=bytes_per_token), degree, model, _wired(degree, cls)
    )


def test_the_fabric_decides_the_figure_not_the_field_guide():
    """The same four A100s cost two orders of magnitude more across the socket than on NVLink.

    This is the control for every assertion below: if the peer matrix and the card's link
    rates did not reach the ring, these would return the same number and the wire would be
    doing nothing. Measured here: 0.025 ms on NVLink against 2.4 ms across the socket.
    """
    on_fabric = _cost("NVIDIA_A100", cls="nvlink")
    on_the_bus = _cost("NVIDIA_A100", cls="sys")
    assert on_fabric > 0.0 and on_the_bus > 0.0
    assert on_the_bus > 50 * on_fabric, f"nvlink={on_fabric} sys={on_the_bus}"


def test_a_card_with_no_published_fabric_rate_refuses_to_price_a_fabric_hop():
    """An L4 has no NVLink, so a matrix claiming an NVLink hop is a contradiction.

    `peer_bandwidth_gbps` answers `0.0` there -- "no opinion" -- rather than substituting the
    bus rate, which would be quietly overruling the topology with the datasheet. The advisory
    then says nothing, which is the right outcome for a reading that cannot be trusted. The
    same card on a bus hop prices normally, which is what makes this a refusal and not a gap.
    """
    assert _cost("NVIDIA_L4", cls="nvlink") == 0.0
    assert _cost("NVIDIA_L4", cls="pix") > 0.0


def test_a_faster_fabric_costs_strictly_less():
    assert _cost("NVIDIA_H100") < _cost("NVIDIA_A100")


def test_an_unrecognized_card_yields_no_opinion_rather_than_a_default():
    """A figure derived from a default link rate is worse than saying nothing."""
    assert _cost(None) == 0.0
    assert _cost("") == 0.0
    assert _cost("Some Future GPU") == 0.0


def test_a_degree_below_two_and_a_plan_that_does_not_all_reduce_both_cost_nothing():
    assert _cost("NVIDIA_A100", degree=1) == 0.0
    assert _cost("NVIDIA_A100", bytes_per_token=0) == 0.0

    from batcher.ml.llm.engines.parallelism import allreduce_ms_per_token

    assert allreduce_ms_per_token(None, 4, "NVIDIA_A100") == 0.0


def test_more_bytes_per_token_cost_proportionally_more():
    """The ring bound is linear in the buffer, so the figure has to track it."""
    one = _cost("NVIDIA_A100", bytes_per_token=10_000_000)
    two = _cost("NVIDIA_A100", bytes_per_token=20_000_000)
    assert two == pytest.approx(2 * one)


def test_the_pcie_advisory_carries_the_computed_figure():
    """End to end: the number reaches the message a user actually reads."""
    message = _message(
        declared=2,
        model_gb=20.0,  # fits a 2-way group, so the "too low" branch does not fire
        vram_gb=24.0,
        device_name="NVIDIA L4",
        needed=2,
        plan=_plan(allreduce_bytes_per_token=10_000_000),
    )
    assert "ms per token" in message, message


def test_an_nvlink_card_is_not_given_the_pcie_lecture_at_all():
    """The positive control for the branch: the figure is attached to the PCIe advisory, and
    an NVLink card does not reach it."""
    assert "ms per token" not in _message(declared=2, device_name="NVIDIA A100-SXM4-80GB")
