"""Model-parallel sizing: which degrees are admissible, what fits, and what it costs.

The arithmetic behind `plan_parallelism` decides whether an LLM stage loads at all, so every
edge here is one that shows up as a failed load or a silently-preempting serving engine rather
than as a wrong number: a degree that does not divide the head count, a group sized on weights
with no cache left, and a budget that cannot hold one replica.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import ResourceError
from batcher.carbonite.accel.kv_cache import kv_bytes_per_token
from batcher.carbonite.accel.parallelism import (
    MAX_TENSOR_DEGREE,
    ParallelPlan,
    allreduce_bytes_per_token,
    minimum_tensor_degree,
    pipeline_bubble_fraction,
    plan_parallelism,
    replicas_for_devices,
    shard_bytes_per_token,
    shard_weight_bytes,
    valid_tensor_degrees,
)

GIB = 1 << 30


def test_valid_degrees_respect_the_grouped_kv_head_count() -> None:
    # 64 attention heads would admit 1/2/4/8, but 8 KV heads is the binding constraint under
    # GQA and it is the one an engine actually shards on.
    assert valid_tensor_degrees(64, 8) == (1, 2, 4, 8)
    assert valid_tensor_degrees(64, 6) == (1, 2)
    # Plain multi-head attention: the two counts are the same, so `0` means "use the heads".
    assert valid_tensor_degrees(32, 0, max_degree=4) == valid_tensor_degrees(32, 32, max_degree=4)


def test_valid_degrees_never_returns_empty() -> None:
    # A degree of 1 always works, and an unknown head count must not be turned into a guess.
    assert valid_tensor_degrees(0, 0) == (1,)
    assert valid_tensor_degrees(-4, 8) == (1,)
    assert valid_tensor_degrees(64, 8, max_degree=0) == (1,)
    # A prime head count admits nothing but 1, rather than the nearest power of two.
    assert valid_tensor_degrees(7, 7) == (1, 7)


def test_valid_degrees_never_exceed_the_head_count() -> None:
    assert max(valid_tensor_degrees(2, 2, max_degree=MAX_TENSOR_DEGREE)) == 2


def test_sharding_divides_weights_by_both_degrees() -> None:
    assert shard_weight_bytes(64 * GIB, 4) == 16 * GIB
    assert shard_weight_bytes(64 * GIB, 4, 4) == 4 * GIB
    assert shard_weight_bytes(64 * GIB, 0, 0) == 64 * GIB
    assert shard_weight_bytes(0, 4) == 0


def test_tensor_parallelism_divides_the_cache_and_pipeline_does_not() -> None:
    # The asymmetry is the whole reason to prefer TP for a long context: it is the only degree
    # that shrinks the per-token cache as well as the weights.
    assert shard_bytes_per_token(1024, 4) == 256
    assert shard_bytes_per_token(1024, 1) == 1024
    assert shard_bytes_per_token(0, 4) == 0


def test_minimum_degree_accounts_for_one_full_sequence_not_just_weights() -> None:
    # Llama-70B shaped: 80 layers, 8 KV heads, head_dim 128, fp16 -> 320 KiB/token.
    per_token = kv_bytes_per_token(80, 8, 128, "fp16")
    weights = 140 * GIB
    usable = 80 * GIB
    # On weights alone TP=2 fits (70 GiB of 80). Add a 128k-token sequence at 320 KiB/token
    # (40 GiB whole-model, 20 GiB at TP=2) and it no longer does.
    assert minimum_tensor_degree(weights, usable, attention_heads=64, kv_heads=8) == 2
    assert (
        minimum_tensor_degree(
            weights,
            usable,
            bytes_per_token=per_token,
            context_tokens=131_072,
            attention_heads=64,
            kv_heads=8,
        )
        == 4
    )


def test_minimum_degree_reports_zero_rather_than_a_degree_that_will_not_fit() -> None:
    # Nothing in the admissible set holds it: the caller needs pipeline stages, and rounding
    # up to the largest degree would hand back a shape that fails at load.
    assert minimum_tensor_degree(4000 * GIB, 40 * GIB, attention_heads=64, kv_heads=8) == 0


def test_minimum_degree_is_one_when_there_is_nothing_to_size() -> None:
    assert minimum_tensor_degree(0, 80 * GIB) == 1
    assert minimum_tensor_degree(140 * GIB, 0) == 1


def test_allreduce_is_free_at_degree_one_and_saturates_with_degree() -> None:
    assert allreduce_bytes_per_token(8192, 80, 1) == 0
    two = allreduce_bytes_per_token(8192, 80, 2)
    four = allreduce_bytes_per_token(8192, 80, 4)
    eight = allreduce_bytes_per_token(8192, 80, 8)
    # The ring factor 2(n-1)/n climbs steeply from 1 to 2 and barely at all from 4 to 8.
    assert two < four < eight
    assert eight < 2 * two
    assert allreduce_bytes_per_token(0, 80, 4) == 0


def test_pipeline_bubble_shrinks_with_microbatches() -> None:
    assert pipeline_bubble_fraction(1, 1) == 0.0
    assert pipeline_bubble_fraction(4, 0) == 0.0
    assert pipeline_bubble_fraction(4, 1) == pytest.approx(0.75)
    assert pipeline_bubble_fraction(4, 32) == pytest.approx(3 / 35)
    assert pipeline_bubble_fraction(2, 8) < pipeline_bubble_fraction(8, 8)


def test_partial_replicas_are_not_replicas() -> None:
    assert replicas_for_devices(8, 2) == 4
    assert replicas_for_devices(7, 2) == 3
    assert replicas_for_devices(1, 2) == 0
    assert replicas_for_devices(0, 2) == 0
    assert replicas_for_devices(8, 0) == 0


def test_plan_prefers_the_smallest_replica_then_replicates() -> None:
    plan = plan_parallelism(
        weight_bytes=140 * GIB,
        usable_bytes=80 * GIB,
        devices=8,
        attention_heads=64,
        kv_heads=8,
        hidden_size=8192,
        layers=80,
    )
    assert (plan.tensor_parallel, plan.pipeline_parallel, plan.replicas) == (2, 1, 4)
    assert plan.devices_per_replica == 2
    assert plan.devices == 8
    assert plan.weight_bytes_per_device == 70 * GIB
    assert plan.allreduce_bytes_per_token > 0
    assert plan.tensor_group_fits_node(8)


def test_plan_never_books_a_group_wider_than_the_budget() -> None:
    # Four devices available: a degree of 8 would be admissible on head count alone.
    plan = plan_parallelism(
        weight_bytes=20 * GIB, usable_bytes=80 * GIB, devices=4, attention_heads=64, kv_heads=8
    )
    assert plan.tensor_parallel == 1
    assert plan.replicas == 4


def test_plan_falls_back_to_pipeline_stages_when_no_degree_fits() -> None:
    plan = plan_parallelism(
        weight_bytes=1200 * GIB,
        usable_bytes=80 * GIB,
        devices=32,
        attention_heads=64,
        kv_heads=8,
        hidden_size=16384,
        layers=126,
    )
    assert plan.tensor_parallel == 8
    assert plan.pipeline_parallel >= 2
    assert plan.weight_bytes_per_device <= 80 * GIB
    # The all-reduce is sized on one stage's layers, not the whole model's.
    assert plan.allreduce_bytes_per_token == allreduce_bytes_per_token(
        16384, -(-126 // plan.pipeline_parallel), 8
    )


def test_plan_refuses_rather_than_returning_a_shape_that_cannot_run() -> None:
    with pytest.raises(ResourceError, match="only 4 are available"):
        plan_parallelism(
            weight_bytes=4000 * GIB,
            usable_bytes=40 * GIB,
            devices=4,
            attention_heads=64,
            kv_heads=8,
        )


def test_an_unknown_footprint_plans_one_device_per_replica() -> None:
    # Consistent with `minimum_tensor_degree`, which answers 1 rather than guessing: an
    # unmeasured model is not evidence that it needs a group, and inventing one would book
    # devices no caller asked for.
    plan = plan_parallelism(weight_bytes=0, usable_bytes=0, devices=8, attention_heads=64)
    assert (plan.tensor_parallel, plan.pipeline_parallel, plan.replicas) == (1, 1, 8)


def test_plan_with_no_devices_is_a_refusal_not_a_crash() -> None:
    plan = plan_parallelism(
        weight_bytes=20 * GIB, usable_bytes=80 * GIB, devices=0, attention_heads=64, kv_heads=8
    )
    assert plan.replicas == 0
    assert plan.devices == 0


def test_group_fits_node_is_silent_when_the_node_width_is_unknown() -> None:
    plan = ParallelPlan(
        tensor_parallel=16,
        pipeline_parallel=1,
        replicas=1,
        weight_bytes_per_device=0,
        bytes_per_token_per_device=0,
        allreduce_bytes_per_token=0,
    )
    assert plan.tensor_group_fits_node(0)
    assert not plan.tensor_group_fits_node(8)


def test_summary_is_flat_and_numeric() -> None:
    plan = plan_parallelism(
        weight_bytes=140 * GIB, usable_bytes=80 * GIB, devices=8, attention_heads=64, kv_heads=8
    )
    summary = plan.summary()
    assert set(summary) == {
        "tensor_parallel",
        "pipeline_parallel",
        "replicas",
        "devices",
        "weight_bytes_per_device",
        "bytes_per_token_per_device",
        "allreduce_bytes_per_token",
    }
    assert all(isinstance(v, float) for v in summary.values())
