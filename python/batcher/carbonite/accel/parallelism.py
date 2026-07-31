"""Sharding one model across devices: what each device then holds, and what it costs.

`kv_cache` sizes a stage that fits on **one** device. The models a GPU cluster is bought for
do not, and the answer is to split the model rather than the batch: tensor parallelism divides
every layer's weights across a group of devices, pipeline parallelism gives each device a
contiguous run of layers, and data parallelism replicates whichever unit fits.

Which of the three to reach for is a resource decision, not a modelling one, and it is made
from arithmetic this module owns:

* **Tensor parallelism divides both halves of the footprint.** Weights *and* KV cache shrink
  by the degree, which is what makes it the lever that turns "does not load" into "loads". It
  pays for that with an all-reduce on every layer of every token, so the group must sit inside
  one NVLink domain to be worth having (`fabric`).
* **Pipeline parallelism divides the weights only.** Each stage keeps its own layers' cache in
  full for the sequences in flight, and the cross-stage traffic is one activation per
  micro-batch rather than per layer — cheap enough to cross a node boundary, which tensor
  parallelism is not. It pays with the bubble: stages idle while the pipeline fills and drains.
* **Data parallelism costs nothing and divides nothing.** It is what the devices left over
  after one replica fits are for.

Nothing here allocates, launches, or communicates. It answers "how small a group can hold this
model", "what does each device end up holding", and "what does the group pay per token", and
the serving engine is told the answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.errors import ResourceError

__all__ = [
    "MAX_TENSOR_DEGREE",
    "ParallelPlan",
    "allreduce_bytes_per_token",
    "minimum_tensor_degree",
    "pipeline_bubble_fraction",
    "plan_parallelism",
    "replicas_for_devices",
    "shard_bytes_per_token",
    "shard_weight_bytes",
    "valid_tensor_degrees",
]

#: Largest tensor-parallel degree worth considering. Beyond eight a group leaves the NVLink
#: domain on every fleet that has one, and the all-reduce on every layer stops being amortized
#: by the compute it interrupts. A model that needs more than this wants pipeline stages.
MAX_TENSOR_DEGREE = 8


def valid_tensor_degrees(
    attention_heads: int, kv_heads: int = 0, max_degree: int = MAX_TENSOR_DEGREE
) -> tuple[int, ...]:
    """Tensor-parallel degrees this model's head counts actually admit.

    A tensor-parallel group splits attention heads evenly across its devices, so a degree that
    does not divide the head count is not a configuration the engine can build — it fails at
    load with a shape error, after the weights have been downloaded and the cluster booked.
    Grouped-query attention adds the tighter constraint: the *key/value* heads are the ones
    that shard, and a model with 8 of them cannot spread over 16 devices without replicating
    them, which no engine does by default.

    Args:
        attention_heads: Attention heads per layer.
        kv_heads: Key/value heads per layer under GQA/MQA, or `0` when the model is plain
            multi-head attention and the two counts are the same.
        max_degree: Largest degree to consider.

    Returns:
        The admissible degrees in increasing order, always including `1`. Empty inputs give
        `(1,)` rather than a guess, because a degree chosen from an unknown head count is how
        a load fails after the download.

    Examples:
        .. doctest::

            >>> from batcher.carbonite.accel.parallelism import valid_tensor_degrees
            >>> valid_tensor_degrees(64, 8)
            (1, 2, 4, 8)
            >>> valid_tensor_degrees(32, 32, max_degree=4)
            (1, 2, 4)
    """
    if attention_heads <= 0 or max_degree < 1:
        return (1,)
    shardable = kv_heads if kv_heads > 0 else attention_heads
    return tuple(
        d
        for d in range(1, min(max_degree, attention_heads) + 1)
        if attention_heads % d == 0 and shardable % d == 0
    )


def shard_weight_bytes(weight_bytes: int, tensor_parallel: int, pipeline_parallel: int = 1) -> int:
    """Resident weight bytes on one device of a `tensor_parallel` x `pipeline_parallel` group.

    Both degrees divide the weights, and they compose: a 16-device group at TP=4, PP=4 holds a
    sixteenth of the model per device. The division is exact for the layer stacks that dominate
    a transformer's parameter count; embeddings and norms replicate, which is a percent or two
    and is left to the headroom that every device already reserves.

    Args:
        weight_bytes: The whole model's resident weights, after quantization.
        tensor_parallel: Devices each layer is split across.
        pipeline_parallel: Contiguous layer groups the model is cut into.

    Returns:
        Bytes one device holds, `0` for a non-positive footprint.
    """
    if weight_bytes <= 0:
        return 0
    devices = max(1, tensor_parallel) * max(1, pipeline_parallel)
    return weight_bytes // devices


def shard_bytes_per_token(bytes_per_token: int, tensor_parallel: int) -> int:
    """KV-cache bytes one device holds per token, under tensor parallelism.

    Tensor parallelism shards the key/value heads, so it divides the cache by the same degree
    it divides the weights — the reason it is the lever that makes a long context fit, and the
    reason pipeline parallelism is not. A pipeline stage holds *fewer layers* of cache but for
    every sequence in flight, so its per-token cost is already accounted for by giving it its
    own layer count rather than by dividing here.

    Args:
        bytes_per_token: The whole model's per-token cache cost, from `kv_bytes_per_token`.
        tensor_parallel: The group's tensor-parallel degree.

    Returns:
        Per-token bytes on one device, `0` for a non-positive input.
    """
    if bytes_per_token <= 0:
        return 0
    return bytes_per_token // max(1, tensor_parallel)


def minimum_tensor_degree(
    weight_bytes: int,
    usable_bytes: int,
    *,
    bytes_per_token: int = 0,
    context_tokens: int = 0,
    attention_heads: int = 0,
    kv_heads: int = 0,
    max_degree: int = MAX_TENSOR_DEGREE,
) -> int:
    """The smallest admissible tensor-parallel degree that holds weights *and* one sequence.

    Sizing on weights alone is the mistake this exists to prevent: a degree chosen so the
    weights just fit leaves no cache, and a serving engine with no cache does not fail — it
    admits one sequence, preempts it, recomputes it, and reports a throughput a third of what
    the hardware can do, with nothing in any log to say why. So "fits" here means the weights
    plus one sequence at the full context the stage must support.

    Args:
        weight_bytes: The model's resident weights, after quantization.
        usable_bytes: One device's memory available to the stage, after headroom.
        bytes_per_token: Whole-model per-token cache cost; `0` sizes on weights alone.
        context_tokens: Peak context length one sequence must reach; `0` sizes on weights alone.
        attention_heads: Attention heads per layer, constraining which degrees are admissible.
        kv_heads: Key/value heads per layer under GQA.
        max_degree: Largest degree to consider.

    Returns:
        The smallest admissible degree that fits, or `0` when none does — which means the model
        needs pipeline stages, a larger device, or a shorter context, and is worth saying
        rather than rounding up to a degree that will also fail.

    Examples:
        .. doctest::

            >>> from batcher.carbonite.accel.parallelism import minimum_tensor_degree
            >>> minimum_tensor_degree(140 * 1 << 30, 80 * 1 << 30, attention_heads=64, kv_heads=8)
            2
    """
    if weight_bytes <= 0 or usable_bytes <= 0:
        return 1
    per_sequence = max(0, bytes_per_token) * max(0, context_tokens)
    for degree in valid_tensor_degrees(attention_heads, kv_heads, max_degree):
        needed = shard_weight_bytes(weight_bytes, degree) + (
            shard_bytes_per_token(per_sequence, degree) if per_sequence else 0
        )
        if needed <= usable_bytes:
            return degree
    return 0


def allreduce_bytes_per_token(
    hidden_size: int, layers: int, tensor_parallel: int, dtype_bytes: int = 2
) -> int:
    """Bytes one token moves across the tensor-parallel group's links, per device.

    Two all-reduces per layer — one after attention's output projection, one after the MLP's —
    each over a hidden-size activation vector. A ring all-reduce moves `2 * (n - 1) / n` of the
    payload per device, which is why the cost climbs steeply from one device to two and barely
    at all from four to eight: the *link rate* is what decides whether the group is viable, not
    the degree.

    Args:
        hidden_size: The model's hidden dimension.
        layers: Transformer layers in the group's share of the model.
        tensor_parallel: The group's degree; `1` communicates nothing.
        dtype_bytes: Width of the activation element, 2 for fp16/bf16.

    Returns:
        Bytes one device sends and receives per token, `0` at degree 1 or for a degenerate
        model shape.

    Examples:
        .. doctest::

            >>> from batcher.carbonite.accel.parallelism import allreduce_bytes_per_token
            >>> allreduce_bytes_per_token(8192, 80, 1)
            0
    """
    degree = max(1, tensor_parallel)
    if degree == 1 or min(hidden_size, layers, dtype_bytes) <= 0:
        return 0
    payload = 2 * layers * hidden_size * dtype_bytes
    return int(payload * 2 * (degree - 1) / degree)


def pipeline_bubble_fraction(stages: int, microbatches: int) -> float:
    """Share of a pipeline step every stage spends idle, in [0, 1).

    A pipeline with `s` stages needs `s - 1` steps to fill and the same to drain, and every
    device is idle for those. Splitting a batch into more micro-batches amortizes the fill over
    more useful steps, which is the only lever: `(s - 1) / (m + s - 1)`. Four stages fed a
    single micro-batch waste three quarters of the cluster; the same four fed thirty-two waste
    under a tenth.

    Args:
        stages: Pipeline stages, `1` for no pipeline.
        microbatches: Micro-batches per step.

    Returns:
        The idle fraction, `0.0` for a single stage.

    Examples:
        .. doctest::

            >>> from batcher.carbonite.accel.parallelism import pipeline_bubble_fraction
            >>> round(pipeline_bubble_fraction(4, 32), 3)
            0.086
    """
    if stages <= 1 or microbatches <= 0:
        return 0.0
    return (stages - 1) / (microbatches + stages - 1)


def replicas_for_devices(devices: int, devices_per_replica: int) -> int:
    """Whole replicas that fit in a device budget.

    A partial replica is not a replica: half of a tensor-parallel group holds half a model and
    serves nothing, so the devices that cannot complete one are better left free for another
    tenant than booked into a group that will never start.

    Args:
        devices: Devices available to the stage.
        devices_per_replica: Devices one replica occupies.

    Returns:
        Whole replicas, `0` when not even one fits.
    """
    if devices <= 0 or devices_per_replica <= 0:
        return 0
    return devices // devices_per_replica


@dataclass(frozen=True, slots=True)
class ParallelPlan:
    """How one model is spread over a device budget, and what that costs per token.

    Attributes:
        tensor_parallel: Devices each layer's weights are split across.
        pipeline_parallel: Contiguous layer groups the model is cut into.
        replicas: Independent copies of that group, each serving its own sequences.
        weight_bytes_per_device: Resident weights one device holds.
        bytes_per_token_per_device: KV-cache cost of one token on one device.
        allreduce_bytes_per_token: Bytes one device moves across the tensor-parallel links
            per token; `0` when the plan does not use tensor parallelism.
    """

    tensor_parallel: int
    pipeline_parallel: int
    replicas: int
    weight_bytes_per_device: int
    bytes_per_token_per_device: int
    allreduce_bytes_per_token: int

    @property
    def devices_per_replica(self) -> int:
        """Devices one replica occupies."""
        return max(1, self.tensor_parallel) * max(1, self.pipeline_parallel)

    @property
    def devices(self) -> int:
        """Devices the whole plan occupies."""
        return self.devices_per_replica * max(0, self.replicas)

    def tensor_group_fits_node(self, devices_per_node: int) -> bool:
        """Whether one tensor-parallel group sits inside a single node.

        The distinction that decides whether a plan is viable at all. A tensor-parallel group
        that spans nodes all-reduces over the network on every layer of every token and is
        almost always the wrong shape; a *pipeline* that spans nodes moves one activation per
        micro-batch and is routinely correct. So this asks about the tensor degree only.

        Args:
            devices_per_node: Devices on one node — the NVLink domain's width.

        Returns:
            True when the group fits, and when the node width is unknown (`0` or less), where
            reporting a violation would be inventing one.
        """
        return devices_per_node <= 0 or self.tensor_parallel <= devices_per_node

    def summary(self) -> dict[str, float]:
        """A flat roll-up of the plan, for the decision log and the dashboard."""
        return {
            "tensor_parallel": float(self.tensor_parallel),
            "pipeline_parallel": float(self.pipeline_parallel),
            "replicas": float(self.replicas),
            "devices": float(self.devices),
            "weight_bytes_per_device": float(self.weight_bytes_per_device),
            "bytes_per_token_per_device": float(self.bytes_per_token_per_device),
            "allreduce_bytes_per_token": float(self.allreduce_bytes_per_token),
        }


def plan_parallelism(
    *,
    weight_bytes: int,
    usable_bytes: int,
    devices: int,
    bytes_per_token: int = 0,
    context_tokens: int = 0,
    attention_heads: int = 0,
    kv_heads: int = 0,
    hidden_size: int = 0,
    layers: int = 0,
    max_tensor_degree: int = MAX_TENSOR_DEGREE,
) -> ParallelPlan:
    """Choose the smallest replica that holds the model, then fill the budget with replicas.

    The ordering is deliberate and is the whole decision: **make one replica as small as it can
    be, then replicate**. A larger tensor-parallel group than the model needs costs an
    all-reduce on every layer for memory nobody uses, and it costs it on every token for the
    life of the stage — while the replicas it displaced would each have served their own
    sequences at full rate. Pipeline stages enter only when tensor parallelism has run out of
    admissible degrees, because their bubble is a throughput loss no batching recovers.

    Args:
        weight_bytes: The model's resident weights, after quantization.
        usable_bytes: One device's memory available to the stage, after headroom.
        devices: Devices the stage may occupy.
        bytes_per_token: Whole-model per-token cache cost, from `kv_bytes_per_token`.
        context_tokens: Peak context length one sequence must reach.
        attention_heads: Attention heads per layer.
        kv_heads: Key/value heads per layer under GQA.
        hidden_size: Hidden dimension, for the all-reduce estimate.
        layers: Transformer layers, for the all-reduce estimate.
        max_tensor_degree: Largest tensor-parallel degree to consider.

    Returns:
        The plan. `replicas` is `0` when the budget cannot hold even one replica, which is a
        refusal rather than a plan and is what the caller should surface.

    Raises:
        ResourceError: When the model cannot be placed on `devices` at any admissible degree —
            the one case where returning a plan would mean returning a shape that cannot run.

    Examples:
        .. doctest::

            >>> from batcher.carbonite.accel import plan_parallelism
            >>> plan = plan_parallelism(
            ...     weight_bytes=140 * (1 << 30),
            ...     usable_bytes=80 * (1 << 30),
            ...     devices=8,
            ...     attention_heads=64,
            ...     kv_heads=8,
            ... )
            >>> plan.tensor_parallel, plan.replicas
            (2, 4)
    """
    budget = max(0, devices)
    tensor = minimum_tensor_degree(
        weight_bytes,
        usable_bytes,
        bytes_per_token=bytes_per_token,
        context_tokens=context_tokens,
        attention_heads=attention_heads,
        kv_heads=kv_heads,
        max_degree=min(max_tensor_degree, budget) if budget else max_tensor_degree,
    )
    pipeline = 1
    if tensor == 0:
        # Tensor parallelism ran out of admissible degrees. Pipeline stages divide the weights
        # again without touching the head counts, so they are what is left — sized to the
        # widest admissible tensor group so the stages stay as few as possible.
        # Reaching here means a positive footprint did not fit a positive device, so both
        # figures are known to be usable and the division below is safe.
        tensor = max(valid_tensor_degrees(attention_heads, kv_heads, max_tensor_degree))
        pipeline = -(-shard_weight_bytes(weight_bytes, tensor) // usable_bytes)
        if budget and tensor * pipeline > budget:
            raise ResourceError(
                f"model needs {tensor * pipeline} devices "
                f"(tensor_parallel={tensor}, pipeline_parallel={pipeline}) but only "
                f"{budget} are available; quantize the weights, shorten the context, or "
                f"add devices"
            )
    per_replica = tensor * pipeline
    return ParallelPlan(
        tensor_parallel=tensor,
        pipeline_parallel=pipeline,
        replicas=replicas_for_devices(budget, per_replica),
        weight_bytes_per_device=shard_weight_bytes(weight_bytes, tensor, pipeline),
        bytes_per_token_per_device=shard_bytes_per_token(bytes_per_token, tensor),
        allreduce_bytes_per_token=allreduce_bytes_per_token(
            hidden_size, -(-layers // pipeline) if layers > 0 else 0, tensor
        ),
    )
