"""KV-cache budgeting — the memory that decides an LLM stage's real throughput.

Batch inference over a language model is not sized by the model's weights. Weights are a fixed
cost paid once; the *variable* cost is the key/value cache, which grows with every token of
every sequence in flight and is what actually runs a device out of memory. A stage told to run
256 concurrent sequences at 8k context on a device that can hold 40 does not run slowly — it
either OOMs on the first full batch or the serving engine silently preempts and recomputes,
which looks like a 3x throughput regression with no error anywhere.

So the budget is a control-plane decision, made before the stage starts, from four facts the
model's configuration already carries: layer count, KV head count, head dimension, and cache
dtype. The formula is exact rather than empirical:

    bytes per token = 2 (K and V) * layers * kv_heads * head_dim * dtype_bytes

Grouped-query attention is why `kv_heads` is separate from attention heads and why it matters
so much: a model with 8 KV heads against 64 attention heads has an eighth of the cache of the
multi-head equivalent, which is often the difference between one device and four.

Nothing here allocates. It answers "how many sequences fit", "how much memory does this
context length cost", and "what is the largest batch this device can sustain", and the serving
engine does the rest.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["KvCacheBudget", "kv_bytes_per_token", "kv_cache_bytes", "max_concurrent_sequences"]

#: Bytes per element for the cache dtypes a serving engine offers. FP8 KV cache halves the
#: cache against FP16 at a small quality cost, and is the single largest lever on concurrency.
_DTYPE_BYTES = {"fp32": 4, "float32": 4, "fp16": 2, "float16": 2, "bf16": 2, "fp8": 1, "int8": 1}


def kv_bytes_per_token(
    layers: int,
    kv_heads: int,
    head_dim: int,
    dtype: str = "fp16",
) -> int:
    """Cache bytes one token of one sequence occupies.

    Args:
        layers: Transformer layers in the model.
        kv_heads: Key/value heads per layer — the *grouped* count under GQA/MQA, which is
            smaller than the attention-head count and is the figure the cache scales with.
        head_dim: Dimension of one head.
        dtype: Cache element type (`fp16`, `bf16`, `fp8`, `int8`, `fp32`).

    Returns:
        Bytes per token, or `0` when any dimension is non-positive or the dtype is unknown.
    """
    width = _DTYPE_BYTES.get(dtype.lower(), 0)
    if min(layers, kv_heads, head_dim, width) <= 0:
        return 0
    return 2 * layers * kv_heads * head_dim * width


def kv_cache_bytes(
    sequences: int,
    context_tokens: int,
    bytes_per_token: int,
) -> int:
    """Cache bytes for a given number of sequences at a given context length.

    Args:
        sequences: Concurrent sequences in flight.
        context_tokens: Tokens cached per sequence — prompt plus generated, at the point of
            peak occupancy, which is the end of generation and not the start.
        bytes_per_token: As reported by `kv_bytes_per_token`.

    Returns:
        Total cache bytes, `0` when any input is non-positive.
    """
    if min(sequences, context_tokens, bytes_per_token) <= 0:
        return 0
    return sequences * context_tokens * bytes_per_token


def max_concurrent_sequences(
    available_bytes: int,
    context_tokens: int,
    bytes_per_token: int,
) -> int:
    """Sequences that fit in a given cache budget at a given context length.

    Args:
        available_bytes: Device memory left for the cache, after weights and headroom.
        context_tokens: Peak tokens cached per sequence.
        bytes_per_token: As reported by `kv_bytes_per_token`.

    Returns:
        Concurrent sequences that fit, `0` when not even one does.
    """
    per_sequence = kv_cache_bytes(1, context_tokens, bytes_per_token)
    if per_sequence <= 0 or available_bytes <= 0:
        return 0
    return int(available_bytes // per_sequence)


@dataclass(frozen=True, slots=True)
class KvCacheBudget:
    """One LLM stage's device-memory plan: weights, cache, and the concurrency they imply.

    Attributes:
        device_bytes: Device memory available to the stage — a whole device, or a MIG
            instance's share.
        weight_bytes: Resident model weights, after any quantization.
        bytes_per_token: Cache cost of one token, from `kv_bytes_per_token`.
        context_tokens: Peak context length the stage must support.
        headroom: Fraction of the device left free for activations, the CUDA context, and
            allocator fragmentation.
    """

    device_bytes: int
    weight_bytes: int
    bytes_per_token: int
    context_tokens: int
    headroom: float = 0.1

    @property
    def usable_bytes(self) -> int:
        """Device memory the stage may actually plan against, after headroom."""
        return int(self.device_bytes * (1.0 - min(0.9, max(0.0, self.headroom))))

    @property
    def cache_bytes(self) -> int:
        """Memory left for the cache once the weights are resident; `0` when they do not fit."""
        return max(0, self.usable_bytes - self.weight_bytes)

    @property
    def fits(self) -> bool:
        """Whether the weights plus at least one full-context sequence fit on the device."""
        return self.max_sequences >= 1

    @property
    def max_sequences(self) -> int:
        """Concurrent sequences the stage can hold at its peak context length."""
        return max_concurrent_sequences(self.cache_bytes, self.context_tokens, self.bytes_per_token)

    @property
    def cache_fraction(self) -> float:
        """Share of usable device memory the cache occupies at full concurrency, in [0, 1].

        Below roughly a third the stage is weight-dominated and wants a smaller device or a
        partition; above two thirds it is cache-dominated, and the levers that matter are the
        cache dtype and the context length rather than the model.
        """
        usable = self.usable_bytes
        return min(1.0, self.cache_bytes / usable) if usable > 0 else 0.0

    def sequences_at(self, context_tokens: int) -> int:
        """Concurrent sequences at a different context length.

        The knob a scheduler actually turns: halving the supported context roughly doubles
        concurrency, and most batch workloads have a long tail of short prompts that never
        needed the maximum.

        Args:
            context_tokens: The context length to size for.

        Returns:
            Sequences that fit at that length.
        """
        return max_concurrent_sequences(self.cache_bytes, context_tokens, self.bytes_per_token)

    def devices_for(self, sequences: int) -> int:
        """Devices needed to hold a target concurrency, replicating weights on each.

        Args:
            sequences: Target concurrent sequences across the stage.

        Returns:
            Device count, or `0` when a single device cannot hold even one sequence (in which
            case the model must be sharded rather than replicated).
        """
        per_device = self.max_sequences
        if per_device <= 0 or sequences <= 0:
            return 0
        return -(-sequences // per_device)

    def summary(self) -> dict[str, float]:
        """A flat roll-up of the plan, for the decision log and the dashboard."""
        return {
            "usable_bytes": float(self.usable_bytes),
            "weight_bytes": float(self.weight_bytes),
            "cache_bytes": float(self.cache_bytes),
            "cache_fraction": self.cache_fraction,
            "bytes_per_token": float(self.bytes_per_token),
            "context_tokens": float(self.context_tokens),
            "max_sequences": float(self.max_sequences),
        }
