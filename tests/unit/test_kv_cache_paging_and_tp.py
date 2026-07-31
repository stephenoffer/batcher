"""KV-cache sizing that matches what a serving engine actually charges for.

Two corrections to an arithmetic that was otherwise exact, and both of them decide whether a
booked GPU group runs at the concurrency it was sized for:

* **Paged attention allocates whole blocks.** A sequence holds the tokens its blocks cover,
  not the tokens it has. On a batch-inference workload of short rows that is most of a block
  per sequence, so a stage sized on raw token counts admits sequences the device cannot hold
  and spends the run preempting and recomputing.
* **A tensor-parallel group divides both halves of the footprint.** Sizing a TP=4 engine
  against one device's arithmetic understates its concurrency fourfold, and a `max_num_seqs`
  set from that number leaves three quarters of the group idle.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.accel.kv_cache import (
    DEFAULT_BLOCK_TOKENS,
    KvCacheBudget,
    kv_bytes_per_token,
    paged_tokens,
)
from batcher.ml.llm.sizing import kv_cache_concurrency

pytestmark = pytest.mark.unit

GIB = 1 << 30


def test_a_sequence_is_charged_for_whole_blocks() -> None:
    assert paged_tokens(20) == 32
    assert paged_tokens(1) == DEFAULT_BLOCK_TOKENS
    assert paged_tokens(16) == 16
    assert paged_tokens(8192) == 8192


def test_paging_can_be_switched_off_for_an_engine_that_does_not_page() -> None:
    assert paged_tokens(20, block_tokens=0) == 20
    assert paged_tokens(20, block_tokens=1) == 20


def test_paging_is_not_applied_to_nothing() -> None:
    assert paged_tokens(0) == 0
    assert paged_tokens(-5) == 0


def _budget(context: int, **kwargs) -> KvCacheBudget:
    return KvCacheBudget(
        device_bytes=80 * GIB,
        weight_bytes=16 * GIB,
        bytes_per_token=kv_bytes_per_token(32, 8, 128, "fp16"),
        context_tokens=context,
        **kwargs,
    )


def test_a_short_prompt_workload_holds_more_cache_than_its_tokens_suggest() -> None:
    # 20 tokens is charged as 32, so the honest concurrency is materially below the naive one.
    paged = _budget(20).max_sequences
    unpaged = _budget(20, block_tokens=0).max_sequences
    assert paged < unpaged
    assert paged == _budget(32).max_sequences


def test_a_block_aligned_context_is_unaffected() -> None:
    # The correction must not move the common case, where every context length in use is
    # already a power of two well above the block size.
    assert _budget(8192).max_sequences == _budget(8192, block_tokens=0).max_sequences


def test_sequences_at_pages_the_length_it_is_asked_about() -> None:
    budget = _budget(8192)
    assert budget.sequences_at(20) == budget.sequences_at(32)
    assert budget.sequences_at(20) > budget.sequences_at(8192)


def test_tensor_parallelism_raises_concurrency_more_than_proportionally() -> None:
    # Weights shrink as well as cache, so doubling the group more than doubles what fits.
    base = {
        "context_tokens": 8192,
        "layers": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "weight_bytes": 16 * GIB,
        "device_bytes": 80 * GIB,
        "dtype": "fp16",
    }
    one = kv_cache_concurrency(**base, tensor_parallel=1)
    two = kv_cache_concurrency(**base, tensor_parallel=2)
    four = kv_cache_concurrency(**base, tensor_parallel=4)
    assert one < two < four
    assert two > 2 * one


def test_a_degree_of_one_is_the_unchanged_single_device_answer() -> None:
    base = {
        "context_tokens": 8192,
        "layers": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "weight_bytes": 16 * GIB,
        "device_bytes": 80 * GIB,
        "dtype": "fp16",
    }
    assert kv_cache_concurrency(**base) == kv_cache_concurrency(**base, tensor_parallel=1)
    assert kv_cache_concurrency(**base, tensor_parallel=0) == kv_cache_concurrency(**base)


def test_a_group_that_still_cannot_hold_the_weights_reports_nothing() -> None:
    assert (
        kv_cache_concurrency(
            context_tokens=8192,
            layers=80,
            kv_heads=8,
            head_dim=128,
            weight_bytes=1400 * GIB,
            device_bytes=80 * GIB,
            dtype="fp16",
            tensor_parallel=2,
        )
        == 0
    )
