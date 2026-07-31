"""Sizing an LLM engine from the workload instead of from the model's maximum.

A vLLM engine reserves its KV cache from `max_model_len`, and the default is the model's
full context window — 128K tokens for a Llama 3.1. KV cache scales linearly with that
number, and every token of cache reserved for a length the data never reaches is a
concurrent sequence the scheduler cannot run. The field guides measure the gap at 2-10x
throughput and prescribe a manual procedure: sample the corpus, compute the P95/P99 of
prompt length, set `max_model_len` to it.

Batcher is a data engine, so it sees the prompts. This module turns "what window does this
workload actually need" into arithmetic over observed prompt sizes.

**The sizing may only ever be generous.** A window smaller than a prompt means that prompt
is truncated, which silently degrades output — strictly worse than a large KV cache. So the
estimate converts characters to tokens at a deliberately pessimistic rate, adds the full
generation budget, applies headroom, and rounds *up* to a bucket. Being wrong in the safe
direction costs some cache; being wrong in the other direction costs correctness.

The other half of the same concern lives here too: whatever window an engine ends up with,
a prompt that overflows it has to be **made to fit**, and the fitting has to use the model's
own tokenizer. Choosing the window and honoring it are one decision with two ends, and
splitting them across modules is how a window gets chosen that nothing enforces.
"""

from __future__ import annotations

__all__ = [
    "auto_max_model_len",
    "estimate_tokens",
    "fit_to_window",
    "kv_cache_concurrency",
    "prompt_window",
]

#: Tokens per character, deliberately **over**-estimated. English text averages ~4 chars per
#: token (0.25); code, non-Latin scripts, and heavy punctuation run denser. 0.5 assumes two
#: characters per token, which no realistic tokenizer beats, so the token estimate is an
#: upper bound rather than a guess that can come in low.
_TOKENS_PER_CHAR = 0.5
#: Multiplier on the largest prompt observed, covering prompts the sample never saw. The
#: sample is one batch of a corpus, so the true maximum can exceed it.
_HEADROOM = 1.5
#: Never propose a window below this: a tiny sample must not produce a window that a
#: slightly longer prompt immediately overflows.
_MIN_WINDOW = 2048
#: Windows are rounded up to a multiple of this, so a workload whose lengths drift slightly
#: between runs keeps proposing the same number instead of rebuilding the cache each time.
_BUCKET = 1024


def estimate_tokens(chars: int) -> int:
    """An **upper bound** on the tokens `chars` characters can encode to.

    Args:
        chars: Character count of the text.

    Returns:
        A token count no realistic tokenizer should exceed.

    Examples:
        .. doctest::

            >>> from batcher.ml.llm.sizing import estimate_tokens
            >>> estimate_tokens(1000)
            500
    """
    return int(max(0, chars) * _TOKENS_PER_CHAR)


def auto_max_model_len(
    max_prompt_chars: int,
    *,
    max_gen_tokens: int = 0,
    model_default: int | None = None,
) -> int | None:
    """The context window this workload needs, or `None` to leave the model's default.

    The window must hold the longest prompt *plus* everything generated from it, so the
    generation budget is added rather than assumed to fit. Headroom then covers prompts
    longer than the ones observed, and the result is rounded up to a bucket so a workload
    whose lengths drift slightly keeps asking for the same window.

    Returns `None` when the proposal would not actually save anything — when it meets or
    exceeds `model_default`, there is no cache to reclaim and the default is left alone
    rather than pinned to a number that only looks deliberate.

    Args:
        max_prompt_chars: The longest prompt observed, in characters.
        max_gen_tokens: The largest number of tokens generation may add to that prompt.
        model_default: The model's own maximum window, when known. The proposal is never
            allowed above it — a window larger than the model supports is an error, not a
            slower run.

    Returns:
        A context window in tokens, or `None` to leave `max_model_len` unset.

    Examples:
        .. doctest::

            >>> from batcher.ml.llm.sizing import auto_max_model_len
            >>> auto_max_model_len(4000, max_gen_tokens=256, model_default=131072)
            4096
            >>> auto_max_model_len(4000, max_gen_tokens=256, model_default=4096) is None
            True
    """
    if max_prompt_chars <= 0:
        return None
    needed = estimate_tokens(max_prompt_chars) * _HEADROOM + max(0, max_gen_tokens)
    window = max(_MIN_WINDOW, _round_up(int(needed)))
    if model_default is not None:
        if window >= model_default:
            return None  # nothing to reclaim; leave the model's own window alone
        return window
    return window


def _round_up(value: int) -> int:
    """`value` rounded up to the next `_BUCKET` multiple."""
    return -(-value // _BUCKET) * _BUCKET


def sized_window(prompts: list, sampling_kwargs: dict) -> int | None:
    """The `max_model_len` this batch's prompts call for, or `None` to leave it unset.

    Sizing is by characters rather than tokens because the tokenizer lives inside the engine
    this is choosing the shape of. `sizing.estimate_tokens` is deliberately an upper bound,
    so the character route can only over-reserve — and over-reserving costs cache, while
    under-reserving would truncate a prompt.
    """
    longest = max((len(prompt_text(p)) for p in prompts), default=0)
    budget = sampling_kwargs.get("max_tokens")
    return auto_max_model_len(longest, max_gen_tokens=int(budget) if budget else 0)


def prompt_text(prompt: object) -> str:
    """The text of a request, whether it is a bare string or a ``{prompt, image}`` dict."""
    if isinstance(prompt, dict):
        return str(prompt.get("prompt", ""))
    return str(prompt)


def prompt_window(llm: object, engine_kwargs: dict) -> int | None:
    """The token budget a prompt must fit in, or `None` when it cannot be determined.

    Prefers the explicit `max_model_len`, else asks the live vLLM config. Reserves a
    slice of the window for the generation itself: filling the whole context with prompt
    leaves no room to decode, which fails just as hard as an over-long prompt.
    """
    declared = engine_kwargs.get("max_model_len")
    if not isinstance(declared, int):
        config = getattr(getattr(llm, "llm_engine", None), "model_config", None)
        declared = getattr(config, "max_model_len", None)
    if not isinstance(declared, int) or declared <= 0:
        return None
    return max(1, declared - _RESERVED_OUTPUT_TOKENS)


#: Tokens held back from the context window for the generation itself.
_RESERVED_OUTPUT_TOKENS = 512


def fit_to_window(prompts: list, tokenizer: object | None, window: int | None) -> list:
    """Truncate any over-length prompt to `window` tokens, leaving the rest untouched.

    A single over-length row used to fail the **whole** request, losing a batch's worth
    of GPU work to one bad input. Windowing keeps the batch alive and warns, so the
    truncation is visible rather than silent. A no-op when no tokenizer or window is
    available — see `_worker_tokenizer` for why guessing is worse than not truncating.
    """
    if tokenizer is None or window is None:
        return prompts
    texts = [p["prompt"] if isinstance(p, dict) else p for p in prompts]
    fitted = _truncate_to_window(texts, tokenizer, window)
    return [
        {**p, "prompt": text} if isinstance(p, dict) else text
        for p, text in zip(prompts, fitted, strict=True)
    ]


def _truncate_to_window(prompts: list, tokenizer: object, max_tokens: int) -> list:
    """Each prompt cut to its first `max_tokens` tokens, warning once if any was cut.

    Keeps the **head** of the prompt: an instruction-shaped prompt puts the task up
    front, so a tail cut is likelier to remove the answer's context than the question.
    """
    out = []
    truncated = 0
    for prompt in prompts:
        ids = tokenizer.encode(str(prompt))
        if len(ids) <= max_tokens:
            out.append(prompt)
            continue
        truncated += 1
        out.append(tokenizer.decode(ids[:max_tokens]))
    if truncated:
        import warnings

        warnings.warn(
            f"{truncated} of {len(prompts)} prompts exceeded the model's context window "
            f"and were truncated to {max_tokens} tokens. Shorten the prompts, or raise "
            "vllm_engine(max_model_len=...), to avoid losing their tails.",
            UserWarning,
            stacklevel=3,
        )
    return out


def _free_device_bytes() -> int:
    """Device memory the smallest visible GPU actually has free, or `0` when none is visible.

    Nominal capacity is the wrong number on a shared device: sizing a KV cache against total
    VRAM when another process holds half of it is how a serving engine OOMs on its first full
    batch. The pool takes the measured resident figure and reserves against the remainder.
    """
    from batcher._internal.hardware import device_telemetry, gpu_inventory
    from batcher._internal.hardware.nvml import own_device_memory
    from batcher.carbonite.accel import VramPool

    capacity = min((int(g.get("memory_bytes") or 0) for g in gpu_inventory()), default=0)
    if capacity <= 0:
        return 0
    pool = VramPool(capacity_bytes=capacity, headroom=0.0)
    for reading in device_telemetry():
        pool.observe_external(
            reading.index,
            reading.memory_used_bytes,
            own_bytes=own_device_memory(reading.index),
        )
    return min(
        (pool.available_bytes(r.index) for r in device_telemetry()),
        default=pool.available_bytes(0),
    )


def kv_cache_concurrency(
    *,
    context_tokens: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
    weight_bytes: int,
    device_bytes: int | None = None,
    dtype: str | None = None,
) -> int:
    """Concurrent sequences one device can hold at a given context length.

    The other half of `auto_max_model_len`: that chooses the window, and this says what the
    window buys. A serving engine's throughput is set by how many sequences it can keep in
    flight, and that is decided by the key/value cache rather than by the weights — a model
    whose weights fit comfortably can still be limited to a handful of sequences once the
    cache for a long context is reserved.

    Pass the result as an engine's `max_num_seqs` to stop it admitting more sequences than the
    device can hold, which otherwise shows up as preemption and recomputation rather than as
    an error.

    Args:
        context_tokens: Peak tokens cached per sequence, prompt plus generation.
        layers: Transformer layers in the model.
        kv_heads: Key/value heads per layer. Under grouped-query attention this is the
            *grouped* count, well below the attention-head count, and the cache scales with
            it — a model with 8 KV heads against 64 attention heads has an eighth of the cache.
        head_dim: Dimension of one attention head.
        weight_bytes: Resident model weights after any quantization.
        device_bytes: Device memory available to the stage. `None` reads the smallest GPU
            this process can see *and subtracts what other processes already hold on it*,
            because the memory a co-tenant is using is the binding constraint and is invisible
            to a per-process allocator. Reports `0` when no device is visible.
        dtype: Cache element type. `None` reads `accelerator.kv_cache_dtype` from the active
            configuration, where FP8 halves the cache against FP16.

    Returns:
        Sequences that fit, or `0` when the weights alone do not fit or no device is visible.

    Examples:
        .. doctest::

            >>> from batcher.ml.llm.sizing import kv_cache_concurrency
            >>> kv_cache_concurrency(
            ...     context_tokens=8192,
            ...     layers=32,
            ...     kv_heads=8,
            ...     head_dim=128,
            ...     weight_bytes=16 << 30,
            ...     device_bytes=80 << 30,
            ...     dtype="fp16",
            ... )
            56
    """
    from batcher.carbonite.accel import KvCacheBudget, kv_bytes_per_token
    from batcher.config import active_config

    accel = active_config().accelerator
    if device_bytes is None:
        device_bytes = _free_device_bytes()
    per_token = kv_bytes_per_token(layers, kv_heads, head_dim, dtype or accel.kv_cache_dtype)
    budget = KvCacheBudget(
        device_bytes=max(0, device_bytes),
        weight_bytes=max(0, weight_bytes),
        bytes_per_token=per_token,
        context_tokens=max(0, accel.max_context_tokens or context_tokens),
        headroom=accel.kv_cache_headroom,
    )
    return budget.max_sequences
