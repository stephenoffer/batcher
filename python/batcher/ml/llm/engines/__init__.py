"""LLM engine adapters — the pluggable ``list[str] -> list[str]`` backends.

An *engine* is the only thing `generate` needs from an LLM: hand it a batch of
requests, get back a string per request in order. Keeping that contract this narrow is
what lets vLLM, an OpenAI-compatible HTTP endpoint, and a deterministic test double be
interchangeable — and what keeps the columnar machinery in `generate` free of any
model library.

`base` holds the contract, `vllm` and `sglang` the two GPU-resident offline backends,
`openai` the OpenAI-compatible served-endpoint backend, `anthropic` the Anthropic Messages
API backend, and `hosted` the two large clouds whose wire shape is neither — AWS Bedrock
and Google Gemini. The private helpers are re-exported here because they are the seams the engine
tests drive directly, without a GPU or a network.
"""

from __future__ import annotations

from batcher.ml.llm.engines.anthropic import anthropic_engine
from batcher.ml.llm.engines.base import Engine, EngineFactory
from batcher.ml.llm.engines.hosted import bedrock_engine, gemini_engine
from batcher.ml.llm.engines.openai import (
    _openai_body as _openai_body,
)
from batcher.ml.llm.engines.openai import (
    _openai_finish_reason as _openai_finish_reason,
)
from batcher.ml.llm.engines.openai import (
    _openai_text as _openai_text,
)
from batcher.ml.llm.engines.openai import (
    _openai_usage as _openai_usage,
)
from batcher.ml.llm.engines.openai import (
    http_engine,
)
from batcher.ml.llm.engines.sglang import sglang_engine
from batcher.ml.llm.engines.vllm import (
    _chat_messages as _chat_messages,
)
from batcher.ml.llm.engines.vllm import (
    _generate_routed as _generate_routed,
)
from batcher.ml.llm.engines.vllm import (
    _group_indices_by_adapter as _group_indices_by_adapter,
)
from batcher.ml.llm.engines.vllm import (
    _vllm_batch_defaults as _vllm_batch_defaults,
)
from batcher.ml.llm.engines.vllm import (
    _with_auto_quant as _with_auto_quant,
)
from batcher.ml.llm.engines.vllm import (
    vllm_engine,
)
from batcher.ml.llm.sizing import _truncate_to_window as _truncate_to_window

__all__ = [
    "Engine",
    "EngineFactory",
    "anthropic_engine",
    "bedrock_engine",
    "gemini_engine",
    "http_engine",
    "sglang_engine",
    "vllm_engine",
]
