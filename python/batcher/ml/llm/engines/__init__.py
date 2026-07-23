"""LLM engine adapters — the pluggable ``list[str] -> list[str]`` backends.

An *engine* is the only thing `generate` needs from an LLM: hand it a batch of
requests, get back a string per request in order. Keeping that contract this narrow is
what lets vLLM, an OpenAI-compatible HTTP endpoint, and a deterministic test double be
interchangeable — and what keeps the columnar machinery in `generate` free of any
model library.

`base` holds the contract, `vllm` the GPU-resident offline backend, and `openai` the
served-endpoint backend. The private helpers are re-exported here because they are the
seams the engine tests drive directly, without a GPU or a network.
"""

from __future__ import annotations

from batcher.ml.llm.engines.base import Engine, EngineFactory
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
    _truncate_to_window as _truncate_to_window,
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

__all__ = ["Engine", "EngineFactory", "http_engine", "vllm_engine"]
