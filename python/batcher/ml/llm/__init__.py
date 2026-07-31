"""LLM batch inference — the Ray Data LLM competitor (offline text generation).

Run a text-generation engine (vLLM, an OpenAI-compatible endpoint, or any callable)
over millions of rows. `engines` holds the pluggable backends; `generate` holds the
columnar work (prompt templating, vision/LoRA request building, JSON parsing, token
usage). Reached as `ds.ml.generate(...)`, or as `llm_generate` over a batch stream.
`structured` turns a generation into *typed columns* (`ds.ml.extract` / `ds.ml.classify`) —
the AI-powered-ETL step, where an unconstrained string becomes a column you can query.
`judge` runs the other direction: a stronger model *scores* generations, as a parsed and
range-checked column rather than a Python loop over examples.
"""

from __future__ import annotations

from batcher.ml.llm.engines import (
    Engine,
    EngineFactory,
    anthropic_engine,
    http_engine,
    vllm_engine,
)
from batcher.ml.llm.generate import llm_generate, llm_udf
from batcher.ml.llm.judge import llm_pairwise_udf, llm_score_udf, llm_verify_udf
from batcher.ml.llm.packing import pack_sequences
from batcher.ml.llm.sizing import kv_cache_concurrency
from batcher.ml.llm.structured import json_schema, llm_classify_udf, llm_extract_udf

__all__ = [
    "Engine",
    "EngineFactory",
    "anthropic_engine",
    "http_engine",
    "json_schema",
    "kv_cache_concurrency",
    "llm_classify_udf",
    "llm_extract_udf",
    "llm_generate",
    "llm_pairwise_udf",
    "llm_score_udf",
    "llm_udf",
    "llm_verify_udf",
    "pack_sequences",
    "vllm_engine",
]
