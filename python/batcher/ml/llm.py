"""LLM batch inference — the Ray Data LLM competitor (offline text generation).

Run a text-generation engine (vLLM, or any callable) over millions of rows. The
engine is built **once per worker** (the load-once pattern) and does its *own*
continuous batching, so Batcher feeds it each input batch as a whole request list
and imposes **no** outer fixed batch size — an outer batch-size PID would fight
vLLM's scheduler (the distinct LLM-stage contract, vs the embeddings stage which
*does* adapt its batch size). On top of that the operator handles the surrounding
columnar work: build each prompt from row columns via a `template`, and optionally
parse each structured/JSON output into typed columns.

The engine is injected as a factory, so this works with vLLM, an OpenAI client, or a
deterministic test double — the engine never depends on a specific library. A vLLM
adapter (`vllm_engine`) lives behind the optional ``batcher[vllm]`` extra.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

from batcher.ml.inference import InferencePool

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["Engine", "EngineFactory", "http_engine", "llm_generate", "vllm_engine"]

# Maps a list of prompts to a list of generated strings (one per prompt, in order).
Engine = Callable[[list[str]], Sequence[str]]
EngineFactory = Callable[[], Engine]


def _render(template: str | None, column: str, batch: pa.RecordBatch) -> list[str]:
    """The prompt for each row: ``column`` verbatim, or `template` formatted with the
    row's columns (``"{system} Q: {question}"``-style ``str.format`` placeholders)."""
    if template is None:
        return [str(v) for v in batch.column(column).to_pylist()]
    rows = batch.to_pylist()
    return [template.format(**row) for row in rows]


def llm_generate(
    batches: Iterable[pa.RecordBatch],
    engine_factory: EngineFactory,
    *,
    prompt_column: str,
    output_column: str = "response",
    template: str | None = None,
    parse_json: bool = False,
    num_workers: int = 2,
    target_batch_rows: int = 256,
) -> Iterator[pa.RecordBatch]:
    """Append an LLM-generated `output_column` to each batch.

    Args:
        batches: an iterable of `pyarrow.RecordBatch`.
        engine_factory: zero-arg callable returning an engine (``list[str]`` →
            sequence of strings); called once per worker so the model loads once.
        prompt_column: the text column to send (ignored if `template` is set, which
            builds prompts from any columns).
        output_column: name of the appended generated column.
        template: optional ``str.format`` template over the row's columns.
        parse_json: parse each output as JSON into a struct column (guided/structured
            decoding); on a parse error the row's value is null.
        num_workers / target_batch_rows: forwarded to `InferencePool` (no latency
            controller — the engine owns its own batching).

    Yields:
        Each input batch with `output_column` appended, in order.
    """
    import pyarrow as pa

    def make_worker() -> Callable[[pa.RecordBatch], pa.RecordBatch]:
        engine = engine_factory()  # built once per worker

        def worker(batch: pa.RecordBatch) -> pa.RecordBatch:
            prompts = _render(template, prompt_column, batch)
            outputs = list(engine(prompts))
            if parse_json:
                col = pa.array([_safe_json(o) for o in outputs])
            else:
                col = pa.array([str(o) for o in outputs], type=pa.string())
            arrays = [batch.column(i) for i in range(batch.num_columns)] + [col]
            names = [*batch.schema.names, output_column]
            return pa.RecordBatch.from_arrays(arrays, names=names)

        return worker

    pool = InferencePool(make_worker, num_workers=num_workers, target_batch_rows=target_batch_rows)
    yield from pool.run(batches)


def _safe_json(text: str) -> object | None:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def vllm_engine(
    model: str,
    *,
    sampling: dict[str, object] | None = None,
    guided_json: dict[str, object] | None = None,
    guided_regex: str | None = None,
    lora_path: str | None = None,
    **engine_kwargs: object,
) -> EngineFactory:
    """An `EngineFactory` backed by vLLM (requires ``batcher[vllm]`` + a GPU).

    Returns a factory that builds a vLLM engine once per worker and exposes it as a
    ``list[str] -> list[str]`` callable, with full control over modern batch-inference
    knobs:

    * `sampling` — `SamplingParams` kwargs (``temperature``, ``top_p``, ``max_tokens``,
      ``stop``, ``n``, ``seed``, ...). Defaults to greedy (``temperature=0``).
    * `guided_json` / `guided_regex` — **structured output**: constrain generation to a
      JSON schema or regex via vLLM's guided decoding (the reliable way to get parseable
      output; pair with ``llm_generate(parse_json=True)``).
    * `lora_path` — serve a LoRA adapter on top of the base `model`.
    * `engine_kwargs` — pass through to ``vllm.LLM`` (``max_model_len``,
      ``gpu_memory_utilization``, ``tensor_parallel_size``, ``quantization``, ...).
    """

    def factory() -> Engine:
        from vllm import LLM, SamplingParams

        llm = LLM(model=model, enable_lora=lora_path is not None, **engine_kwargs)
        sampling_kwargs = {"temperature": 0.0, **(sampling or {})}
        guided = _guided_decoding(guided_json, guided_regex)
        if guided is not None:
            sampling_kwargs["guided_decoding"] = guided
        params = SamplingParams(**sampling_kwargs)
        lora = _lora_request(lora_path)

        def engine(prompts: list[str]) -> list[str]:
            outputs = llm.generate(prompts, params, lora_request=lora)
            return [o.outputs[0].text for o in outputs]

        return engine

    return factory


def _guided_decoding(guided_json: dict | None, guided_regex: str | None) -> object | None:
    """A vLLM `GuidedDecodingParams` for JSON-schema or regex-constrained output."""
    if guided_json is None and guided_regex is None:
        return None
    from vllm.sampling_params import GuidedDecodingParams

    if guided_json is not None:
        return GuidedDecodingParams(json=guided_json)
    return GuidedDecodingParams(regex=guided_regex)


def _lora_request(lora_path: str | None) -> object | None:
    """A vLLM `LoRARequest` for `lora_path`, or `None` when no adapter is served."""
    if lora_path is None:
        return None
    from vllm.lora.request import LoRARequest

    return LoRARequest("adapter", 1, lora_path)


def http_engine(
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    system: str | None = None,
    chat: bool = True,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> EngineFactory:
    """An `EngineFactory` calling an OpenAI-compatible HTTP endpoint (the Ray Data
    ``HttpRequestProcessor`` analog) — batch inference against a *served* model.

    Targets ``{base_url}/chat/completions`` (``chat=True``, the default) or
    ``/completions``; with chat, the **server applies the model's chat template**, so a
    plain prompt string is wrapped as a user message (with an optional `system`
    message). Works against vLLM's OpenAI server, llama.cpp, or a hosted API. `api_key`
    sets the bearer token; `max_tokens`/`temperature` control decoding.
    """

    def factory() -> Engine:
        from batcher.ml.serving.http import post_json

        url = base_url.rstrip("/") + ("/chat/completions" if chat else "/completions")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        def engine(prompts: list[str]) -> list[str]:
            out: list[str] = []
            for prompt in prompts:
                body = _openai_body(model, prompt, chat, system, max_tokens, temperature)
                # Retries with backoff handle the 429 rate limits hosted APIs return.
                resp = post_json(url, body, headers=headers, timeout=timeout)
                out.append(_openai_text(resp, chat))
            return out

        return engine

    return factory


def _openai_body(
    model: str, prompt: str, chat: bool, system: str | None, max_tokens: int, temperature: float
) -> dict[str, object]:
    common = {"model": model, "max_tokens": max_tokens, "temperature": temperature}
    if not chat:
        return {**common, "prompt": prompt}
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    return {**common, "messages": messages}


def _openai_text(response: dict, chat: bool) -> str:
    choice = response["choices"][0]
    return choice["message"]["content"] if chat else choice["text"]
