"""AWS Bedrock and Google Gemini — the hosted providers whose wire shape is not OpenAI's.

Three of the four engines already here reach a model over HTTP, and all three do it through
`ml.serving.http.post_json` with a per-worker thread pool, a token-bucket limiter, and the
same `on_error` contract. What separates them is only the request body and where the text
sits in the reply. Those are exactly the two things this module supplies for the two large
clouds Batcher could not otherwise reach:

* **Bedrock** is how most of AWS runs a model, and its **Converse** API is the reason to
  target it rather than each vendor's own shape. Converse is one request format across every
  model on Bedrock — Claude, Llama, Mistral, Titan, Nova — so switching model families is a
  string change rather than a rewrite. It is also the only engine here that must **sign** its
  requests (SigV4), which is why it takes a different route to the same HTTP call.
* **Gemini** is Google's, reachable identically through the Gemini API and through Vertex AI;
  the difference is the host and the credential, so both are the same engine with a different
  `base_url`.

Both keep the `Engine` contract exactly — ``list[str] -> list[str]``, one string per request,
in order — so `ds.ml.generate`, `llm_generate`, `ds.ml.extract` and `ds.ml.classify` reach
them by swapping one factory and nothing columnar changes.
"""

from __future__ import annotations

from typing import Any

from batcher.ml.llm.channels import finish_reason_sink, usage_sink
from batcher.ml.llm.engines.base import Engine, EngineFactory, unpack_request
from batcher.ml.llm.engines.limits import _estimated_tokens, build_limiter

__all__ = ["bedrock_engine", "gemini_engine"]

#: Bedrock's ``stopReason`` and Gemini's ``finishReason`` normalized onto the
#: ``"stop"``/``"length"`` vocabulary the other engines report, so a truncation is detectable
#: the same way whichever provider produced it. Anything unlisted passes through unchanged.
_FINISH_REASON = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "STOP": "stop",
    "max_tokens": "length",
    "MAX_TOKENS": "length",
    "content_filtered": "content_filter",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
}


def _checked_on_error(on_error: str) -> None:
    """Reject an `on_error` mode neither engine implements.

    Raises:
        PlanError: If `on_error` is not ``"raise"`` or ``"null"``.
    """
    if on_error not in ("raise", "null"):
        from batcher._internal.errors import PlanError

        raise PlanError(f"on_error must be 'raise' or 'null', got {on_error!r}")


def bedrock_engine(
    model: str,
    *,
    region: str | None = None,
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
    top_p: float | None = None,
    stop_sequences: list[str] | None = None,
    additional_model_request_fields: dict | None = None,
    on_error: str = "raise",
    timeout: float = 60.0,
    retries: int = 3,
    concurrency: int = 8,
    requests_per_minute: float | None = None,
    tokens_per_minute: float | None = None,
) -> EngineFactory:
    """An `EngineFactory` calling AWS Bedrock's Converse API (requires ``boto3``).

    Converse is one request shape for every model family Bedrock hosts, so
    ``model="anthropic.claude-sonnet-4-20250514-v1:0"`` and
    ``model="meta.llama3-1-70b-instruct-v1:0"`` are the same call with a different string.
    Credentials come from the standard AWS chain — environment, profile, instance role,
    IRSA — so a worker on EC2 or EKS needs none passed in.

    The prompts in each batch go out over up to `concurrency` in-flight requests with input
    order preserved, so a batch costs the slowest request rather than their sum.

    Examples:
        .. doctest::

            >>> import batcher as bt  # doctest: +SKIP
            >>> engine = bt.ml.bedrock_engine(  # doctest: +SKIP
            ...     "anthropic.claude-sonnet-4-20250514-v1:0", region="us-west-2"
            ... )
            >>> ds.ml.generate(engine, prompt_column="question").collect()  # doctest: +SKIP

    Args:
        model: the Bedrock model id or inference-profile ARN.
        region: the AWS region; falls back to the boto3 session's own resolution.
        system: a system instruction prepended to every conversation.
        max_tokens: tokens to sample per request (a per-row ``max_tokens`` overrides it).
        temperature: sampling temperature; omitted from the request when unset, because
            several Bedrock models reject a null.
        top_p: nucleus-sampling mass; omitted when unset.
        stop_sequences: strings that end a generation.
        additional_model_request_fields: model-specific fields Converse passes straight
            through (a Claude ``thinking`` block, a Llama ``repetition_penalty``).
        on_error: ``"raise"`` (default) to fail the batch, or ``"null"`` to yield an empty
            generation for that row and continue.
        timeout: per-request timeout in seconds.
        retries: retry attempts per request on a transient failure. Bedrock throttles with
            ``ThrottlingException``, which boto3 retries itself; this is the outer bound.
        concurrency: in-flight requests per batch. Set to 1 to serialize.
        requests_per_minute: client-side cap on requests per minute, **per worker**.
        tokens_per_minute: client-side cap on tokens per minute, per worker.

    Returns:
        A zero-arg factory building the Bedrock-backed `Engine` once per worker.

    Raises:
        PlanError: if `on_error` is not ``"raise"`` or ``"null"``.
    """
    _checked_on_error(on_error)
    limiter = build_limiter(requests_per_minute, tokens_per_minute)

    def factory() -> Engine:
        from concurrent.futures import ThreadPoolExecutor

        from batcher._internal.optional import require

        boto3 = require("boto3", feature="bedrock_engine", provides="boto3", extra="aws")
        from botocore.config import Config

        # `max_attempts=1` because retries are owned here, once. boto3's own adaptive retry
        # would multiply against this loop, turning `retries=3` into up to twelve attempts
        # and a tail latency nobody chose. `read_timeout` is what actually bounds a hung
        # generation; the connect timeout only covers reaching the endpoint.
        config = Config(
            read_timeout=timeout,
            connect_timeout=min(timeout, 10.0),
            retries={"max_attempts": 1, "mode": "standard"},
        )
        client = boto3.client("bedrock-runtime", region_name=region, config=config)

        def call_one(request: Any) -> _Reply:
            prompt, image, overrides = unpack_request(request, ("max_tokens", "temperature"))
            body = _converse_body(
                model,
                prompt,
                system,
                max_tokens,
                temperature,
                top_p,
                stop_sequences,
                additional_model_request_fields,
                overrides,
                image,
            )
            if limiter is not None:
                limiter.acquire(_estimated_tokens(prompt, _limiter_view(body, prompt)))
            try:
                response = _converse_with_retry(client, body, retries)
            except Exception:
                if on_error == "raise":
                    raise
                return _Reply("", (None, None), None)
            return _Reply(
                _converse_text(response),
                _converse_usage(response),
                _FINISH_REASON.get(response.get("stopReason"), response.get("stopReason")),
            )

        pool = ThreadPoolExecutor(max_workers=max(1, concurrency))
        engine = _batched_engine(call_one, pool, concurrency)
        engine.close = lambda: pool.shutdown(wait=False)
        return engine

    return factory


def gemini_engine(
    model: str,
    *,
    api_key: str | None = None,
    base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
    top_p: float | None = None,
    stop_sequences: list[str] | None = None,
    response_schema: dict | None = None,
    safety_settings: list[dict] | None = None,
    extra_body: dict | None = None,
    on_error: str = "raise",
    timeout: float = 60.0,
    retries: int = 3,
    backoff: float = 0.5,
    concurrency: int = 8,
    requests_per_minute: float | None = None,
    tokens_per_minute: float | None = None,
) -> EngineFactory:
    """An `EngineFactory` calling Google's Gemini ``generateContent`` API.

    The same engine reaches Vertex AI: point `base_url` at the Vertex endpoint for your
    project and region and supply an OAuth token as `api_key`. Nothing else differs, because
    the request body is the same on both.

    `response_schema` is the reliable route to parseable output — Gemini constrains the
    generation to the schema rather than being asked to produce JSON — so pair it with
    ``llm_generate(parse_json=True)`` and get typed columns instead of a string to salvage.

    Examples:
        .. doctest::

            >>> import batcher as bt  # doctest: +SKIP
            >>> engine = bt.ml.gemini_engine("gemini-2.0-flash")  # doctest: +SKIP
            >>> ds.ml.generate(engine, prompt_column="question").collect()  # doctest: +SKIP

    Args:
        model: the Gemini model name, such as ``"gemini-2.0-flash"``.
        api_key: the API key; falls back to ``$GEMINI_API_KEY`` then ``$GOOGLE_API_KEY``.
        base_url: the API root. Point this at Vertex AI to use a project endpoint instead.
        system: a system instruction applied to every request.
        max_tokens: tokens to sample per request (a per-row ``max_tokens`` overrides it).
        temperature: sampling temperature; omitted from the request when unset.
        top_p: nucleus-sampling mass; omitted when unset.
        stop_sequences: strings that end a generation.
        response_schema: a JSON schema the response must conform to, which also switches
            the response MIME type to ``application/json``.
        safety_settings: Gemini ``safetySettings`` blocks, to raise or lower the thresholds
            at which a generation is blocked.
        extra_body: extra top-level fields merged into every request.
        on_error: ``"raise"`` (default) to fail the batch, or ``"null"`` to yield an empty
            generation for that row and continue.
        timeout: per-request timeout in seconds.
        retries: retry attempts per request on a transient failure (429/5xx/connection).
        backoff: base seconds for the jittered exponential backoff between retries.
        concurrency: in-flight requests per batch. Set to 1 to serialize.
        requests_per_minute: client-side cap on requests per minute, **per worker**.
        tokens_per_minute: client-side cap on tokens per minute, per worker.

    Returns:
        A zero-arg factory building the Gemini-backed `Engine` once per worker.

    Raises:
        PlanError: if `on_error` is not ``"raise"`` or ``"null"``.
    """
    _checked_on_error(on_error)
    limiter = build_limiter(requests_per_minute, tokens_per_minute)

    def factory() -> Engine:
        import os
        from concurrent.futures import ThreadPoolExecutor

        from batcher.ml.serving.http import post_json

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        url = f"{base_url.rstrip('/')}/models/{model}:generateContent"
        headers = {"Content-Type": "application/json"}
        if key and key.startswith("ya29."):
            headers["Authorization"] = f"Bearer {key}"  # a Vertex OAuth token, not an API key
        elif key:
            # In the header rather than the query string: a URL carrying a credential is
            # logged by every proxy and access log between here and Google.
            headers["x-goog-api-key"] = key

        def call_one(request: Any) -> _Reply:
            prompt, image, overrides = unpack_request(request, ("max_tokens", "temperature"))
            body = _gemini_body(
                prompt,
                system,
                max_tokens,
                temperature,
                top_p,
                stop_sequences,
                response_schema,
                safety_settings,
                extra_body,
                overrides,
                image,
            )
            if limiter is not None:
                limiter.acquire(_estimated_tokens(prompt, _limiter_view(body, prompt)))
            try:
                response = post_json(
                    url, body, headers=headers, timeout=timeout, retries=retries, backoff=backoff
                )
            except Exception:
                if on_error == "raise":
                    raise
                return _Reply("", (None, None), None)
            return _Reply(
                _gemini_text(response), _gemini_usage(response), _gemini_finish_reason(response)
            )

        pool = ThreadPoolExecutor(max_workers=max(1, concurrency))
        engine = _batched_engine(call_one, pool, concurrency)
        engine.close = lambda: pool.shutdown(wait=False)
        return engine

    return factory


class _Reply:
    """One request's outcome: the text plus its usage and finish reason."""

    __slots__ = ("finish_reason", "text", "usage")

    def __init__(
        self, text: str, usage: tuple[int | None, int | None], finish_reason: str | None
    ) -> None:
        self.text = text
        self.usage = usage
        self.finish_reason = finish_reason


def _batched_engine(call_one: Any, pool: Any, concurrency: int) -> Engine:
    """The batch loop both engines share: overlap the requests, then report every signal.

    ``Executor.map`` preserves input order regardless of completion order, so the returned
    strings stay aligned with the rows that produced them — the property everything columnar
    downstream depends on.
    """

    def engine(prompts: list) -> list[str]:
        if concurrency <= 1 or len(prompts) <= 1:
            replies = [call_one(p) for p in prompts]
        else:
            replies = list(pool.map(call_one, prompts))
        usage = [r.usage for r in replies]
        usage_sink().report(usage)
        finish_reason_sink().report([r.finish_reason for r in replies])
        engine.last_usage = usage  # the documented legacy channel
        return [r.text for r in replies]

    return engine


def _limiter_view(body: dict, prompt: str) -> dict:
    """A body the token-bucket limiter can charge, in the shape it knows how to read.

    `_estimated_tokens` reads ``max_tokens`` and counts image blocks under ``messages``;
    neither provider spells it that way. Translating here keeps one estimator rather than
    giving each provider its own, which is how one of them quietly stops counting images.
    """
    return {
        "max_tokens": _declared_max_tokens(body),
        "messages": [{"content": [{"type": "image"}] * _image_blocks(body)}],
        "prompt": prompt,
    }


def _declared_max_tokens(body: dict) -> int:
    """The generation budget a Bedrock or Gemini body reserved, or 0 when it named none."""
    inference = body.get("inferenceConfig")
    if isinstance(inference, dict) and "maxTokens" in inference:
        return int(inference["maxTokens"])
    generation = body.get("generationConfig")
    if isinstance(generation, dict) and "maxOutputTokens" in generation:
        return int(generation["maxOutputTokens"])
    return 0


def _image_blocks(body: dict) -> int:
    """How many images a Bedrock or Gemini body carries, for the token estimate."""
    count = 0
    for message in body.get("messages") or ():
        count += sum(1 for block in message.get("content", ()) if "image" in block)
    for content in body.get("contents") or ():
        count += sum(1 for part in content.get("parts", ()) if "inline_data" in part)
    return count


def _converse_body(
    model: str,
    prompt: str,
    system: str | None,
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    stop_sequences: list[str] | None,
    additional_fields: dict | None,
    overrides: dict,
    image: Any,
) -> dict[str, Any]:
    """The Bedrock Converse request. Unset fields are omitted rather than sent as null,
    because several model families on Bedrock reject a null they did not ask for."""
    inference: dict[str, Any] = {"maxTokens": int(overrides.get("max_tokens", max_tokens))}
    temp = overrides.get("temperature", temperature)
    if temp is not None:
        inference["temperature"] = float(temp)
    if top_p is not None:
        inference["topP"] = float(top_p)
    if stop_sequences:
        inference["stopSequences"] = list(stop_sequences)
    body: dict[str, Any] = {
        "modelId": model,
        "messages": [{"role": "user", "content": _converse_content(prompt, image)}],
        "inferenceConfig": inference,
    }
    if system is not None:
        body["system"] = [{"text": system}]
    if additional_fields:
        body["additionalModelRequestFields"] = additional_fields
    return body


def _converse_content(prompt: str, image: Any) -> list[dict[str, Any]]:
    """The user turn's content blocks: the text, preceded by an image block for vision.

    Converse takes image **bytes**, not base64 — boto3 encodes the blob itself on the way
    out, so encoding here would send the base64 of the base64.
    """
    if image is None:
        return [{"text": prompt}]
    return [
        {"image": {"format": "png", "source": {"bytes": _png_bytes(image)}}},
        {"text": prompt},
    ]


def _png_bytes(image: Any) -> bytes:
    """A decoded image as PNG bytes (a passthrough for something already encoded)."""
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _converse_with_retry(client: Any, body: dict[str, Any], retries: int) -> dict[str, Any]:
    """One Converse call, retrying the throttles and transient faults with jittered backoff.

    The retryable set is named rather than "any exception": a ``ValidationException`` is a
    malformed request and an ``AccessDeniedException`` is a permissions problem, and retrying
    either only multiplies the latency before the same failure surfaces.
    """
    import time

    from batcher.ml.serving.base import _jittered_backoff

    retryable = (
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ModelTimeoutException",
        "ModelNotReadyException",
    )
    last: Exception | None = None
    for attempt in range(max(0, retries) + 1):
        try:
            return client.converse(**body)
        except Exception as exc:
            if _error_code(exc) not in retryable:
                raise
            last = exc
        if attempt < retries:
            time.sleep(_jittered_backoff(0.5, attempt))
    from batcher._internal.errors import BackendError

    raise BackendError(f"bedrock converse failed after {retries + 1} attempts: {last}")


def _error_code(exc: Exception) -> str:
    """The AWS error code behind a botocore exception, or the exception's type name.

    Read defensively rather than as ``exc.response["Error"]["Code"]``: a connection-level
    failure carries no ``response`` at all, and a partially-formed one carries a non-dict.
    Neither is retryable-or-not on its own account, and neither may turn the retry decision
    into a second exception raised from inside the handler for the first.
    """
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return type(exc).__name__
    error = response.get("Error")
    if not isinstance(error, dict):
        return type(exc).__name__
    return str(error.get("Code") or type(exc).__name__)


def _converse_text(response: dict) -> str:
    """The generation: every ``text`` block of the reply concatenated.

    A guardrail intervention returns HTTP 200 with no text block at all, so reading
    ``content[0]["text"]`` unconditionally would crash a batch on one filtered row.
    """
    message = (response.get("output") or {}).get("message") or {}
    return "".join(block.get("text", "") for block in message.get("content") or ())


def _converse_usage(response: dict) -> tuple[int | None, int | None]:
    """The ``(inputTokens, outputTokens)`` Converse reports, or ``(None, None)``."""
    usage = response.get("usage") or {}
    return usage.get("inputTokens"), usage.get("outputTokens")


def _gemini_body(
    prompt: str,
    system: str | None,
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    stop_sequences: list[str] | None,
    response_schema: dict | None,
    safety_settings: list[dict] | None,
    extra_body: dict | None,
    overrides: dict,
    image: Any,
) -> dict[str, Any]:
    """The Gemini ``generateContent`` request body, with unset fields omitted."""
    config: dict[str, Any] = {"maxOutputTokens": int(overrides.get("max_tokens", max_tokens))}
    temp = overrides.get("temperature", temperature)
    if temp is not None:
        config["temperature"] = float(temp)
    if top_p is not None:
        config["topP"] = float(top_p)
    if stop_sequences:
        config["stopSequences"] = list(stop_sequences)
    if response_schema is not None:
        # Both together: the schema constrains the decoding, and the MIME type is what makes
        # Gemini honor it. Setting only the schema returns prose that happens to describe it.
        config["responseMimeType"] = "application/json"
        config["responseSchema"] = response_schema
    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": _gemini_parts(prompt, image)}],
        "generationConfig": config,
    }
    if system is not None:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if safety_settings:
        body["safetySettings"] = safety_settings
    if extra_body:
        body.update(extra_body)
    return body


def _gemini_parts(prompt: str, image: Any) -> list[dict[str, Any]]:
    """The user turn's parts: the text, preceded by inline image data for vision."""
    if image is None:
        return [{"text": prompt}]
    import base64

    data = base64.b64encode(_png_bytes(image)).decode("ascii")
    return [{"inline_data": {"mime_type": "image/png", "data": data}}, {"text": prompt}]


def _gemini_text(response: dict) -> str:
    """The generation: every text part of the first candidate concatenated.

    An empty ``candidates`` list is what a safety block returns, and a candidate with no
    ``parts`` is what a ``MAX_TOKENS`` stop during a thinking turn returns. Both are ordinary
    outcomes on a large batch, so both produce an empty string rather than an exception.
    """
    candidates = response.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts if isinstance(part, dict))


def _gemini_usage(response: dict) -> tuple[int | None, int | None]:
    """The ``(prompt, candidate)`` token counts Gemini reports, or ``(None, None)``."""
    usage = response.get("usageMetadata") or {}
    return usage.get("promptTokenCount"), usage.get("candidatesTokenCount")


def _gemini_finish_reason(response: dict) -> str | None:
    """The first candidate's finish reason, normalized to the shared vocabulary."""
    candidates = response.get("candidates") or []
    if not candidates:
        return "content_filter"  # no candidate at all is what a blocked prompt returns
    reason = candidates[0].get("finishReason")
    return _FINISH_REASON.get(reason, reason) if reason is not None else None
