"""Embedding encoders backed by a *served* endpoint, not a local model.

`sentence_transformer_encoder` loads the weights into the worker; these instead call an
HTTP embedding service — an OpenAI-compatible ``/embeddings`` API (OpenAI, Azure OpenAI,
Together, vLLM's embedding server, ...) or a HuggingFace Text-Embeddings-Inference (TEI)
endpoint. That is the right shape when the embedding model runs behind a GPU service you
do not want to co-locate with the data pipeline, or when it is a hosted API with no local
weights at all.

Each factory returns a **load-once class UDF** shaped exactly like
`sentence_transformer_encoder`: instantiate once per worker (so the connection pool is
built once, not per batch), call on a `pyarrow.RecordBatch`, and it appends the embedding
column. So it drops into ``ds.ml.embed(<encoder>)`` / ``ds.ml.map_batches`` and reuses the
whole distributed / concurrency machinery, and it produces the same tensor (or
``fixed_size_list``) column the local encoder does — the two are interchangeable at the
call site.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from batcher.ml.embed import _check_output_type, _to_embedding_column

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["openai_embedding_encoder", "tei_encoder"]


def _append_embedding_column(batch: Any, output_column: str, col: Any) -> Any:
    """Append (or replace) `output_column` on `batch` — the shared write-back step."""
    if output_column in batch.schema.names:
        idx = batch.schema.get_field_index(output_column)
        return batch.set_column(idx, output_column, col)
    return batch.append_column(output_column, col)


def _chunks(items: list, size: int) -> list[list]:
    """Split `items` into consecutive chunks of at most `size` (order preserved)."""
    size = max(1, size)
    return [items[i : i + size] for i in range(0, len(items), size)]


class _ApiEncoder:
    """Base class UDF: embed a text column by calling `_embed_chunk` over sub-batches.

    Holds the endpoint config and a worker-lifetime thread pool so a batch's sub-requests
    overlap on the network (each blocks on I/O with the GIL released) and connections are
    not rebuilt per batch. Subclasses supply `_embed_chunk`, which turns one list of texts
    into one list of equal-length vectors, in request order.
    """

    def __init__(
        self,
        *,
        text_column: str,
        output_column: str,
        normalize: bool,
        output_type: str,
        max_batch: int,
        concurrency: int,
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self._text_column = text_column
        self._output_column = output_column
        self._normalize = normalize
        self._output_type = output_type
        self._max_batch = max(1, max_batch)
        self._pool = ThreadPoolExecutor(max_workers=max(1, concurrency))

    def _embed_chunk(self, texts: list[str]) -> list[Sequence[float]]:
        raise NotImplementedError

    def _embed_all(self, texts: list[str]) -> list[Sequence[float]]:
        """Embed every text, chunked to `max_batch` per request and re-assembled in order.

        A single chunk is embedded inline; several are dispatched concurrently over the
        worker's pool. ``executor.map`` preserves order, so the flattened result lines up
        with `texts` row-for-row.
        """
        groups = _chunks(texts, self._max_batch)
        if len(groups) <= 1:
            results = [self._embed_chunk(g) for g in groups]
        else:
            results = list(self._pool.map(self._embed_chunk, groups))
        return [vec for group in results for vec in group]

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        # A null (or non-string) cell becomes an empty string so every row still yields a
        # vector aligned to it; filter nulls upstream if a real embedding is required.
        raw = batch.column(self._text_column).to_pylist()
        texts = ["" if v is None else str(v) for v in raw]
        vectors = self._embed_all(texts)
        col = _to_embedding_column(
            vectors, normalize=self._normalize, output_type=self._output_type
        )
        return _append_embedding_column(batch, self._output_column, col)


def openai_embedding_encoder(
    model: str,
    text_column: str,
    *,
    base_url: str = "https://api.openai.com/v1",
    api_key: str | None = None,
    output_column: str = "embedding",
    dimensions: int | None = None,
    normalize: bool = False,
    output_type: str = "fixed_size_list",
    max_batch: int = 512,
    timeout: float = 60.0,
    concurrency: int = 8,
) -> type:
    """A load-once class UDF that embeds `text_column` via an OpenAI-compatible endpoint.

    Calls ``{base_url}/embeddings`` — the shape OpenAI, Azure OpenAI, Together, and vLLM's
    embedding server all speak — and appends the vector as `output_column`. Drops into
    ``ds.ml.embed(...)`` / ``ds.ml.map_batches`` exactly like `sentence_transformer_encoder`,
    so the served-model and local-model paths are interchangeable at the call site. No
    optional dependency: the request goes over the standard library.

    Each batch's texts are sent in requests of at most `max_batch` inputs (the embeddings
    API accepts an array of inputs per call), dispatched **concurrently** up to
    `concurrency` in-flight requests so a large batch is not serialized. Results are
    reassembled by the response ``index`` field, so a server that returns them out of
    order still lines up row-for-row.

    Examples:
        .. doctest::

            >>> from batcher.ml import openai_embedding_encoder  # doctest: +SKIP
            >>> enc = openai_embedding_encoder(  # doctest: +SKIP
            ...     "text-embedding-3-small", "text", api_key="sk-...", dimensions=256
            ... )
            >>> ds.ml.embed(enc).collect()  # doctest: +SKIP

    Args:
        model: the embedding model name the endpoint expects.
        text_column: the string column to embed.
        base_url: the OpenAI-compatible API root (defaults to OpenAI itself).
        api_key: bearer token; falls back to ``$OPENAI_API_KEY`` when unset.
        output_column: the appended (or replaced) embedding column.
        dimensions: request a shortened embedding width (Matryoshka models such as
            ``text-embedding-3-*``); omitted from the body when unset.
        normalize: L2-normalize each vector, so a dot product is cosine similarity.
        output_type: ``"fixed_size_list"`` (default, what Lance ANN indexing expects) or
            ``"tensor"`` for a fixed-shape-tensor column.
        max_batch: inputs per request; larger amortizes the round trip, up to the
            endpoint's input-array limit.
        timeout: per-request timeout in seconds.
        concurrency: in-flight requests per batch. Set to 1 to serialize.

    Returns:
        A class to instantiate once per worker, callable on a `pyarrow.RecordBatch`.

    Raises:
        PlanError: if `output_type` is not ``"tensor"`` or ``"fixed_size_list"``.
    """
    _check_output_type(output_type)

    class _OpenAIEmbeddingEncoder(_ApiEncoder):
        def __init__(self) -> None:
            import os

            super().__init__(
                text_column=text_column,
                output_column=output_column,
                normalize=normalize,
                output_type=output_type,
                max_batch=max_batch,
                concurrency=concurrency,
            )
            self._url = base_url.rstrip("/") + "/embeddings"
            self._headers = {"Content-Type": "application/json"}
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if key:
                self._headers["Authorization"] = f"Bearer {key}"

        def _embed_chunk(self, texts: list[str]) -> list[Sequence[float]]:
            from batcher.ml.serving.http import post_json

            body: dict[str, object] = {"model": model, "input": texts}
            if dimensions is not None:
                body["dimensions"] = dimensions
            resp = post_json(self._url, body, headers=self._headers, timeout=timeout)
            data = resp["data"]
            # The API documents `index`, and streaming servers may return out of order;
            # sorting by it is the only thing that keeps a vector paired with its text.
            ordered = sorted(data, key=lambda d: d.get("index", 0))
            return [d["embedding"] for d in ordered]

    return _OpenAIEmbeddingEncoder


def tei_encoder(
    text_column: str,
    *,
    base_url: str,
    output_column: str = "embedding",
    normalize: bool = True,
    truncate: bool = True,
    output_type: str = "fixed_size_list",
    max_batch: int = 32,
    timeout: float = 60.0,
    concurrency: int = 8,
) -> type:
    """A load-once class UDF embedding `text_column` via a HuggingFace TEI endpoint.

    Calls the Text-Embeddings-Inference ``{base_url}/embed`` route, which takes
    ``{"inputs": [...]}`` and returns a plain list of vectors. TEI is the standard way to
    serve an open embedding model (BGE, GTE, E5, ...) on a GPU; this points the pipeline at
    that service instead of loading the weights into every worker. Drops into
    ``ds.ml.embed(...)`` / ``ds.ml.map_batches`` like `sentence_transformer_encoder`.

    `normalize` and `truncate` are handled **server-side** by TEI (they are fields on the
    request), so an over-length input is truncated to the model's context rather than
    failing the batch, and normalization happens before the vector crosses the network.

    Examples:
        .. doctest::

            >>> from batcher.ml import tei_encoder  # doctest: +SKIP
            >>> enc = tei_encoder("text", base_url="http://localhost:8080")  # doctest: +SKIP
            >>> ds.ml.embed(enc).collect()  # doctest: +SKIP

    Args:
        text_column: the string column to embed.
        base_url: the TEI server root (e.g. ``http://localhost:8080``).
        output_column: the appended (or replaced) embedding column.
        normalize: ask TEI to L2-normalize each vector (its own ``normalize`` field).
        truncate: ask TEI to truncate an over-length input rather than error.
        output_type: ``"fixed_size_list"`` (default) or ``"tensor"``.
        max_batch: inputs per request; TEI batches server-side up to its own limit.
        timeout: per-request timeout in seconds.
        concurrency: in-flight requests per batch. Set to 1 to serialize.

    Returns:
        A class to instantiate once per worker, callable on a `pyarrow.RecordBatch`.

    Raises:
        PlanError: if `output_type` is not ``"tensor"`` or ``"fixed_size_list"``.
    """
    _check_output_type(output_type)

    class _TeiEncoder(_ApiEncoder):
        def __init__(self) -> None:
            super().__init__(
                text_column=text_column,
                output_column=output_column,
                # TEI normalizes server-side; never re-normalize an already-unit vector.
                normalize=False,
                output_type=output_type,
                max_batch=max_batch,
                concurrency=concurrency,
            )
            self._url = base_url.rstrip("/") + "/embed"
            self._headers = {"Content-Type": "application/json"}

        def _embed_chunk(self, texts: list[str]) -> list[Sequence[float]]:
            from batcher.ml.serving.http import post_json

            body = {"inputs": texts, "normalize": normalize, "truncate": truncate}
            # TEI's /embed returns the vectors as a bare list, already in input order.
            return post_json(self._url, body, headers=self._headers, timeout=timeout)

    return _TeiEncoder


EmbeddingEncoder = Callable[[list[str]], Sequence[Sequence[float]]]
"""A batch of texts → their vectors, in order — the served-endpoint encoder contract."""
