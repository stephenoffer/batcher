"""Online (low-latency) serving — a thin Ray Serve adapter over the batch primitives.

Batcher is throughput-oriented (offline batch inference). This adds the *online*
counterpart: a Ray Serve deployment that loads a model **once** (the same load-once
factory the batch path uses) and answers per-request calls, coalescing concurrent
requests with Serve's native ``@serve.batch``. It deliberately introduces **no** second
execution engine — the same ``build`` factory feeds both the offline `InferencePool`
and this online deployment, so a model proven in batch serves online unchanged.

The coalesced batch runs in a **thread executor**, never on the replica's event loop. A
model forward pass is a long, synchronous, GIL-releasing call; awaiting it inline would
stall the loop for its whole duration, so every other request already queued on that
replica would wait behind it even though Serve had capacity to accept more.

Gated behind the optional ``batcher-engine[serve]`` extra; importing this module is cheap
(Ray Serve is imported only when a deployment is built).
"""

from __future__ import annotations

import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["serve_deployment"]


def serve_deployment(
    build: Callable[[], Callable[[list[Any]], list[Any]]],
    *,
    name: str = "batcher-model",
    max_batch_size: int = 16,
    batch_wait_timeout_s: float = 0.01,
    **deployment_options: Any,
) -> Any:
    """A Ray Serve deployment wrapping a load-once, request-batched predictor.

    Examples:
        .. doctest::

            >>> from ray import serve  # doctest: +SKIP
            >>> from batcher.ml import serve_deployment  # doctest: +SKIP
            >>> deployment = serve_deployment(build_predictor, num_replicas=2)  # doctest: +SKIP
            >>> serve.run(deployment.bind())  # doctest: +SKIP

    Args:
        build: a zero-arg factory (or class) returning a batched predictor — a
            ``list[input] -> list[output]`` callable. Run once per replica, so the
            model loads a single time (the same factory shape as `vllm_engine` /
            `InferencePool`).
        name: the Serve deployment name.
        max_batch_size: the most requests `@serve.batch` coalesces into one call.
        batch_wait_timeout_s: how long Serve waits to fill a batch before flushing.
        deployment_options: forwarded to ``@serve.deployment`` (e.g. ``num_replicas``,
            ``ray_actor_options={"num_gpus": 1}``, ``autoscaling_config``).

    Returns:
        A Ray Serve deployment class — ``serve.run(serve_deployment(...).bind())``.

    Raises:
        MissingDependencyError: if Ray Serve is not installed (naming ``pip install
            'batcher-engine[serve]'``).
    """
    # One worker: a model is rarely thread-safe and a GPU wants its calls serialized,
    # which is exactly what `@serve.batch` already assumes. The point of the executor is
    # to get the blocking call OFF the event loop, not to run predictions in parallel.
    from batcher._internal.optional import require

    serve = require("ray.serve", feature="online serving", provides="ray[serve]", extra="serve")

    factory = build

    @serve.deployment(name=name, **deployment_options)
    class _BatcherDeployment:
        def __init__(self) -> None:
            built = factory()
            # `build` may itself be a class (load-once); resolve to the callable.
            self._predict = built() if isinstance(built, type) else built
            self._pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"batcher-serve-{name}"
            )

        @serve.batch(max_batch_size=max_batch_size, batch_wait_timeout_s=batch_wait_timeout_s)
        async def _batched(self, inputs: list[Any]) -> list[Any]:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._pool, lambda: list(self._predict(inputs)))

        def __del__(self) -> None:
            pool = getattr(self, "_pool", None)
            if pool is not None:
                pool.shutdown(wait=False)
            # The replica's model is the expensive thing it holds — a CUDA context, an HTTP
            # session, a database handle — and a downscale or a reconfigure releases the
            # replica without releasing them. Same optional `close()` contract the batch path
            # honors in `teardown_udf` and `_close_workers`, and the same best-effort rule:
            # a replica going away must not raise on the way out.
            close = getattr(getattr(self, "_predict", None), "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()

        async def __call__(self, request: Any) -> Any:
            return await self._batched(request)

    return _BatcherDeployment
