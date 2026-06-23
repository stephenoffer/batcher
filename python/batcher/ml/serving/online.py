"""Online (low-latency) serving — a thin Ray Serve adapter over the batch primitives.

Batcher is throughput-oriented (offline batch inference). This adds the *online*
counterpart: a Ray Serve deployment that loads a model **once** (the same load-once
factory the batch path uses) and answers per-request calls, coalescing concurrent
requests with Serve's native ``@serve.batch``. It deliberately introduces **no** second
execution engine — the same ``build`` factory feeds both the offline `InferencePool`
and this online deployment, so a model proven in batch serves online unchanged.

Gated behind the optional ``batcher-engine[serve]`` extra; importing this module is cheap
(Ray Serve is imported only when a deployment is built).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import BackendError

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
        BackendError: if Ray Serve is not installed (``pip install 'batcher-engine[serve]'``).
    """
    try:
        from ray import serve
    except ImportError as exc:  # pragma: no cover - optional extra
        raise BackendError("online serving needs: pip install 'batcher-engine[serve]'") from exc

    factory = build

    @serve.deployment(name=name, **deployment_options)
    class _BatcherDeployment:
        def __init__(self) -> None:
            built = factory()
            # `build` may itself be a class (load-once); resolve to the callable.
            self._predict = built() if isinstance(built, type) else built

        @serve.batch(max_batch_size=max_batch_size, batch_wait_timeout_s=batch_wait_timeout_s)
        async def _batched(self, inputs: list[Any]) -> list[Any]:
            return list(self._predict(inputs))

        async def __call__(self, request: Any) -> Any:
            return await self._batched(request)

    return _BatcherDeployment
