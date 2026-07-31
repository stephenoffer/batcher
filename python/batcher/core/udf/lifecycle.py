"""Build and tear down a `map_batches` UDF instance (Core, layer 3).

A class `fn` is a load-once factory: it is instantiated once to load the model, and the
callable instance handles each batch — the GPU-inference pattern. This module owns the two
ends of that instance's life so the plan-walking (`execute`), the per-stage application
(`apply`), the streaming stages (`stream`), and the strategy probe (`strategy`) share one
definition and cannot disagree about when a model is built or released.

It deliberately depends on nothing else in `core.udf`, so it sits below the modules that use
it and breaks what would otherwise be an import cycle between them.
"""

from __future__ import annotations

import contextlib

from batcher.plan.logical import MapBatches

__all__ = ["build_udf_callable", "release_prebuilt", "teardown_udf"]


def build_udf_callable(fn: object) -> object:
    """Resolve a `map_batches` `fn` to the per-batch callable.

    A *class* (type) is a stateful factory: it is instantiated once here to load
    the model, and the instance (which must be callable) handles each batch. Any
    other callable is used directly. This is what lets a model load once per worker
    instead of once per batch — the GPU-inference pattern. Called once per worker
    (locally: once; distributed: once per actor).

    This is also the last point before a model loads that Batcher still controls, which is
    where the node's model cache has to be chosen: `huggingface_hub` fixes its cache path at
    import, so a worker that has already loaded a model cannot be moved off the container
    overlay afterwards. The call is memoized, self-limiting (it declines when an operator named
    a cache, when the default one already holds weights, and when no local volume is mounted),
    and returns without touching anything on a UDF that has nothing to do with models.
    """
    if not isinstance(fn, type):
        return fn
    from batcher._internal.site.model_cache import use_node_local_model_cache

    use_node_local_model_cache()
    return fn()


def teardown_udf(built: object, op: MapBatches) -> None:
    """Release a built class-UDF instance's resources via its optional ``close()``.

    Called only when the current path OWNS the instance — i.e. `op.fn` was a class this path
    built. A prebuilt instance passed in by a long-lived streaming loop or distributed actor is
    that owner's to tear down at *its* lifetime end, not here. This gives a load-once model
    (`def __call__(self, batch)`) a deterministic place to free a GPU allocation, an HTTP
    session, or a DB connection between partitions, instead of waiting for the garbage collector.

    Best-effort by contract: a failing `close()` must never fail the query (the results are
    already produced). Only a `close` attribute is honored — `object` defines none, so an
    unrelated inherited method is never triggered.
    """
    if not isinstance(op.fn, type):
        return
    close = getattr(built, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()


def release_prebuilt(plan: object) -> None:
    """Close every class-UDF instance `prebuild_factories` built into `plan`.

    `teardown_udf` deliberately declines a prebuilt instance — "that owner's to tear down at
    *its* lifetime end, not here" — and the streaming entry points that call
    `prebuild_factories` are that owner. They never did it, so a load-once model streamed
    through `iter_batches` held its GPU allocation, HTTP session, or database handle for the
    lifetime of the process. Not even garbage collection reclaimed it: the plan holds the
    instance, and the caller usually holds the plan.

    Best-effort and idempotent-ish by the same contract as `teardown_udf`: the rows are
    already produced, so a failing `close` must not fail the query, and a model whose `close`
    tolerates a second call is the ordinary shape.
    """
    from batcher.plan.logical import MapBatches
    from batcher.plan.visitor import walk

    for node in walk(plan):
        if not isinstance(node, MapBatches) or isinstance(node.fn, type):
            continue  # a *type* was never built here; `teardown_udf` owns that case
        close = getattr(node.fn, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()
