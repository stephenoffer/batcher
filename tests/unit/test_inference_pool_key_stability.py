"""A warm inference pool's key must outlive the callable whose address formed it.

`_pipeline_signature` keys a session-warm actor pool on `id(node.fn)`. An address is only
unique among *live* objects, so if the model callable is freed while its pool stays warm,
CPython may hand the same address to a different model — whose pipeline then matches the
stale key and runs inference on the wrong model, silently and with no error. The pool
registry therefore pins the callables it keyed on, exactly as `kyber.plan_cache` does.
"""

from __future__ import annotations

import batcher.dist.executors.map as dmap


class _Model:
    def __call__(self, batch):  # pragma: no cover - never invoked
        return batch


def _plan_with(fn):
    """A one-node `MapBatches` chain over `fn` (the shape `_pipeline_signature` walks)."""
    from batcher.plan.logical import MapBatches

    return MapBatches(input=None, fn=fn, batch_size=None)


def test_a_cached_pool_pins_the_callable_its_key_was_built_from():
    """Storing a pool must keep the callable alive, so its `id()` cannot be recycled."""
    dmap._POOL_KEEPALIVE.clear()
    registry: dict[tuple, list] = {}
    plan = _plan_with(_Model())
    sig = dmap._pipeline_signature(plan)

    registry[sig] = ["actor"]
    dmap._pin_pool_key(sig, plan)

    assert sig in dmap._POOL_KEEPALIVE
    assert dmap._POOL_KEEPALIVE[sig] == dmap._pipeline_functions(plan)
    dmap._POOL_KEEPALIVE.clear()


def test_two_live_models_never_share_a_pool_key():
    """The property the pin protects: distinct live callables key distinct pools.

    Pinned, the first model cannot be collected while its pool is cached, so the second
    model cannot land on its address and inherit its actors.
    """
    dmap._POOL_KEEPALIVE.clear()
    first = _plan_with(_Model())
    sig_first = dmap._pipeline_signature(first)
    dmap._pin_pool_key(sig_first, first)

    second = _plan_with(_Model())
    assert dmap._pipeline_signature(second) != sig_first
    dmap._POOL_KEEPALIVE.clear()


def test_tearing_down_pools_releases_their_pins():
    """A pin must not outlive its pool, or the registry leaks every model ever loaded."""
    dmap._POOL_KEEPALIVE.clear()
    plan = _plan_with(_Model())
    sig = dmap._pipeline_signature(plan)
    registry: dict[tuple, list] = {sig: []}
    dmap._pin_pool_key(sig, plan)

    dmap._shutdown_pools(registry)

    assert dmap._POOL_KEEPALIVE == {}
    assert registry == {}
