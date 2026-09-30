"""`metadata.require_durable` turns the silent in-process fallback into an error.

By default a durable backend that fails to construct degrades to an in-process store with a
warning, so a job keeps running while learning nothing across runs. With the flag set the
same misconfiguration raises instead.
"""

from __future__ import annotations

import pytest

from batcher import Config, MetadataConfig, config_context
from batcher._internal.errors import ConfigError
from batcher.core import runtime
from batcher.metadata.backends import InProcessBackend

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_hub():
    saved = (runtime._hub, runtime._hub_backend_key)
    runtime._hub, runtime._hub_backend_key = None, None
    yield
    runtime._hub, runtime._hub_backend_key = saved


def _broken(require_durable: bool) -> Config:
    # rocksdb with no uri is refused at construction, whether or not rocksdict is installed.
    return Config().replace(
        metadata=MetadataConfig(backend="rocksdb", uri=None, require_durable=require_durable)
    )


def test_the_default_still_degrades_to_in_process():
    with config_context(_broken(False)):
        hub = runtime.default_hub()
    assert isinstance(hub._backend, InProcessBackend)


def test_require_durable_raises_instead_of_degrading():
    with config_context(_broken(True)), pytest.raises(ConfigError, match="require_durable"):
        runtime.default_hub()
