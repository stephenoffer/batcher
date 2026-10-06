"""`Config.non_defaults(with_origin=True)`: which layer set each value.

The active config is built in layers -- defaults, the auto-detection step, the config
file, `BATCHER_*` variables, `set_config`/`set_option`, then any `config_context` -- and
"why is this set?" is the first question when one machine behaves differently from
another. Each test builds the static layers itself (`_initial_config` runs at import), so
the attribution is checked against a known file and environment rather than whatever the
test process happened to inherit.
"""

from __future__ import annotations

import json
import threading

import pytest

from batcher.config import (
    Config,
    ExecutionConfig,
    active_config,
    option_context,
    set_config,
    set_option,
)
from batcher.config import config as config_module

pytestmark = pytest.mark.unit


@pytest.fixture
def layers(tmp_path, monkeypatch):
    """Rebuild the static layers from a known file and environment; restore after."""
    path = tmp_path / "batcher.json"
    path.write_text(
        json.dumps({"execution": {"morsel_rows": 4096, "sort_merge_fanin": 8}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("BATCHER_CONFIG_FILE", str(path))
    monkeypatch.setenv("BATCHER_EXECUTION_SORT_MERGE_FANIN", "32")
    saved_layers = list(config_module._STATIC_LAYERS)
    saved_active = active_config()
    config_module._STATIC_LAYERS.clear()
    set_config(config_module._initial_config())
    # The static config is not an explicit choice.
    token = config_module._explicit_config.set(None)
    yield
    config_module._STATIC_LAYERS[:] = saved_layers
    set_config(saved_active)
    config_module._explicit_config.reset(token)


def _origin(key: str) -> str:
    return active_config().non_defaults(with_origin=True)[key]["origin"]


def test_a_value_from_the_config_file_is_attributed_to_the_file(layers):
    assert _origin("execution.morsel_rows") == "file"


def test_an_environment_variable_wins_over_the_file_and_says_so(layers):
    entry = active_config().non_defaults(with_origin=True)["execution.sort_merge_fanin"]
    assert entry == {"value": 32, "origin": "environment"}


def test_set_option_is_explicit(layers):
    set_option("execution.query_timeout_s", 30.0)
    assert _origin("execution.query_timeout_s") == "explicit"
    # The static layers set_option built on keep their own attribution.
    assert _origin("execution.morsel_rows") == "file"


def test_an_enclosing_scope_is_context(layers):
    with option_context("execution.morsel_rows", 2048):
        entry = active_config().non_defaults(with_origin=True)["execution.morsel_rows"]
        assert entry == {"value": 2048, "origin": "context"}
    assert _origin("execution.morsel_rows") == "file"


def test_a_value_set_on_a_detached_config_is_explicit(layers):
    """A config built by hand, never activated, owns the values no layer holds."""
    cfg = active_config().replace(execution=ExecutionConfig(morsel_rows=1024, sort_merge_fanin=32))
    origins = cfg.non_defaults(with_origin=True)
    assert origins["execution.morsel_rows"]["origin"] == "explicit"
    assert origins["execution.sort_merge_fanin"]["origin"] == "environment"


def test_only_non_default_values_are_reported_and_the_plain_form_is_unchanged(layers):
    with_origin = active_config().non_defaults(with_origin=True)
    plain = active_config().non_defaults()
    assert set(with_origin) == set(plain)
    assert {k: v["value"] for k, v in with_origin.items()} == plain
    assert Config().non_defaults(with_origin=True) == {}


def test_another_threads_set_option_is_not_attributed_here(layers):
    """`set_config` is context-scoped, so the layer it adds is too.

    A thread that calls `set_option` changes only its own active config; this thread's
    origin report must not see an "explicit" layer it never installed.
    """
    worker = threading.Thread(target=lambda: set_option("execution.morsel_rows", 1024))
    worker.start()
    worker.join()
    assert active_config().execution.morsel_rows == 4096
    assert _origin("execution.morsel_rows") == "file"
    assert "explicit" not in {name for name, _ in config_module.origin_layers()}
