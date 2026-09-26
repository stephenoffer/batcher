"""Regression: ``BATCHER_<SECTION>_<FIELD>`` values are parsed into the declared field type.

Two defects. A sequence or mapping field stored the raw string, so
``BATCHER_EXECUTION_UDF_ENV_ALLOWLIST=HF_TOKEN,FOO`` became ``"HF_TOKEN,FOO"`` and
``tuple(allowlist)`` downstream split it into single characters, silently scrubbing
``HF_TOKEN`` from every isolated UDF. And a malformed scalar raised a bare ``ValueError``
from inside ``import batcher`` naming neither the variable nor the option.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import ConfigError
from batcher.config.config import Config


@pytest.mark.unit
def test_udf_env_allowlist_is_comma_split() -> None:
    cfg = Config.from_env({"BATCHER_EXECUTION_UDF_ENV_ALLOWLIST": "HF_TOKEN, FOO"})
    assert cfg.execution.udf_env_allowlist == ("HF_TOKEN", "FOO")


@pytest.mark.unit
def test_empty_sequence_env_is_the_empty_tuple() -> None:
    cfg = Config.from_env({"BATCHER_EXECUTION_UDF_ENV_ALLOWLIST": ""})
    assert cfg.execution.udf_env_allowlist == ()


@pytest.mark.unit
def test_quantile_probs_elements_are_floats() -> None:
    cfg = Config.from_env({"BATCHER_OPTIMIZER_QUANTILE_PROBS": "0,0.5,1"})
    assert cfg.optimizer.quantile_probs == (0.0, 0.5, 1.0)
    assert all(type(p) is float for p in cfg.optimizer.quantile_probs)


@pytest.mark.unit
def test_fixed_length_tuple_is_parsed_and_length_checked() -> None:
    cfg = Config.from_env({"BATCHER_DISTRIBUTED_SHUFFLE_PORT_RANGE": "9000,9100"})
    assert cfg.distributed.shuffle_port_range == (9000, 9100)
    with pytest.raises(ConfigError, match="BATCHER_DISTRIBUTED_SHUFFLE_PORT_RANGE"):
        Config.from_env({"BATCHER_DISTRIBUTED_SHUFFLE_PORT_RANGE": "9000"})


@pytest.mark.unit
def test_mapping_field_is_json() -> None:
    cfg = Config.from_env({"BATCHER_DISTRIBUTED_RUNTIME_ENV": '{"pip": ["x"]}'})
    assert cfg.distributed.runtime_env == {"pip": ["x"]}
    with pytest.raises(ConfigError, match="BATCHER_DISTRIBUTED_RUNTIME_ENV"):
        Config.from_env({"BATCHER_DISTRIBUTED_RUNTIME_ENV": "[1, 2]"})


@pytest.mark.unit
def test_malformed_scalar_names_the_variable() -> None:
    with pytest.raises(ConfigError, match=r"BATCHER_EXECUTION_MORSEL_ROWS='16k': expected int"):
        Config.from_env({"BATCHER_EXECUTION_MORSEL_ROWS": "16k"})
