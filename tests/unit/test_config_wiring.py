"""Config centralization wiring: the PID-gain fix, the engine-config wire shape,
and the file/env precedence layers added when Config became the single source of
truth for tunables.
"""

from __future__ import annotations

import json

from batcher.config import (
    Config,
    ExecutionConfig,
    MemoryConfig,
    PIDConfig,
    config_context,
)


def test_pid_defaults_are_canonical_gains():
    """Regression for the ki/kd transposition: the only PID controller that exists
    (bc-udf BatchSizeController / ml.inference._LatencyController) uses
    kp=0.4, ki=0.05, kd=0.1. Config must match — not the old swapped 0.4/0.1/0.05."""
    pid = Config().pid
    assert (pid.kp, pid.ki, pid.kd) == (0.4, 0.05, 0.1)
    assert pid.integral_clamp == 5.0
    assert pid.max_step_fraction == 0.5


def test_latency_controller_reads_config_gains():
    from batcher.ml.inference import _LatencyController

    scoped = Config().replace(pid=PIDConfig(kp=0.9, ki=0.0, kd=0.0))
    with config_context(scoped):
        ctrl = _LatencyController(target_ms=10.0, min_rows=1, max_rows=1000, initial=100)
        assert ctrl._pid.kp == 0.9
        # Under-target latency (headroom) grows the batch with the scoped gains.
        grown = ctrl.update(observed_ms=5.0)
        assert grown >= 100


def test_engine_config_json_shape_and_defaults():
    """The wire contract with bc_ir::EngineConfig: morsel_rows + morsel_bytes +
    parallelism + the spill envelope. By default the budget is 0 (unbounded — the
    in-memory fast path is unchanged) and spill_dir is None."""
    payload = json.loads(Config().engine_config_json())
    assert payload == {
        "morsel_rows": 16_384,
        "morsel_bytes": 1 << 20,
        "parallelism": 0,
        "memory_budget_bytes": 0,
        "spill_dir": None,
    }

    scoped = Config().replace(
        execution=ExecutionConfig(morsel_rows=4096, morsel_bytes=1 << 15, parallelism=3)
    )
    payload = json.loads(scoped.engine_config_json())
    assert payload == {
        "morsel_rows": 4096,
        "morsel_bytes": 1 << 15,
        "parallelism": 3,
        "memory_budget_bytes": 0,
        "spill_dir": None,
    }


def test_engine_config_ships_spill_budget_when_capped():
    """Setting `memory.max_memory_bytes` opts into out-of-core spilling: the budget
    shipped to Rust is the cap scaled by `hard_limit`, and `spill_dir` rides along."""
    scoped = Config().replace(
        memory=MemoryConfig(max_memory_bytes=1_000_000, hard_limit=0.9, spill_dir="/scratch")
    )
    payload = json.loads(scoped.engine_config_json())
    assert payload["memory_budget_bytes"] == 900_000
    assert payload["spill_dir"] == "/scratch"


def test_from_env_overlays_nested_sections():
    cfg = Config.from_env(
        {
            "BATCHER_EXECUTION_MORSEL_ROWS": "4096",
            "BATCHER_OPTIMIZER_CARDINALITY_EQ_SELECTIVITY": "0.25",
            "BATCHER_PID_KP": "0.7",
        }
    )
    assert cfg.execution.morsel_rows == 4096
    assert cfg.optimizer.cardinality.eq_selectivity == 0.25
    assert cfg.pid.kp == 0.7
    # Untouched fields keep their defaults.
    assert cfg.execution.parallelism == 0


def test_from_file_then_env_precedence(tmp_path):
    """defaults < file < env: env wins where both set a key, file wins over default."""
    path = tmp_path / "cfg.json"
    path.write_text(
        json.dumps(
            {
                "execution": {"morsel_rows": 1024, "parallelism": 2},
                "memory": {"soft_limit": 0.5},
            }
        )
    )
    filed = Config.from_file(path)
    assert filed.execution.morsel_rows == 1024
    assert filed.memory.soft_limit == 0.5

    # env overlays on top of the file-derived base: morsel_rows from env wins,
    # parallelism (file-only) survives, soft_limit (file-only) survives.
    layered = Config.from_env({"BATCHER_EXECUTION_MORSEL_ROWS": "8192"}, base=filed)
    assert layered.execution.morsel_rows == 8192
    assert layered.execution.parallelism == 2
    assert layered.memory.soft_limit == 0.5


def test_unknown_cardinality_threshold_is_consistent():
    """The 1e11/1e12 inconsistency is resolved: one home, one value."""
    assert Config().optimizer.cardinality.unknown_rows == 1e12


def test_validate_rejects_bad_values():
    """Out-of-range / inconsistent tunables raise a typed ConfigError, early."""
    import pytest

    from batcher._internal.errors import ConfigError
    from batcher.config import DistributedConfig

    # Defaults are valid and validate() returns self for chaining.
    assert Config().validate() is not None

    with pytest.raises(ConfigError, match="soft_limit"):
        Config().replace(memory=MemoryConfig(soft_limit=0.95, hard_limit=0.90)).validate()
    with pytest.raises(ConfigError, match="recovery_max_attempts"):
        Config().replace(distributed=DistributedConfig(recovery_max_attempts=0)).validate()
    with pytest.raises(ConfigError, match="task_max_retries"):
        Config().replace(distributed=DistributedConfig(task_max_retries=-1)).validate()
    with pytest.raises(ConfigError, match="flight_idle_timeout_s"):
        Config().replace(distributed=DistributedConfig(flight_idle_timeout_s=0)).validate()


def test_set_config_validates(monkeypatch):
    """set_config / config_context reject bad config at the entry point."""
    import pytest

    from batcher._internal.errors import ConfigError
    from batcher.config import set_config

    bad = Config().replace(memory=MemoryConfig(soft_limit=2.0))
    with pytest.raises(ConfigError):
        set_config(bad)
    with pytest.raises(ConfigError), config_context(bad):
        pass
