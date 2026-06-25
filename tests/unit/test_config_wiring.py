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


# The performance-threshold knobs (mirror bc_arrow::RuntimeTuning) that ride along
# in every engine-config payload. Their defaults equal the Rust EngineConfig::default().
_TUNING_DEFAULTS = {
    "bloom_fp_rate": 0.01,
    "bloom_min_build_rows": 1 << 16,
    "window_parallel_row_threshold": 1 << 15,
    "radix_parallel_threshold": 200_000,
    "sort_merge_fanin": 16,
    "skew_bucket_factor": 4,
    "skew_min_bucket_rows": 4 * 16_384,
    "skew_min_bucket_bytes": 4 * (1 << 20),
}


def test_engine_config_json_shape_and_defaults():
    """The wire contract with bc_ir::EngineConfig: morsel_rows + morsel_bytes +
    parallelism + the spill envelope + the performance-threshold knobs. By default the
    budget is 0 (unbounded — the in-memory fast path is unchanged) and spill_dir is
    None."""
    payload = json.loads(Config().engine_config_json())
    assert payload == {
        "morsel_rows": 16_384,
        "morsel_bytes": 1 << 20,
        "parallelism": 0,
        "memory_budget_bytes": 0,
        "spill_dir": None,
        "fuse_linear": True,
        **_TUNING_DEFAULTS,
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
        "fuse_linear": True,
        **_TUNING_DEFAULTS,
    }


def test_tuning_knobs_ship_and_are_overridable():
    """The performance-threshold knobs serialize with their defaults and a non-default
    value flows through to the wire payload (where the Rust data plane consumes it)."""
    payload = json.loads(Config().engine_config_json())
    for key, default in _TUNING_DEFAULTS.items():
        assert payload[key] == default, key

    scoped = Config().replace(
        execution=ExecutionConfig(radix_parallel_threshold=50_000, bloom_fp_rate=0.05)
    )
    payload = json.loads(scoped.engine_config_json())
    assert payload["radix_parallel_threshold"] == 50_000
    assert payload["bloom_fp_rate"] == 0.05
    # Untouched tuning knobs keep their defaults.
    assert payload["sort_merge_fanin"] == _TUNING_DEFAULTS["sort_merge_fanin"]


def test_engine_config_with_op_budgets_adds_string_keyed_map():
    """`engine_config_json_with` adds the per-operator spill budgets as a
    string-keyed object (serde_json parses the keys back to op ids); an empty map
    reproduces the base wire shape exactly so callers with no PhysicalOp DAG are
    unaffected."""
    cfg = Config()
    assert cfg.engine_config_json_with({}) == cfg.engine_config_json()

    payload = json.loads(cfg.engine_config_json_with({0: 1_048_576, 3: 2_048}))
    assert payload["op_budgets"] == {"0": 1_048_576, "3": 2_048}
    # The base knobs still ride along unchanged.
    assert payload["morsel_rows"] == 16_384
    assert payload["memory_budget_bytes"] == 0


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
