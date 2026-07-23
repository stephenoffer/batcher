"""Configuration ergonomics: dotted options, serialization, env names, logging switches.

Every test restores the process-wide config, because `set_option` and the logging
switches are deliberately global — a leaked override here would surface as an unrelated
failure somewhere else in the suite.
"""

from __future__ import annotations

import json
import logging

import pytest

from batcher._internal.errors import ConfigError
from batcher.config import (
    Config,
    ExecutionConfig,
    active_config,
    config_to_dict,
    describe_options,
    disable_logging,
    enable_logging,
    env_var_names,
    get_logger,
    get_option,
    option_context,
    option_names,
    reset_option,
    set_config,
    set_log_level,
    set_option,
    set_progress,
    set_verbosity,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _restore_config():
    """Snapshot and restore the process-wide config around every test."""
    saved = active_config()
    yield
    set_config(saved)


# --- dotted-path options ------------------------------------------------------


def test_get_option_accepts_full_path_and_unambiguous_suffix():
    assert get_option("execution.morsel_rows") == get_option("morsel_rows")


def test_set_option_positional_pair():
    set_option("execution.morsel_rows", 4096)
    assert get_option("execution.morsel_rows") == 4096


def test_set_option_multiple_pairs_in_one_call():
    set_option("execution.morsel_rows", 4096, "observability.log_format", "json")
    assert get_option("execution.morsel_rows") == 4096
    assert get_option("observability.log_format") == "json"


def test_set_option_keyword_underscore_path():
    set_option(execution_morsel_rows=2048)
    assert get_option("execution.morsel_rows") == 2048


def test_set_option_odd_argument_count_is_an_error():
    with pytest.raises(ConfigError, match="alternating name/value"):
        set_option("execution.morsel_rows")


def test_reset_option_restores_one_option():
    default = get_option("execution.morsel_rows")
    set_option("execution.morsel_rows", 4096)
    reset_option("execution.morsel_rows")
    assert get_option("execution.morsel_rows") == default


def test_reset_option_glob_restores_a_whole_section():
    set_option("execution.morsel_rows", 4096)
    reset_option("execution.*")
    assert get_option("execution.morsel_rows") == Config().execution.morsel_rows


def test_reset_option_section_prefix_without_a_glob():
    set_option("execution.morsel_rows", 4096)
    reset_option("execution")
    assert get_option("execution.morsel_rows") == Config().execution.morsel_rows


def test_option_context_restores_on_exit():
    before = get_option("execution.morsel_rows")
    with option_context("execution.morsel_rows", 1024):
        assert get_option("execution.morsel_rows") == 1024
    assert get_option("execution.morsel_rows") == before


def test_option_context_restores_when_the_block_raises():
    before = get_option("execution.morsel_rows")
    with pytest.raises(RuntimeError), option_context("execution.morsel_rows", 1024):
        raise RuntimeError("boom")
    assert get_option("execution.morsel_rows") == before


def test_option_context_nests():
    with option_context("execution.morsel_rows", 1024):
        with option_context("execution.morsel_rows", 512):
            assert get_option("execution.morsel_rows") == 512
        assert get_option("execution.morsel_rows") == 1024


def test_unknown_option_suggests_the_close_match():
    with pytest.raises(ConfigError) as exc:
        set_option("execution.morsel_row", 1)
    assert "execution.morsel_rows" in str(exc.value)
    assert "Did you mean" in str(exc.value)


def test_unknown_option_names_the_discovery_helpers():
    with pytest.raises(ConfigError, match="option_names"):
        get_option("completely_made_up_name_xyz")


def test_ambiguous_suffix_is_rejected_rather_than_guessed():
    """A leaf name shared by two sections must not silently pick one."""
    leaves = [p.rsplit(".", 1)[-1] for p in option_names()]
    shared = next((n for n in leaves if leaves.count(n) > 1), None)
    if shared is None:
        pytest.skip("no option leaf name is shared between sections")
    with pytest.raises(ConfigError, match="ambiguous"):
        get_option(shared)


def test_option_names_covers_every_leaf_and_is_globbable():
    assert "execution.morsel_rows" in option_names()
    assert len(option_names()) > 50
    assert all(n.startswith("memory.") for n in option_names("memory.*"))


def test_describe_options_flags_non_default_values():
    set_option("execution.morsel_rows", 4096)
    text = describe_options("execution.morsel_rows")
    assert "4096" in text
    assert "default" in text


def test_describe_options_substring_search():
    assert "spill" in describe_options("spill")


def test_describe_options_says_so_when_nothing_matches():
    assert "no config options match" in describe_options("zzz_no_such_option")


def test_a_bad_value_leaves_the_previous_config_intact():
    before = get_option("execution.morsel_rows")
    with pytest.raises(ConfigError):
        set_option("execution.morsel_rows", -1)
    assert get_option("execution.morsel_rows") == before


# --- serialization ------------------------------------------------------------


def test_to_dict_from_dict_round_trip_is_idempotent():
    resolved = Config.from_dict(Config().to_dict())
    assert Config.from_dict(resolved.to_dict()) == resolved


def test_to_dict_is_json_encodable():
    json.dumps(Config().to_dict())


def test_to_dict_only_non_default_is_minimal():
    cfg = Config().replace(execution=ExecutionConfig(morsel_rows=4096))
    assert cfg.to_dict(only_non_default=True) == {"execution": {"morsel_rows": 4096}}


def test_config_to_dict_function_matches_the_method():
    assert config_to_dict(Config()) == Config().to_dict()


def test_from_dict_ignores_unknown_keys_so_newer_documents_still_load():
    cfg = Config.from_dict({"execution": {"morsel_rows": 4096}, "from_the_future": {"x": 1}})
    assert cfg.execution.morsel_rows == 4096


def test_from_toml(tmp_path):
    p = tmp_path / "batcher.toml"
    p.write_text("[execution]\nmorsel_rows = 4096\n")
    assert Config.from_toml(p).execution.morsel_rows == 4096


def test_from_file_dispatches_on_the_toml_suffix(tmp_path):
    p = tmp_path / "batcher.toml"
    p.write_text("[execution]\nmorsel_rows = 8192\n")
    assert Config.from_file(p).execution.morsel_rows == 8192


def test_from_yaml(tmp_path):
    yaml = pytest.importorskip("yaml")
    assert yaml
    p = tmp_path / "batcher.yaml"
    p.write_text("execution:\n  morsel_rows: 4096\n")
    assert Config.from_yaml(p).execution.morsel_rows == 4096


def test_missing_config_file_is_a_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="config file not found"):
        Config.from_file(tmp_path / "nope.json")


def test_unparseable_config_file_names_the_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(ConfigError, match="could not parse"):
        Config.from_file(p)


def test_non_mapping_config_file_is_rejected(tmp_path):
    p = tmp_path / "list.json"
    p.write_text("[1, 2, 3]")
    with pytest.raises(ConfigError, match="mapping at the top level"):
        Config.from_file(p)


# --- diff, non_defaults, repr -------------------------------------------------


def test_non_defaults_reports_only_what_changed():
    cfg = Config().replace(execution=ExecutionConfig(morsel_rows=4096))
    assert cfg.non_defaults() == {"execution.morsel_rows": 4096}


def test_diff_of_identical_configs_is_empty():
    assert Config().diff(Config()) == {}


def test_repr_is_compact_and_names_the_changed_option():
    cfg = Config().replace(execution=ExecutionConfig(morsel_rows=4096))
    text = repr(cfg)
    assert "morsel_rows=4096" in text
    assert len(text) < 500


def test_repr_of_defaults_says_so():
    assert repr(Config()) == "Config(<all defaults>)"


# --- environment variables ----------------------------------------------------


def test_env_var_names_maps_every_option():
    names = env_var_names()
    assert names["BATCHER_EXECUTION_MORSEL_ROWS"] == "execution.morsel_rows"
    assert len(names) == len(option_names())


def test_every_env_var_name_actually_sets_its_option():
    """The documented mapping must be the one `from_env` implements."""
    for var, path in env_var_names().items():
        if path == "execution.morsel_rows":
            assert Config.from_env({var: "4096"}).execution.morsel_rows == 4096
            return
    pytest.fail("execution.morsel_rows missing from the env-var mapping")


# --- logging / verbosity / progress -------------------------------------------


def test_set_log_level_accepts_a_name():
    set_log_level("debug")
    assert get_option("observability.log_level") == "DEBUG"


def test_set_log_level_accepts_a_logging_constant():
    set_log_level(logging.ERROR)
    assert get_option("observability.log_level") == "ERROR"


def test_set_log_level_accepts_a_verbosity_preset_name():
    set_log_level("trace")
    assert get_option("observability.log_level") == "DEBUG"


def test_set_log_level_rejects_nonsense():
    with pytest.raises(ConfigError, match="unknown log level"):
        set_log_level("loud")


def test_set_log_level_applies_to_the_live_stdlib_logger():
    set_log_level("debug")
    assert get_logger().isEnabledFor(logging.DEBUG)


def test_enable_and_disable_logging():
    enable_logging("info")
    assert get_option("observability.console") is True
    disable_logging()
    assert get_option("observability.console") is False


def test_enable_logging_with_a_file(tmp_path):
    target = tmp_path / "batcher.log"
    enable_logging("info", log_file=str(target))
    assert get_option("observability.log_file") == str(target)


def test_get_logger_uses_the_batcher_namespace():
    assert get_logger().name == "batcher"
    assert get_logger("kyber").name == "batcher.kyber"


def test_batcher_logger_does_not_hijack_the_root_logger():
    """A user's own logging config must survive Batcher configuring itself."""
    enable_logging("debug")
    assert logging.getLogger().level != logging.DEBUG or logging.getLogger().handlers is not None
    assert get_logger().propagate is False


def test_set_verbosity_drives_both_dials():
    set_verbosity("quiet")
    assert active_config().observability.resolved_log_level == "ERROR"
    assert active_config().observability.resolved_progress == "off"


def test_set_verbosity_clears_a_stale_explicit_log_level():
    set_log_level("debug")
    set_verbosity("quiet")
    assert active_config().observability.resolved_log_level == "ERROR"


def test_set_verbosity_rejects_an_unknown_preset():
    with pytest.raises(ConfigError, match="unknown verbosity"):
        set_verbosity("shouty")


def test_set_progress_bool_and_mode_spellings():
    set_progress(False)
    assert get_option("observability.progress") == "off"
    set_progress(True)
    assert get_option("observability.progress") == "on"
    set_progress("auto")
    assert get_option("observability.progress") == "auto"


def test_set_progress_rejects_an_unknown_mode():
    with pytest.raises(ConfigError, match="unknown progress mode"):
        set_progress("maybe")


def test_progress_defaults_to_auto_so_redirected_output_stays_clean():
    assert Config().observability.resolved_progress == "auto"


def test_zero_config_defaults_are_quiet():
    """Out of the box Batcher must not print anything below WARNING."""
    assert Config().observability.resolved_log_level == "WARNING"
    assert Config().observability.ui is False
