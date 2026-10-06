"""A config's printable views never carry its secrets, and a redacted one cannot be loaded.

`Config.__repr__` and `to_dict()` printed `distributed.shuffle_token` in plain text, and a
repr is exactly what lands in a log line or a traceback. Redacting the views is half the
fix. The other half is that a redacted dict must not reload as though `"<redacted>"` were
the secret: a shuffle fleet would then authenticate with the placeholder and fail somewhere
far from the cause.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import ConfigError
from batcher.config import (
    Config,
    DistributedConfig,
    ObservabilityConfig,
    config_context,
    config_to_dict,
    describe_options,
)
from batcher.config.serde import REDACTED, SECRET_OPTIONS

pytestmark = pytest.mark.unit

TOKEN = "s3cret-shuffle-token"
API_KEY = "s3cret-lineage-key"


def _secret_config() -> Config:
    return Config().replace(
        distributed=DistributedConfig(shuffle_token=TOKEN),
        observability=ObservabilityConfig(openlineage_api_key=API_KEY),
    )


def _leaks(text: str) -> bool:
    return TOKEN in text or API_KEY in text


def test_the_secret_options_are_real_option_paths():
    """A typo in the tuple would silently redact nothing."""
    cfg = _secret_config()
    for path in SECRET_OPTIONS:
        section, name = path.split(".")
        assert getattr(getattr(cfg, section), name) in (TOKEN, API_KEY)


def test_repr_redacts_every_secret():
    text = repr(_secret_config())
    assert not _leaks(text)
    assert f"distributed.shuffle_token={REDACTED!r}" in text


def test_to_dict_redacts_by_default():
    for only_non_default in (False, True):
        out = _secret_config().to_dict(only_non_default=only_non_default)
        assert out["distributed"]["shuffle_token"] == REDACTED
        assert out["observability"]["openlineage_api_key"] == REDACTED
        assert not _leaks(repr(out))


def test_non_defaults_diff_and_describe_options_redact():
    cfg = _secret_config()
    assert cfg.non_defaults()["distributed.shuffle_token"] == REDACTED
    assert cfg.diff(Config())["observability.openlineage_api_key"] == REDACTED
    origins = cfg.non_defaults(with_origin=True)
    assert origins["distributed.shuffle_token"]["value"] == REDACTED
    with config_context(cfg):
        described = describe_options("shuffle_token")
    assert "shuffle_token" in described
    assert not _leaks(described)


def test_an_unset_secret_is_shown_as_unset():
    """None / "" give nothing away, and "no token configured" is the useful thing to see."""
    out = Config().to_dict()
    assert out["distributed"]["shuffle_token"] is None
    assert out["observability"]["openlineage_api_key"] == ""


def test_the_unredacted_round_trip_is_closed():
    cfg = Config.from_dict(_secret_config().to_dict(redact_secrets=False))
    assert cfg.distributed.shuffle_token == TOKEN
    assert cfg.observability.openlineage_api_key == API_KEY
    assert Config.from_dict(cfg.to_dict(redact_secrets=False)) == cfg


def test_a_redacted_dict_without_secrets_still_round_trips():
    resolved = Config.from_dict(Config().to_dict())
    assert Config.from_dict(resolved.to_dict()) == resolved


@pytest.mark.parametrize("path", sorted(SECRET_OPTIONS))
def test_a_redacted_secret_cannot_be_loaded(path):
    """Loading the placeholder fails naming the field, rather than using it as the secret."""
    section, name = path.split(".")
    with pytest.raises(ConfigError, match=rf"{path}.*redact"):
        Config.from_dict({section: {name: REDACTED}})


def test_reloading_a_redacted_to_dict_fails_loudly():
    with pytest.raises(ConfigError, match="redact_secrets=False"):
        Config.from_dict(config_to_dict(_secret_config()))
