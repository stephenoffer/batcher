"""Converting a `Config` to and from dicts, files, and environment-variable names.

`Config.from_file` reads a JSON document; this module is the rest of that story — the
`to_dict` direction that makes the round-trip closed, TOML and YAML readers for the file
formats teams actually keep configuration in, and the `diff` / `non_defaults` views that
answer "what is actually set here?" without printing 180 lines of defaults.

Kept beside `config.py` rather than inside it: the config contract is the dataclasses, and
serialization is a second concern that would push that already-large module further past
the size limit for no gain in cohesion.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import ConfigError

if TYPE_CHECKING:
    from batcher.config.config import Config

__all__ = ["config_to_dict", "env_var_names"]


def config_to_dict(config: Config, *, only_non_default: bool = False) -> dict[str, Any]:
    """Convert a `Config` to a nested plain-dict, ready for JSON/TOML/YAML.

    The inverse of `Config.from_dict`, and closed under round-trip: feeding the result
    back produces an equal `Config`. Values are plain scalars, so the dict is safe to
    `json.dumps`, log, or ship as part of a job manifest.

    Examples:
        .. doctest::

            >>> from batcher.config import Config, config_to_dict
            >>> config_to_dict(Config())["execution"]["morsel_rows"]
            16384

        .. doctest::

            >>> from batcher.config import Config, ExecutionConfig, config_to_dict
            >>> cfg = Config().replace(execution=ExecutionConfig(morsel_rows=4096))
            >>> config_to_dict(cfg, only_non_default=True)
            {'execution': {'morsel_rows': 4096}}

    Args:
        config: The config to convert.
        only_non_default: Emit only the values that differ from the built-in defaults,
            producing the minimal document that reproduces this config.

    Returns:
        A nested dict mirroring the config's section structure.
    """
    from batcher.config.config import Config as _Config

    return _to_dict(config, _Config() if only_non_default else None)


def _to_dict(obj: object, default: object | None) -> dict[str, Any]:
    """Recursively convert a frozen config object, pruning against `default` when given."""
    out: dict[str, Any] = {}
    for field in dataclasses.fields(obj):  # type: ignore[arg-type]
        value = getattr(obj, field.name)
        base = None if default is None else getattr(default, field.name)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            nested = _to_dict(value, base)
            if nested or default is None:
                out[field.name] = nested
        elif default is None or value != base:
            out[field.name] = value
    return out


def env_var_names(config: Config | None = None) -> dict[str, str]:
    """Map every ``BATCHER_*`` environment variable onto the option path it sets.

    Every leaf tunable has an environment variable, derived mechanically by upper-casing
    its path and joining with underscores, so ``execution.morsel_rows`` is
    ``BATCHER_EXECUTION_MORSEL_ROWS``. This function is that mapping made explicit —
    useful for generating deployment manifests, and for checking a variable name before
    you bake it into a Dockerfile and wonder why nothing changed.

    Examples:
        .. doctest::

            >>> from batcher.config import env_var_names
            >>> env_var_names()["BATCHER_EXECUTION_MORSEL_ROWS"]
            'execution.morsel_rows'

    Args:
        config: The config whose fields to enumerate, or None for the defaults. The
            values are irrelevant; only the field structure is read.

    Returns:
        A dict mapping environment-variable name to dotted option path.
    """
    from batcher.config.config import Config as _Config
    from batcher.config.options import _leaves

    source = _Config() if config is None else config
    return {f"BATCHER_{path.upper().replace('.', '_')}": path for path, _ in _leaves(source)}


def read_document(path: str | os.PathLike[str], *, fmt: str | None = None) -> dict[str, Any]:
    """Parse a JSON, TOML, or YAML config document into a nested dict.

    TOML uses the standard-library `tomllib`. YAML needs `pyyaml` installed, and says so
    rather than failing on an import error deep in a traceback.

    Args:
        path: The document to read.
        fmt: Force a parser (``"json"``, ``"toml"``, ``"yaml"``). None selects it from
            the file suffix, defaulting to JSON.

    Returns:
        The parsed document as a nested dict.

    Raises:
        ConfigError: If the file is missing, unparseable, YAML support is not installed,
            or the document's top level is not a mapping.
    """
    file = Path(path)
    if not file.is_file():
        msg = f"config file not found: {file}"
        raise ConfigError(msg)
    text = file.read_text()
    suffix = f".{fmt.lower().lstrip('.')}" if fmt else file.suffix.lower()
    try:
        data = _parse(text, suffix)
    except ConfigError:
        raise
    except Exception as exc:
        msg = f"could not parse config file {file}: {exc}"
        raise ConfigError(msg) from exc
    if not isinstance(data, dict):
        msg = (
            f"config file {file} must contain a mapping at the top level, got {type(data).__name__}"
        )
        raise ConfigError(msg)
    return data


def _parse(text: str, suffix: str) -> Any:
    """Dispatch to the parser for `suffix`, raising `ConfigError` if it is unavailable."""
    if suffix == ".toml":
        import tomllib

        return tomllib.loads(text)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            msg = "reading a YAML config needs pyyaml — install it with `pip install pyyaml`"
            raise ConfigError(msg) from exc
        return yaml.safe_load(text)
    return json.loads(text)
