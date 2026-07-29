"""Configuring the engine: options, scoped overrides, and profiles.

Configuration is a value, not global mutable state you have to remember to undo.
``option_context`` and ``config_context`` scope an override to a block, so a memory-tight
step cannot leak its settings into the rest of the program.

    python examples/operations/configuration.py
"""

from __future__ import annotations

import dataclasses

import batcher as bt
from batcher.config import (
    active_config,
    config_context,
    config_to_dict,
    describe_options,
    get_option,
    option_context,
    option_names,
    reset_option,
    set_option,
)


def main() -> None:
    # What can be configured, and what it means.
    names = option_names()
    print("option count:", len(names))
    assert len(names) > 10
    described = describe_options()
    assert isinstance(described, str | dict)

    # Read a single option.
    morsel = get_option("execution.morsel_rows")
    print("morsel_rows:", morsel)
    assert isinstance(morsel, int)

    # Scope an override to a block, which is the recommended shape.
    with option_context("execution.morsel_rows", 4096):
        assert get_option("execution.morsel_rows") == 4096
        out = bt.from_pydict({"x": [1, 2, 3]}).select(y=bt.col("x") * 2).to_pydict()
        assert out["y"] == [2, 4, 6]
    # The override is gone outside the block.
    assert get_option("execution.morsel_rows") == morsel

    # A global set, with an explicit reset.
    set_option("execution.morsel_rows", 8192)
    assert get_option("execution.morsel_rows") == 8192
    reset_option("execution.morsel_rows")
    assert get_option("execution.morsel_rows") == morsel

    # The whole config as an immutable value you can print, diff, and review.
    cfg = active_config()
    as_dict = config_to_dict(cfg)
    print("config sections:", sorted(as_dict)[:6])
    assert "execution" in as_dict
    assert "memory" in as_dict

    # Build a variant with `replace` and install it for a block. Nothing mutates.
    tight = cfg.replace(
        memory=dataclasses.replace(cfg.memory, default_total_bytes=256 * 1024 * 1024)
    )
    assert tight.memory.default_total_bytes != cfg.memory.default_total_bytes
    with config_context(tight):
        assert active_config().memory.default_total_bytes == 256 * 1024 * 1024
        result = bt.from_pydict({"x": list(range(100))}).select(total=bt.col("x").sum()).to_pydict()
        assert result["total"] == [4950]
    assert active_config().memory.default_total_bytes == cfg.memory.default_total_bytes


if __name__ == "__main__":
    main()
