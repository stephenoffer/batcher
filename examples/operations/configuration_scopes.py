"""Setting engine options, and keeping the change scoped.

A global `set_option` is a process-wide change that outlives the function that made it.
`option_context` is the same setting with a lifetime, and it is what you want in library
code — the caller's configuration is not yours to keep.

    python examples/operations/configuration_scopes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col
from batcher.config import get_option, option_context, option_names, set_option


def main() -> None:
    lineitem = tpch("lineitem").select("l_orderkey", "l_quantity")

    print("configurable options:", len(option_names()))
    assert len(option_names()) > 10

    original = get_option("execution.morsel_rows")
    print("morsel_rows:", original)

    # Scoped: the option reverts when the block exits, even on an exception.
    with option_context("execution.morsel_rows", 4096):
        assert get_option("execution.morsel_rows") == 4096
        small_morsels = lineitem.agg(total=col("l_quantity").sum()).to_pydict()
    assert get_option("execution.morsel_rows") == original

    # The morsel size is a scheduling knob: it changes how work is divided, never the
    # answer. That is the invariant worth pinning.
    default_morsels = lineitem.agg(total=col("l_quantity").sum()).to_pydict()
    assert small_morsels == default_morsels
    print("same answer at both morsel sizes:", default_morsels)

    # Global, for a script that owns the process. Restore it when you are done.
    try:
        set_option("execution.morsel_rows", 8192)
        assert get_option("execution.morsel_rows") == 8192
        again = lineitem.agg(total=col("l_quantity").sum()).to_pydict()
        assert again == default_morsels
    finally:
        set_option("execution.morsel_rows", original)
    assert get_option("execution.morsel_rows") == original


if __name__ == "__main__":
    main()
