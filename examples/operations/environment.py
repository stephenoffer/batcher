"""What is installed, what the engine sees, and what to paste into a bug report.

Half of "it works on my machine" is an optional extra present in one environment and
absent in the other. These calls answer that in one line, and they are the first thing to
include when reporting a problem.

    python examples/operations/environment.py
"""

from __future__ import annotations

import batcher as bt
from batcher.config import env_var_names, option_names


def main() -> None:
    # Versions of Batcher, the compiled engine, Python, the platform, and the optional
    # backends that are actually importable.
    versions = bt.versions()
    print("versions:", versions)
    assert isinstance(versions, dict)
    assert len(versions) > 0

    # The same thing, printed for a bug report.
    bt.show_versions()

    # Which build of the native engine is loaded. Worth checking before trusting a
    # benchmark: a debug build and a release build are not comparable.
    if "engine_profile" in versions:
        print("engine profile:", versions["engine_profile"])

    # Every configurable option, and the environment variable that overrides it. This is
    # how you configure a container without editing code.
    names = option_names()
    env = env_var_names()
    print("options:", len(names), "| env vars:", len(env))
    assert len(names) > 10
    assert len(env) > 0
    # Each env var maps to an option.
    sample = sorted(env)[:3]
    print("sample env vars:", sample)

    # The package version is also on the module directly.
    assert isinstance(bt.__version__, str)
    print("batcher version:", bt.__version__)

    # A smoke test that the whole stack is wired: build a plan, run it, check the answer.
    out = bt.from_pydict({"x": [1, 2, 3]}).select(total=bt.col("x").sum()).to_pydict()
    assert out["total"] == [6]
    print("engine smoke test passed")


if __name__ == "__main__":
    main()
