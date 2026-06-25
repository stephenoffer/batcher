"""Auto-discovery of benchmark suite modules.

Importing a suite package imports every non-underscore submodule in it, so adding a
benchmark family is just dropping a ``.py`` file into ``suites/standard/`` or
``suites/operators/`` — there is no ``__init__`` to edit and no registration list to
keep in sync. Each module registers its cases through the ``suite(...)`` decorators at
import time.
"""

from __future__ import annotations

import importlib
import pkgutil


def import_submodules(package_name: str) -> None:
    """Import every non-underscore submodule of ``package_name`` for its side effects."""
    package = importlib.import_module(package_name)
    for info in pkgutil.iter_modules(package.__path__):
        if not info.name.startswith("_"):
            importlib.import_module(f"{package_name}.{info.name}")
