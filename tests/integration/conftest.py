"""Shared setup for the distributed integration tests.

The Ray bring-up/tear-down helpers live in `tests/_ray_cluster.py` so a test can import
them by an unambiguous module name; they are re-exported here because this is where the
integration suite looks for them. Import them from `_ray_cluster` in new tests — a bare
``from conftest import ...`` resolves to whichever `conftest` pytest imported first,
which breaks any run spanning two test directories.
"""

from __future__ import annotations

from _ray_cluster import init_test_ray, shutdown_test_ray

__all__ = ["init_test_ray", "shutdown_test_ray"]
