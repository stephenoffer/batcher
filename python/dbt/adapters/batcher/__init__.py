"""The module dbt imports for ``type: batcher``; the adapter lives in `batcher.integrations.dbt`.

dbt-core finds an adapter by importing ``dbt.adapters.<type>`` and reading its ``Plugin``.
``dbt`` and ``dbt.adapters`` are namespace packages, so this directory has no
``__init__.py`` above it.
"""

from batcher.integrations.dbt.adapter import Plugin

__all__ = ["Plugin"]
