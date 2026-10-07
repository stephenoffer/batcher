"""A dbt adapter pilot: table and view materializations, ``dbt run`` and ``dbt test``.

dbt finds an adapter by importing ``dbt.adapters.<type>``, so the plugin dbt loads lives in
the shim package ``dbt/adapters/batcher`` shipped beside this one, which re-exports
`batcher.integrations.dbt.adapter.Plugin`. This package itself imports nothing from dbt, so
importing it never needs the ``dbt`` extra; `batcher.integrations.dbt.adapter` does.

Not yet verified against a live dbt-core installation; see tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations
