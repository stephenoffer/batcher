"""Live smoke test: ``dbt build`` of a two-model project on the Batcher adapter.

Skipped unless ``BATCHER_LIVE_DBT=1`` and dbt-core is installed (the ``dbt`` extra) with this
repository's ``python/`` importable, so ``dbt.adapters.batcher`` resolves.
Run: ``BATCHER_LIVE_DBT=1 pytest tests/integration/live/test_live_dbt.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("BATCHER_LIVE_DBT"), reason="set BATCHER_LIVE_DBT=1"),
]


def test_dbt_build_runs_and_tests_a_table_and_a_view(tmp_path: Path) -> None:
    dbt_main = pytest.importorskip("dbt.cli.main")
    project = tmp_path / "proj"
    (project / "models").mkdir(parents=True)
    (project / "dbt_project.yml").write_text(
        "name: live\nversion: '1.0'\nconfig-version: 2\nprofile: live\n"
    )
    (project / "models" / "orders.sql").write_text(
        "{{ config(materialized='table') }}\nselect 1 as id union all select 2 as id\n"
    )
    (project / "models" / "big.sql").write_text(
        "{{ config(materialized='view') }}\nselect id from {{ ref('orders') }} where id > 1\n"
    )
    (project / "models" / "schema.yml").write_text(
        "version: 2\nmodels:\n  - name: orders\n    columns:\n      - name: id\n"
        "        data_tests: [unique, not_null]\n  - name: big\n    columns:\n"
        "      - name: id\n        data_tests: [not_null]\n"
    )
    (project / "profiles.yml").write_text(
        "live:\n  target: dev\n  outputs:\n    dev:\n      type: batcher\n"
        f"      database: wh\n      schema: analytics\n      path: {tmp_path / 'warehouse'}\n"
        "      threads: 2\n"
    )
    result = dbt_main.dbtRunner().invoke(
        ["build", "--project-dir", str(project), "--profiles-dir", str(project)]
    )
    assert result.success, result.exception
