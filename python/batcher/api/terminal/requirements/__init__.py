"""The worker-requirements preflight: what a plan's remote workers must be able to import.

`report` builds and renders it for ``ds.explain(requirements=True)``; `scan` reads one UDF's
needs from the payload a worker would receive.
"""

from __future__ import annotations

from batcher.api.terminal.requirements.report import annotate_requirements, requirements_report

__all__ = ["annotate_requirements", "requirements_report"]
