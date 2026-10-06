"""INSERT / DELETE / UPDATE / MERGE as plan rewrites over a session catalog."""

from batcher._sql.dml.apply import DmlResult, apply_dml
from batcher._sql.dml.rewrite import align_insert

__all__ = ["DmlResult", "align_insert", "apply_dml"]
