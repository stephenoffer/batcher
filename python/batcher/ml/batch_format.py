"""Re-export of the `batch_format` conversion, which now lives in `interop`.

The implementation moved to `batcher.interop.formats` so the *executor* can reach it:
`core.udf.{apply,call,processes}` convert around every user function, and `core` is a
subsystem that must not import this front-end package (`.claude/rules/architecture.md`).
This module keeps `batcher.ml.batch_format` working as an import path.
"""

from __future__ import annotations

from batcher.interop.formats import FORMATS, result_to_arrowable, to_format

__all__ = ["FORMATS", "result_to_arrowable", "to_format"]
