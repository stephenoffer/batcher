"""Live progress for a long-running distributed or batch-inference job.

`progress` folds the distributed observability events into a bounded per-job snapshot;
`measures` holds the stateless computations that snapshot and its diagnostics are built
from. The public import path `batcher.observe.inference` is unchanged by the split.
"""

from __future__ import annotations

from batcher.observe.inference.progress import DEFAULT_MAX_JOBS, InferenceProgress

__all__ = ["DEFAULT_MAX_JOBS", "InferenceProgress"]
