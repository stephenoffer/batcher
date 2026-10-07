"""Shared provider quotas: one rate and concurrency limit obeyed by every worker.

`lease` is the one entry point; `quota` holds the policy and `actor` its cluster home.
"""

from __future__ import annotations

from batcher.dist.limits.quota import LocalQuota, QuotaConfig, QuotaState
from batcher.dist.limits.shared import close_quota, lease, uses_ray

__all__ = ["LocalQuota", "QuotaConfig", "QuotaState", "close_quota", "lease", "uses_ray"]
