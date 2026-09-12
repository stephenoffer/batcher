"""Ray Data pipelines for the h2o ``groupby`` task (façade)."""

from __future__ import annotations

from suites.h2o.groupby_ray.base import IMPLS, RayQuery
from suites.h2o.groupby_ray.runner import case_with_ray, ray_impl

__all__ = ["IMPLS", "RayQuery", "case_with_ray", "ray_impl"]
