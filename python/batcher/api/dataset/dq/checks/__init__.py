"""Constraint builders, grouped by the family a user names in one breath.

Each function here returns a ready `Constraint` value — a name plus the boolean `Expr`
that is TRUE for a valid row, or the aggregate that must fall within bounds. They hold no
dataset and execute nothing, which is what lets the same builder serve the `ds.dq`
accessor, the metadata proof in `api.dataset.meta.prove`, and a test that only wants to
read the predicate.

The accessor is the public spelling; these are its bodies, kept out of it so that adding
a check is adding a small function to a family module rather than growing one class file.
"""

from __future__ import annotations

from batcher.api.dataset.dq.checks import aggregates, relations, schema, strings, temporal, values

__all__ = ["aggregates", "relations", "schema", "strings", "temporal", "values"]
