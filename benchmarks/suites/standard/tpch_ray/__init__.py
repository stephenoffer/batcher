"""Native Ray Data pipelines for all 22 TPC-H queries.

Ray Data has no SQL surface, so without this package the standard suite leaves it
``n/a`` on every TPC-H query and the "distributed comparator" column is empty. These
are hand-written ``ray.data.Dataset`` pipelines -- the same 22 workloads, expressed
the way a Ray Data user would write them.

Coverage used to stop at four queries (q1/q6/q12/q14) on the grounds that chained
shuffle-joins were "impractically slow" and several queries were "not expressible in
its API at all". Both premises had expired:

* The slowness was a harness bug, not Ray Data. ``engines/ray.py`` handed every table
  to ``ray.data.from_arrow``, which makes exactly **one block** -- and a block is Ray
  Data's unit of parallelism, so a 6M-row ``lineitem`` join ran single-threaded on a
  96-core box. Registering tables through the Parquet connector fixes the blocking.
* Ray 2.56's ``Dataset.join`` supports ``LEFT_SEMI``/``LEFT_ANTI``, which is what the
  ``EXISTS`` / ``NOT IN`` queries (q4, q16, q21, q22) need. There is no longer a TPC-H
  query the API cannot express.

The query bodies live in ``queries_a`` (q1-q8), ``queries_b`` (q9-q16), and
``queries_c`` (q17-q22); the registry they self-register into is ``base``, and
``runner`` builds the suite case. The harness still gates every timing on the result
matching the DuckDB reference, so a pipeline that drifted from the SQL is reported
``FAILED`` rather than quietly timed.
"""

from __future__ import annotations

from .runner import case_with_ray, ray_impl

__all__ = ["case_with_ray", "ray_impl"]
