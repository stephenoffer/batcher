"""The Join Order Benchmark — all 113 queries over the real IMDb database.

JOB exists to answer a question TPC-H and TPC-DS cannot: how good is the optimizer when the
data is *correlated*. Its 113 queries join 3 to 16 real tables under predicates whose
selectivities are not independent, which is where Leis et al. showed cardinality estimates go
wrong by orders of magnitude and take the join order with them. That makes it the benchmark
most directly aimed at Batcher's stated moat — re-optimizing on measured cardinalities at
pipeline breakers — and the one where a win or a loss says the most about the optimizer
rather than about the operators.

The statements are vendored verbatim from the reference implementation into
:data:`QUERY_FILE` by ``tools/vendor_job_queries.py``; the tables come from the archive that
implementation distributes (``sources.job``). This module only splits the file and fans each
query across the SQL-capable engines.

Every query is `SELECT MIN(...) ... FROM a, b, ... WHERE <equi-joins and filters>` — a single
row out of a large join. That shape is deliberate on the benchmark's part: the result is
trivial to compare, so what is being measured is the plan, not the output.
"""

from __future__ import annotations

import os

from registry import suite
from suites.standard._vendored import register_vendored

job = suite("job", dataset="job")

QUERY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_queries.sql")

# JOB is 113 queries by definition; anything else means the vendored file is wrong.
QUERY_COUNT = 113

QUERIES = register_vendored(
    job, QUERY_FILE, count=QUERY_COUNT, vendor_tool="tools/vendor_job_queries.py"
)
