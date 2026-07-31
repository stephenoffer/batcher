"""Session entry points that create `Dataset`s.

A re-export façade over one module per responsibility: `frames` (Python and Arrow
objects), `frameworks` (pandas/Polars/DuckDB/Ray/... plus the type-dispatching
`from_any`), `read` (path and table sources, and the ``read_*`` shorthands),
`generate` (`range`/`date_range`), `combine` (`concat`), `sql` (the default
catalog), `admin` (maintenance and streaming control), `versions`, and `accelerators`
(what GPU hardware this process and its cluster can see).

Everything funnels through `_scan`, the single place a `Source` becomes a
`Dataset` and therefore the single place the governance rewrite can be enforced.
"""

from __future__ import annotations

from batcher.api.session import accelerators as _accelerators
from batcher.api.session import admin as _admin
from batcher.api.session import combine as _combine
from batcher.api.session import frames as _frames
from batcher.api.session import frameworks as _frameworks
from batcher.api.session import generate as _generate
from batcher.api.session import read as _read_mod
from batcher.api.session import sql as _sql
from batcher.api.session import versions as _versions
from batcher.api.session._scan import _scan as _scan
from batcher.api.session.accelerators import *  # noqa: F403
from batcher.api.session.admin import *  # noqa: F403  (governed by admin.__all__)
from batcher.api.session.combine import *  # noqa: F403
from batcher.api.session.frames import *  # noqa: F403
from batcher.api.session.frameworks import *  # noqa: F403
from batcher.api.session.generate import *  # noqa: F403
from batcher.api.session.read import *  # noqa: F403
from batcher.api.session.sql import *  # noqa: F403
from batcher.api.session.sql import _catalog as _catalog
from batcher.api.session.versions import *  # noqa: F403

__all__ = [
    *_accelerators.__all__,
    *_admin.__all__,
    *_combine.__all__,
    *_frames.__all__,
    *_frameworks.__all__,
    *_generate.__all__,
    *_read_mod.__all__,
    *_sql.__all__,
    *_versions.__all__,
]
