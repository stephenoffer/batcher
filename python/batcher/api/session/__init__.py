"""Session entry points that create `Dataset`s.

A re-export façade over one module per responsibility: `frames` (Python and Arrow
objects), `frameworks` (pandas/Polars/DuckDB/Ray/... plus the type-dispatching
`from_any`), `read` (path and table sources, and the ``read_*`` shorthands),
`generate` (`range`/`date_range`), `combine` (`concat`), `sql` (the default
catalog), `admin` (maintenance and streaming control), `versions`, and `accelerators`
(what GPU hardware this process and its cluster can see).

Everything funnels through `_scan`, the single place a `Source` becomes a
`Dataset` and therefore the single place the governance rewrite can be enforced.

The public names resolve lazily (PEP 562, via `batcher._lazy`), so reaching one of
these modules does not import the other eight. `_scan` and `_catalog` are private and
stay eager: they are internal call targets rather than surface, and the modules that
import them do so by name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._exports import SESSION_EXPORTS as _SESSION_EXPORTS
from batcher._exports import SESSION_SHADOWED as _SESSION_SHADOWED
from batcher._lazy import install as _install

if TYPE_CHECKING:
    # The declaration of this façade's surface, and the generator's input. Eager only
    # for type checkers and editors — at runtime `__getattr__` binds these on first touch.
    from batcher.api.session.accelerators import *  # noqa: F403
    from batcher.api.session.admin import *  # noqa: F403
    from batcher.api.session.cache import *  # noqa: F403
    from batcher.api.session.combine import *  # noqa: F403
    from batcher.api.session.frames import *  # noqa: F403
    from batcher.api.session.frameworks import *  # noqa: F403
    from batcher.api.session.generate import *  # noqa: F403
    from batcher.api.session.read import *  # noqa: F403
    from batcher.api.session.sql import *  # noqa: F403
    from batcher.api.session.versions import *  # noqa: F403

#: The `session` surface, in the order the eager façade declared it.
__all__ = list(_SESSION_EXPORTS)

__getattr__, __dir__ = _install(__name__, _SESSION_EXPORTS, shadowed=_SESSION_SHADOWED)
