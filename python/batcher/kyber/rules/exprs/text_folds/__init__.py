"""String constant folding, grouped by whether the function takes an argument.

`plain` folds a call carrying only its operand -- the length variants, the digests,
`hex`/`base64`, the pads, `repeat`, `reverse`, `initcap`, and the trims. `args` folds
the ones carrying a pattern, index, or replacement as well. `literals` holds the lifter
and the two operand gates both share.

Importing this package runs each module's ``@rule`` decorators. Re-export and
registration only -- no logic here.
"""

from __future__ import annotations

from batcher.kyber.rules.exprs.text_folds import args as _args  # noqa: F401
from batcher.kyber.rules.exprs.text_folds import plain as _plain  # noqa: F401

__all__: list[str] = []
