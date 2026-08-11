"""Prove a constraint from metadata — a data contract that holds should cost nothing.

Every pipeline worth trusting runs a contract at its boundary: `amount` is in range, `id` is
never null, the key is unique. And in the overwhelmingly common case — the case the contract
exists to *confirm* — the answer is "yes, it holds", which is exactly the answer a Parquet
footer already contains. Reading three numbers should not cost a scan of a billion rows.

The proof needs no per-constraint reasoning, because a constraint is already a boolean `Expr`
that is TRUE for a valid row. So "nothing violates it" is precisely "the filter `NOT valid`
keeps no row" — a question the zone-map layer has always been able to answer, and now gets
asked. `in_range(x, 0, 10000)` over a column whose footer says `[1, 1000]` folds to an empty
filter and the contract is discharged without opening a data page.

Two honest limits, both of which fall back to executing rather than guessing:

* a **`check()`** constraint carries a user-supplied predicate that may evaluate to NULL, and
  a NULL validity counts as a *violation* while `NOT NULL` is NULL (so `filter(NOT valid)`
  would not see it). The built-in constraints are all total by construction, and say so; a
  custom one is not assumed to be.
* a **uniqueness** constraint needs an exact distinct count, which a columnar footer does not
  record. An immutable in-memory relation computes one, so it is proved there.

Layer: `api`, the conductor — it asks Kyber whether the answer is provable and executes when
it is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from batcher.api.dataset.meta._facts import MetaBase
from batcher.kyber.shortcuts import distinct

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.plan.expr_ir import Expr

__all__ = ["constraints_provably_hold", "keys_provably_unique", "violations_provably_absent"]


def constraints_provably_hold(ds: Dataset, constraints: Sequence[object]) -> bool:
    """Whether metadata proves **every** constraint holds — the contract discharged for free.

    The common case a data contract exists to *confirm*: the data is fine. And "fine" is usually
    already written in the footer — `in_range(amount, 0, 10000)` over a column whose recorded
    range is `[1, 1000]` cannot be violated, and proving that is three numbers, not a billion
    rows.

    All-or-nothing, and conservative at every step: one constraint that cannot be proved (a
    regex, a custom `check`, a composite uniqueness) returns False, and the caller validates the
    whole set by executing, exactly as it did before. So this only ever removes work — it can
    never wave a violation through, which is the one failure this must not have.
    """
    from batcher.api.dataset.dq.constraints import RowConstraint, UniqueConstraint

    if not constraints:
        return True  # no contract to violate
    for constraint in constraints:
        # Dispatch on the type rather than on the presence of an attribute. Duck-typing on
        # `keys` read a *reference* constraint's `reference_keys` sibling as a key tuple once
        # the constraint vocabulary grew past two kinds, and the failure mode of guessing
        # wrong here is discharging a contract that does not hold.
        if isinstance(constraint, UniqueConstraint):
            if not keys_provably_unique(ds, constraint.keys):
                return False
        elif isinstance(constraint, RowConstraint):
            if not constraint.total or not violations_provably_absent(ds, constraint.valid):
                return False
        else:
            return False  # relation-level, schema, or referential — not provable from a footer
    return True


def violations_provably_absent(ds: Dataset, valid: Expr) -> bool:
    """Whether metadata proves that **no** row of `ds` violates `valid`.

    `valid` must be a *total* predicate — never NULL — or this is not the right question to
    ask: a NULL validity is a violation, and `NOT NULL` is NULL, so the probe below would not
    see it. Every built-in constraint is total by construction; a `check()` is not assumed to
    be, and its caller does not reach here.

    Returns False whenever the proof is unavailable, which simply means the caller executes —
    the answer is the same either way.
    """
    from batcher.api.terminal.metadata_answer import metadata_is_empty

    probe = ds.filter(~valid)
    return metadata_is_empty(probe._plan, probe._sources) is True


def keys_provably_unique(ds: Dataset, keys: Sequence[str]) -> bool:
    """Whether metadata proves the `keys` combination occurs at most once per row.

    Proved for a single key that is a *primary key* — an exact distinct count reaching the
    exact non-null row count, with no nulls (two null keys would share a partition and violate
    uniqueness, so a nullable unique column is not enough). A composite key is never proved: no
    format records a multi-column distinct count, and inventing one is how a `DISTINCT` gets
    dropped from a query that needed it.
    """
    if len(keys) != 1:
        return False
    meta = MetaBase(ds)
    return meta.ask(distinct.is_key, keys[0], ndv=(keys[0],)) is True
