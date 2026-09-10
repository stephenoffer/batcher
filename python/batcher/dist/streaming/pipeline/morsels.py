"""The pipeline's published-morsel ledger: what is in flight, and what it came from.

Split out of `schedule` because it is a data structure rather than a scheduling step, and
because the scheduler's own module has a size budget its dispatch/recovery halves already
fill. Nothing here talks to Ray or to Flight — it is bookkeeping the loop consults, which is
what makes the settling cascade testable on its own.
"""

from __future__ import annotations

__all__ = ["Morsels"]


class Morsels:
    """Every published-but-unsettled morsel, and the parent each one came from.

    Small enough to be a dict of dicts and important enough not to be: settling a morsel has
    to cascade to its parent, and the cascade is where a scheduling bug turns into a memory
    leak (a morsel nobody releases) or a wrong answer (a morsel released while its subtree is
    still being recomputed).
    """

    __slots__ = ("_by_path", "_ids")

    def __init__(self) -> None:
        self._by_path: dict[tuple, dict] = {}
        # A stable small integer per path, because a Flight ticket's fields are integers. The
        # mapping persists for the whole run, so a replayed ancestor is re-issued the *same*
        # id and therefore republishes under the same tickets — the idempotence the module
        # docstring depends on.
        self._ids: dict[tuple, int] = {}

    def id_of(self, path: tuple) -> int:
        """A stable integer naming `path`, minted once and reused on every replay."""
        if path not in self._ids:
            self._ids[path] = len(self._ids)
        return self._ids[path]

    def add(self, path: tuple, *, holder, ticket, parent: tuple | None) -> None:
        self._by_path[path] = {
            "holder": holder,
            "ticket": ticket,
            "parent": parent,
            "pending": None,  # children not yet settled; None until this morsel is consumed
        }

    def get(self, path: tuple) -> dict | None:
        return self._by_path.get(path)

    def pop(self, path: tuple) -> dict | None:
        return self._by_path.pop(path, None)

    def paths_held_by(self, actor) -> list[tuple]:
        return [p for p, rec in self._by_path.items() if rec["holder"] is actor]

    def paths_under_partition(self, pidx: int) -> list[tuple]:
        return [p for p in self._by_path if p and p[0] == pidx]

    def __contains__(self, path: tuple) -> bool:
        return path in self._by_path
