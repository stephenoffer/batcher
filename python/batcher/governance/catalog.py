"""`SecurityCatalog` — the declared policies, and how they resolve for a principal.

The catalog is the *decision* half of governance: given a table, a principal, and the
table's columns, it answers "which columns are visible", "which are masked and how",
and "which rows may be seen". It never rewrites a plan (that is `enforce`) and never
executes anything.

Tables are named by the path a source is read from — ``"/data/customers.parquet"`` —
because that path is the only identity a file-backed table has that is stable across
runs and knowable *before* the table is read. Policies must be declarable before the
first read; they cannot be keyed on a handle the user does not have yet.

Resolution rules, in one place because they are the security contract:

* **Access.** A table with no `Grant` is open. A table with any `Grant` is
  deny-by-default: the principal sees the union of the columns granted to its roles.
* **Masking.** A column's mask is its explicit `ColumnMask` if it has one, else the
  `TagMask` of any tag it carries. A principal holding an exempt role reads the raw
  value.
* **Rows.** Every non-exempt `RowFilter` on the table applies, conjoined.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from batcher.governance.policy import ColumnMask, Grant, MaskFn, PredicateFn, RowFilter, TagMask
from batcher.governance.principal import Principal

__all__ = ["SecurityCatalog"]


class SecurityCatalog:
    """A mutable collection of grants, masks, tags, and row filters.

    Built once (at session start, or loaded from your own policy store) and then read
    concurrently by the enforcement rewrite. Declaration methods return `self` so a
    catalog reads as a policy document.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> catalog = (
            ...     bt.SecurityCatalog()
            ...     .grant("analyst", on="/data/customers.parquet", select=["id", "email"])
            ...     .tag("/data/customers.parquet", "email", "pii")
            ...     .mask_tag("pii", lambda c: bt.mask(c, show_last=0))
            ... )
            >>> analyst = bt.Principal("ana", roles=["analyst"])
            >>> catalog.visible_columns(
            ...     "/data/customers.parquet", ["id", "email", "ssn"], analyst
            ... )
            ['id', 'email']
    """

    __slots__ = ("_grants", "_masks", "_row_filters", "_tag_masks", "_tags")

    def __init__(self) -> None:
        """Create an empty catalog. An empty catalog governs nothing."""
        self._grants: list[Grant] = []
        self._masks: dict[tuple[str, str], ColumnMask] = {}
        self._tags: dict[tuple[str, str], set[str]] = {}
        self._tag_masks: dict[str, TagMask] = {}
        self._row_filters: list[RowFilter] = []

    # --- declaration -------------------------------------------------------
    def grant(self, role: str, *, on: str, select: Sequence[str] | None = None) -> SecurityCatalog:
        """Grant `role` the right to `SELECT` `select` (or every column) of table `on`.

        The first grant on a table makes it deny-by-default for every role.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.SecurityCatalog()
                >>> cat.grant("analyst", on="/data/customers.parquet", select=["id"]) is cat
                True
                >>> analyst = bt.Principal("ana", roles=["analyst"])
                >>> cat.visible_columns("/data/customers.parquet", ["id", "ssn"], analyst)
                ['id']

        Args:
            role: The role receiving the privilege.
            on: The table name (the path it is read from).
            select: Column names, or None for every column.

        Returns:
            This catalog, for chaining.
        """
        self._grants.append(
            Grant(role=role, table=on, columns=None if select is None else frozenset(select))
        )
        return self

    def mask_column(
        self, table: str, column: str, mask: MaskFn, *, exempt: Iterable[str] = ()
    ) -> SecurityCatalog:
        """Read `table`.`column` through `mask`, except for principals holding `exempt`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.SecurityCatalog()
                >>> cat.mask_column("/data/t.parquet", "email", lambda c: bt.mask(c)) is cat
                True
                >>> analyst = bt.Principal("ana", roles=["analyst"])
                >>> cat.mask_for("/data/t.parquet", "email", analyst) is None
                False

        Args:
            table: The table name.
            column: The column to mask.
            mask: Given the column's expression, the expression to read instead.
            exempt: Roles that read the raw value.

        Returns:
            This catalog, for chaining.
        """
        self._masks[table, column] = ColumnMask(table, column, mask, frozenset(exempt))
        return self

    def tag(self, table: str, column: str, *tags: str) -> SecurityCatalog:
        """Classify `table`.`column` with one or more `tags` (e.g. ``"pii"``).

        Tags carry no policy on their own; `mask_tag` attaches one.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.SecurityCatalog()
                >>> cat.tag("/data/t.parquet", "email", "pii") is cat
                True
                >>> cat.mask_tag("pii", lambda c: bt.mask(c)) is cat
                True
                >>> analyst = bt.Principal("ana", roles=["analyst"])
                >>> cat.mask_for("/data/t.parquet", "email", analyst) is None
                False

        Args:
            table: The table name.
            column: The column being classified.
            *tags: Tag names.

        Returns:
            This catalog, for chaining.
        """
        self._tags.setdefault((table, column), set()).update(tags)
        return self

    def mask_tag(self, tag: str, mask: MaskFn, *, exempt: Iterable[str] = ()) -> SecurityCatalog:
        """Mask every column tagged `tag`, in every table, except for `exempt` roles.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.SecurityCatalog().tag("/data/t.parquet", "email", "pii")
                >>> cat.mask_tag("pii", lambda c: bt.mask(c), exempt=["admin"]) is cat
                True
                >>> admin = bt.Principal("root", roles=["admin"])
                >>> cat.mask_for("/data/t.parquet", "email", admin) is None
                True

        Args:
            tag: The tag to govern.
            mask: Given a tagged column's expression, the expression to read instead.
            exempt: Roles that read the raw value.

        Returns:
            This catalog, for chaining.
        """
        self._tag_masks[tag] = TagMask(tag, mask, frozenset(exempt))
        return self

    def filter_rows(
        self,
        table: str,
        predicate: PredicateFn,
        *,
        name: str = "row_filter",
        exempt: Iterable[str] = (),
    ) -> SecurityCatalog:
        """Restrict `table` to rows satisfying ``predicate(principal)``, except for `exempt`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.SecurityCatalog()
                >>> cat.filter_rows(
                ...     "/data/t.parquet",
                ...     lambda p: bt.col("region") == p.attrs["region"],
                ...     name="region_scope",
                ... ) is cat
                True
                >>> analyst = bt.Principal("ana", roles=["analyst"], attrs={"region": "EU"})
                >>> [f.name for f in cat.row_filters_for("/data/t.parquet", analyst)]
                ['region_scope']

        Args:
            table: The table name.
            predicate: Given the principal, the predicate rows must satisfy.
            name: A label used in explain output and audit events.
            exempt: Roles that see every row.

        Returns:
            This catalog, for chaining.
        """
        self._row_filters.append(RowFilter(table, predicate, name, frozenset(exempt)))
        return self

    # --- resolution --------------------------------------------------------
    def governs(self, table: str) -> bool:
        """Whether any policy in this catalog mentions `table`.

        A table nobody has written a policy about is left exactly as it was — the check
        that keeps installing a catalog from perturbing unrelated queries.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.SecurityCatalog().grant("analyst", on="/data/sales.parquet")
                >>> cat.governs("/data/sales.parquet")
                True
                >>> cat.governs("/data/other.parquet")
                False

        Args:
            table: The table name.

        Returns:
            True if a grant, mask, tag, or row filter names `table`.
        """
        return (
            any(g.table == table for g in self._grants)
            or any(t == table for t, _ in self._masks)
            or any(t == table for t, _ in self._tags)
            or any(f.table == table for f in self._row_filters)
        )

    def visible_columns(
        self, table: str, columns: Sequence[str], principal: Principal
    ) -> list[str]:
        """The subset of `columns` that `principal` may `SELECT` from `table`, in order.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.SecurityCatalog().grant(
                ...     "analyst", on="/data/t.parquet", select=["id", "email"]
                ... )
                >>> analyst = bt.Principal("ana", roles=["analyst"])
                >>> cat.visible_columns("/data/t.parquet", ["id", "email", "ssn"], analyst)
                ['id', 'email']

        Args:
            table: The table name.
            columns: The table's columns, in schema order.
            principal: The identity running the query.

        Returns:
            The visible columns, preserving `columns`' order. Every column when the
            table carries no grant; otherwise the union of the principal's roles' grants.
        """
        grants = [g for g in self._grants if g.table == table]
        if not grants:
            return list(columns)
        allowed: set[str] = set()
        for g in grants:
            if not principal.has_role(g.role):
                continue
            if g.columns is None:
                return list(columns)
            allowed |= g.columns
        return [c for c in columns if c in allowed]

    def mask_for(self, table: str, column: str, principal: Principal) -> MaskFn | None:
        """The mask to read `table`.`column` through, or None to read it raw.

        An explicit `mask_column` wins over any `mask_tag`; among tags, the first tag in
        sorted order whose mask *applies to this principal* wins, so resolution is
        deterministic regardless of the order tags were declared in. A column may carry
        several sensitivity tags: being exempt from one tag's mask does not grant raw
        access while another tag still masks it — the strictest applicable tag governs.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.SecurityCatalog().mask_column(
                ...     "/data/t.parquet", "ssn", lambda c: bt.mask(c)
                ... )
                >>> analyst = bt.Principal("ana", roles=["analyst"])
                >>> cat.mask_for("/data/t.parquet", "ssn", analyst) is None
                False
                >>> cat.mask_for("/data/t.parquet", "id", analyst) is None
                True

        Args:
            table: The table name.
            column: The column being read.
            principal: The identity running the query.

        Returns:
            The mask function, or None if unmasked or the principal is exempt.
        """
        explicit = self._masks.get((table, column))
        if explicit is not None:
            return None if principal.has_any_role(explicit.exempt_roles) else explicit.mask
        for tag in sorted(self._tags.get((table, column), ())):
            tag_mask = self._tag_masks.get(tag)
            if tag_mask is None:
                continue
            # An exemption from *this* tag's mask does not free the column: a column
            # carrying several tags must still be masked by any tag the principal is not
            # exempt from. Skip exempted tags and keep looking; only fall through to raw
            # when no applicable tag masks this principal.
            if principal.has_any_role(tag_mask.exempt_roles):
                continue
            return tag_mask.mask
        return None

    def row_filters_for(self, table: str, principal: Principal) -> list[RowFilter]:
        """Every row filter on `table` that `principal` is not exempt from.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.SecurityCatalog().filter_rows(
                ...     "/data/t.parquet",
                ...     lambda p: bt.col("region") == "EU",
                ...     name="eu_only",
                ...     exempt=["admin"],
                ... )
                >>> analyst = bt.Principal("ana", roles=["analyst"])
                >>> [f.name for f in cat.row_filters_for("/data/t.parquet", analyst)]
                ['eu_only']
                >>> cat.row_filters_for("/data/t.parquet", bt.Principal("root", roles=["admin"]))
                []

        Args:
            table: The table name.
            principal: The identity running the query.

        Returns:
            The applicable filters, in declaration order. They are conjoined by
            `enforce`, so the order affects only the shape of the predicate, not the rows.
        """
        return [
            f
            for f in self._row_filters
            if f.table == table and not principal.has_any_role(f.exempt_roles)
        ]
