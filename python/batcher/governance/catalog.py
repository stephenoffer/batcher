"""`SecurityCatalog` — the declared policies, and how they resolve for a principal.

The catalog is the *decision* half of governance: given a table, a principal, and the
table's columns, it answers "which columns are visible", "which are masked and how",
and "which rows may be seen". It never rewrites a plan (that is `enforce`) and never
executes anything.

Tables are named by the path a source is read from — ``"/data/customers.parquet"`` —
because that path is the only identity a file-backed table has that is stable across
runs and knowable *before* the table is read. Policies must be declarable before the
first read; they cannot be keyed on a handle the user does not have yet.

Table names are **canonicalized** (`io.filesystem.canonical_path`) on the way in and on
the way out, so ``s3a://vault/pii.parquet``, ``S3://vault/pii.parquet`` and
``s3://vault//pii.parquet`` are one table rather than three. They name one object, and a
policy that fired on only one spelling of it was a policy with two documented bypasses.

Resolution rules, in one place because they are the security contract:

* **Access.** A table with no `Grant` at all is open. **One grant governs the whole
  table**: every privilege on it becomes deny-by-default, and the principal holds the
  union of what its roles are granted, privilege by privilege. Judging each privilege
  separately would mean granting `INSERT` to a load role left it free to *overwrite*,
  `DELETE` being unmentioned and so still open.
* **Denial.** A `Denial` beats every `Grant`, always. It is the only way to say "all
  columns except this one" without enumerating a complement that a new column breaks.
* **Masking.** A column's mask is its explicit `ColumnMask` if it has one, else the
  `TagMask` of any tag it carries. A principal holding an exempt role reads the raw
  value.
* **Rows.** Every non-exempt `RowFilter` on the table applies, conjoined.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from batcher._internal.errors import PlanError
from batcher.governance._validate import (
    check_callable,
    column_set,
    policy_name,
    reject_bare_string,
)
from batcher.governance.policy import (
    ColumnMask,
    Denial,
    Grant,
    MaskFn,
    PredicateFn,
    RowFilter,
    TagMask,
    normalize_privilege,
)
from batcher.governance.principal import Principal
from batcher.io.filesystem import canonical_path

__all__ = ["SecurityCatalog"]

#: Returned by `SecurityCatalog._denied_columns` when a denial covers the whole table
#: rather than a column list. A distinct object rather than `None` or an empty set,
#: because both of those already mean "nothing is denied" and confusing the two would
#: fail *open* — the one direction an authorization check must never fail in.
_ALL = object()


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

    __slots__ = ("_denials", "_grants", "_masks", "_row_filters", "_tag_masks", "_tags")

    def __init__(self) -> None:
        """Create an empty catalog. An empty catalog governs nothing."""
        self._grants: list[Grant] = []
        self._denials: list[Denial] = []
        self._masks: dict[tuple[str, str], ColumnMask] = {}
        self._tags: dict[tuple[str, str], set[str]] = {}
        self._tag_masks: dict[str, TagMask] = {}
        self._row_filters: list[RowFilter] = []

    # --- declaration -------------------------------------------------------
    def grant(
        self,
        role: str,
        *,
        on: str,
        select: Sequence[str] | None = None,
        privilege: str = "SELECT",
    ) -> SecurityCatalog:
        """Grant `role` the `privilege` on `select` (or every column) of table `on`.

        The first grant on a table makes **every** privilege on it deny-by-default, for
        every role. So a table with a `SELECT` grant and no `INSERT` grant can be read by
        the granted roles and written by nobody, which is what "this table has an access
        policy" should mean. A table with no grant at all is untouched.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.SecurityCatalog()
                >>> cat.grant("analyst", on="/data/customers.parquet", select=["id"]) is cat
                True
                >>> analyst = bt.Principal("ana", roles=["analyst"])
                >>> cat.visible_columns("/data/customers.parquet", ["id", "ssn"], analyst)
                ['id']

                A load job may append but not read:

                >>> _ = cat.grant("loader", on="/data/customers.parquet", privilege="INSERT")
                >>> loader = bt.Principal("etl", roles=["loader"])
                >>> cat.holds("/data/customers.parquet", loader, "INSERT")
                True
                >>> cat.holds("/data/customers.parquet", loader, "SELECT")
                False

        Args:
            role: The role receiving the privilege.
            on: The table name (the path it is read from), in any spelling.
            select: Column names, or None for every column. `SELECT` only.
            privilege: One of `batcher.governance.PRIVILEGES`.

        Returns:
            This catalog, for chaining.

        Raises:
            PlanError: If `role` or `on` is not a non-empty string, `select` is a bare
                string rather than a sequence of column names, `privilege` is not a known
                privilege, or `select` was given for a privilege that acts on whole rows.
        """
        self._grants.append(
            Grant(
                role=policy_name(role, "role name"),
                table=self._table(on),
                columns=column_set(select),
                privilege=privilege,
            )
        )
        return self

    def revoke(self, role: str, *, on: str, privilege: str = "SELECT") -> SecurityCatalog:
        """Withdraw every `privilege` grant `role` holds on table `on`.

        The inverse of `grant`, and it removes grants rather than adding a countervailing
        rule: after a revoke the catalog reads as though the grant had never been
        written, which is what makes a catalog reviewable. Revoking a privilege nobody
        granted is not an error — the end state is the same either way, and an
        offboarding script that must first check what it is undoing is a script that
        races itself.

        Revoking is table-wide. To take away *some* columns, either re-`grant` the
        narrower list or write a `deny`, which is the only way to express "everything
        except this column" without enumerating a complement that the next added column
        silently widens.

        **Revoke does not deny.** Removing the only grant on a table removes the
        deny-by-default that grant established, so the table goes back to being open for
        that privilege. That is the SQL semantics, and it is the thing to get right in an
        offboarding path: if the intent is "this role must never read this", write
        `deny`, which survives a later grant.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.SecurityCatalog().grant("intern", on="/data/t.parquet")
                >>> intern = bt.Principal("sam", roles=["intern"])
                >>> cat.holds("/data/t.parquet", intern, "SELECT")
                True
                >>> cat.revoke("intern", on="/data/t.parquet") is cat
                True
                >>> [g.role for g in cat.grants_on("/data/t.parquet")]
                []

        Args:
            role: The role losing the privilege.
            on: The table name, in any spelling.
            privilege: One of `batcher.governance.PRIVILEGES`.

        Returns:
            This catalog, for chaining.

        Raises:
            PlanError: If `role` or `on` is not a non-empty string, or `privilege` is not
                a known privilege.
        """
        role, table = policy_name(role, "role name"), self._table(on)
        want = normalize_privilege(privilege, None)
        self._grants = [
            g
            for g in self._grants
            if not (g.role == role and g.table == table and g.privilege == want)
        ]
        return self

    def deny(
        self,
        role: str,
        *,
        on: str,
        select: Sequence[str] | None = None,
        privilege: str = "SELECT",
    ) -> SecurityCatalog:
        """Refuse `role` the `privilege` on table `on`, whatever any grant says.

        A denial wins over every grant, the same precedence `DENY` has in SQL Server and
        in Unity Catalog. Two things need it and neither can be written with grants
        alone: **"everything except `salary`"**, because the complement of a column list
        is wrong the moment a column is added to the table; and **a hard block on a role
        that other roles' grants would otherwise union around**, because a principal
        holding both ``analyst`` and ``auditor`` sees whatever either role sees.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = (
                ...     bt.SecurityCatalog()
                ...     .grant("analyst", on="/data/hr.parquet")
                ...     .deny("analyst", on="/data/hr.parquet", select=["salary"])
                ... )
                >>> analyst = bt.Principal("ana", roles=["analyst"])
                >>> cat.visible_columns("/data/hr.parquet", ["id", "salary"], analyst)
                ['id']

                Block a whole privilege, grant or no grant:

                >>> _ = cat.deny("analyst", on="/data/hr.parquet", privilege="DELETE")
                >>> cat.holds("/data/hr.parquet", analyst, "DELETE")
                False

        Args:
            role: The role being refused.
            on: The table name, in any spelling.
            select: Column names to refuse, or None for the whole table. `SELECT` only.
            privilege: One of `batcher.governance.PRIVILEGES`.

        Returns:
            This catalog, for chaining.

        Raises:
            PlanError: If `role` or `on` is not a non-empty string, `select` is a bare
                string, `privilege` is not a known privilege, or `select` was given for a
                privilege that acts on whole rows.
        """
        self._denials.append(
            Denial(
                role=policy_name(role, "role name"),
                table=self._table(on),
                columns=column_set(select),
                privilege=privilege,
            )
        )
        return self

    @staticmethod
    def _table(name: object) -> str:
        """Validate a declared table name and reduce it to its canonical spelling.

        Both halves matter and they fail differently. Skipping the validation stores a
        policy under a name nothing matches; skipping the canonicalization stores it
        under one of several spellings of the same object, so `s3a://` reads past a rule
        written about `s3://`.
        """
        return canonical_path(policy_name(name, "table name"))

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

        Raises:
            PlanError: If `table` or `column` is not a non-empty string, or `mask` is
                not callable.
        """
        table, column = self._table(table), policy_name(column, "column name")
        check_callable(mask, "mask_column(mask=...)", "mask(column_expression) -> expression")
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

        Raises:
            PlanError: If `table` or `column` is not a non-empty string, or no tag was
                given — ``tag(table, column)`` with no tags reads as a classification
                but stores nothing, so a later `mask_tag` governs nothing.
        """
        table, column = self._table(table), policy_name(column, "column name")
        if not tags:
            raise PlanError(
                f"tag({table!r}, {column!r}) was given no tags.",
                hint="Pass at least one, e.g. catalog.tag(table, column, 'pii').",
            )
        self._tags.setdefault((table, column), set()).update(
            policy_name(t, "tag name") for t in tags
        )
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

        Raises:
            PlanError: If `tag` is not a non-empty string, or `mask` is not callable.
        """
        tag = policy_name(tag, "tag name")
        check_callable(mask, "mask_tag(mask=...)", "mask(column_expression) -> expression")
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

        Raises:
            PlanError: If `table` or `name` is not a non-empty string, or `predicate`
                is not callable.
        """
        table = self._table(table)
        check_callable(
            predicate, "filter_rows(predicate=...)", "predicate(principal) -> expression"
        )
        self._row_filters.append(
            RowFilter(table, predicate, policy_name(name, "row-filter name"), frozenset(exempt))
        )
        return self

    def __repr__(self) -> str:
        """Count what is declared, per policy kind.

        "Did my policy actually get installed?" is the question a catalog is printed to
        answer, and the default `object.__repr__` — an address — answers none of it.
        """
        return (
            f"SecurityCatalog(grants={len(self._grants)}, denials={len(self._denials)}, "
            f"masks={len(self._masks)}, tags={len(self._tags)}, "
            f"tag_masks={len(self._tag_masks)}, row_filters={len(self._row_filters)})"
        )

    # --- resolution --------------------------------------------------------
    def _any_policy(self) -> bool:
        """Whether this catalog declares anything at all.

        `enforce` needs it to decide whether an unnameable source is benign (nothing is
        governed, so nothing was skipped) or a policy that may silently not have applied.
        Private because "is my catalog empty" is not a question a user's code should
        branch on — `governs(table)` is the public, per-table answer.
        """
        return bool(
            self._grants
            or self._denials
            or self._masks
            or self._tags
            or self._tag_masks
            or self._row_filters
        )

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
        table = canonical_path(table)
        return (
            any(g.table == table for g in self._grants)
            or any(d.table == table for d in self._denials)
            or any(t == table for t, _ in self._masks)
            or any(t == table for t, _ in self._tags)
            or any(f.table == table for f in self._row_filters)
        )

    def _governed_tables(self) -> frozenset[str]:
        """Every table name any policy in this catalog mentions, canonical already.

        `governs` answers the same question for one table by rescanning every policy list.
        A caller asking about thousands of paths — the pinned-file check in
        `api.security._binding` — would then be quadratic in policies x paths at *plan*
        time, which is the `O(files)` control-plane work the architecture rule forbids.
        Building the set once makes each of those a set membership.

        Private because "which tables are governed" is a question about the catalog's
        contents rather than about a principal's access, and answering it publicly would
        hand any caller the list of tables somebody thought worth protecting.
        """
        return frozenset(
            [g.table for g in self._grants]
            + [d.table for d in self._denials]
            + [t for t, _ in self._masks]
            + [t for t, _ in self._tags]
            + [f.table for f in self._row_filters]
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

        Raises:
            PlanError: If `columns` is a bare string, which would be read as one column
                per character, or `principal` is not a `Principal`.
        """
        reject_bare_string(
            columns,
            what="visible_columns(columns=...)",
            param="columns",
            reads_as="one column per character",
        )
        if not isinstance(principal, Principal):
            raise PlanError(
                f"visible_columns needs a Principal, but got "
                f"{type(principal).__name__} {principal!r}.",
                hint='Build one with bt.Principal("name", roles=[...]).',
            )
        table = canonical_path(table)
        denied = self._denied_columns(table, principal, "SELECT")
        if denied is _ALL:
            return []
        if not any(g.table == table for g in self._grants):
            return [c for c in columns if c not in denied]
        # The table carries a grant, so reads are deny-by-default on it. Only SELECT
        # grants widen what is visible: an INSERT grant is not a statement about reading.
        grants = [g for g in self._grants if g.table == table and g.privilege == "SELECT"]
        allowed: set[str] = set()
        for g in grants:
            if not principal.has_role(g.role):
                continue
            if g.columns is None:
                # Granted every column: the only thing that can still remove one is a
                # denial, which outranks the grant.
                return [c for c in columns if c not in denied]
            allowed |= g.columns
        return [c for c in columns if c in allowed and c not in denied]

    def mask_for(self, table: str, column: str, principal: Principal) -> MaskFn | None:
        """The mask to read `table`.`column` through, or None to read it raw.

        An explicit `mask_column` wins over any `mask_tag` *when it applies*; among tags,
        the first tag in sorted order whose mask *applies to this principal* wins, so
        resolution is deterministic regardless of the order tags were declared in. A
        column may carry several policies (an explicit mask and/or several sensitivity
        tags): being exempt from one policy's mask does not grant raw access while another
        policy still masks it — the strictest applicable policy governs. In particular, an
        exemption from the explicit mask falls through to any tag mask the principal is not
        also exempt from, so a narrow explicit exemption cannot silently disable a broad
        tag-based safety net.

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
        table = canonical_path(table)
        explicit = self._masks.get((table, column))
        if explicit is not None and not principal.has_any_role(explicit.exempt_roles):
            return explicit.mask
        # Either no explicit mask, or the principal is exempt from it. An exemption from
        # the explicit mask does not grant raw access while a tag mask still applies —
        # the same most-restrictive-wins contract that governs multiple tags. Fall
        # through so a narrow explicit exemption cannot bypass a broad tag safety net.
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
        table = canonical_path(table)
        return [
            f
            for f in self._row_filters
            if f.table == table and not principal.has_any_role(f.exempt_roles)
        ]

    def _denied_columns(self, table: str, principal: Principal, privilege: str) -> object:
        """The columns `principal` is denied on `table`, or `_ALL` for the whole table.

        `table` is already canonical here — every caller comes through a public method
        that has folded it — so this does not fold it again.
        """
        denied: set[str] = set()
        for d in self._denials:
            if d.table != table or d.privilege != privilege or not principal.has_role(d.role):
                continue
            if d.columns is None:
                return _ALL
            denied |= d.columns
        return denied

    def holds(self, table: str, principal: Principal, privilege: str = "SELECT") -> bool:
        """Whether `principal` may exercise `privilege` on `table` at all.

        The table-level question, and the one the write path asks: *may this identity
        insert into this table*, not *which columns may it see*. `visible_columns` is the
        column-level answer, and only `SELECT` has one.

        A table nobody has granted anything on is open. Once *any* grant names it the
        table is governed, and every privilege on it must be granted explicitly — so a
        role given `INSERT` cannot also `DELETE`, which is the point of granting `INSERT`
        rather than "write". A denial refuses regardless.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.SecurityCatalog().grant(
                ...     "loader", on="/data/t.parquet", privilege="INSERT"
                ... )
                >>> loader = bt.Principal("etl", roles=["loader"])
                >>> cat.holds("/data/t.parquet", loader, "INSERT")
                True
                >>> cat.holds("/data/t.parquet", loader, "DELETE")  # granting INSERT is not
                False
                >>> cat.holds("/data/untouched.parquet", loader, "DELETE")  # no policy at all
                True

        Args:
            table: The table name, in any spelling.
            principal: The identity running the query.
            privilege: One of `batcher.governance.PRIVILEGES`.

        Returns:
            True if no grant names this table at all, or the privilege is granted to one
            of the principal's roles; and in either case not denied to any of them.

        Raises:
            PlanError: If `privilege` is not a known privilege, or `principal` is not a
                `Principal`.
        """
        if not isinstance(principal, Principal):
            raise PlanError(
                f"holds needs a Principal, but got {type(principal).__name__} {principal!r}.",
                hint='Build one with bt.Principal("name", roles=[...]).',
            )
        table = canonical_path(table)
        want = normalize_privilege(privilege, None)
        if self._denied_columns(table, principal, want) is _ALL:
            return False
        if not any(g.table == table for g in self._grants):
            # Nobody has written an access policy about this table, so it is open — the
            # rule that keeps installing a catalog from locking every unrelated path.
            return True
        # Some grant exists, so the table is governed and every privilege on it is
        # deny-by-default. Judging each privilege separately instead would mean granting
        # INSERT to a load role left it able to *overwrite* — DELETE being unmentioned
        # and therefore open — which is the opposite of what writing that grant means.
        return any(
            g.table == table and g.privilege == want and principal.has_role(g.role)
            for g in self._grants
        )

    def grants_on(self, table: str, privilege: str | None = None) -> list[Grant]:
        """Every grant declared on `table`, optionally narrowed to one `privilege`.

        The read side of `grant` and `revoke`: what a catalog review, an access report, or
        an offboarding script needs in order to say what a table currently allows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = (
                ...     bt.SecurityCatalog()
                ...     .grant("analyst", on="/data/t.parquet")
                ...     .grant("loader", on="s3a://bucket/t.parquet", privilege="INSERT")
                ... )
                >>> [g.role for g in cat.grants_on("/data/t.parquet")]
                ['analyst']

                The spelling used to look it up need not be the one it was declared with:

                >>> [g.privilege for g in cat.grants_on("s3://bucket/t.parquet")]
                ['INSERT']

        Args:
            table: The table name, in any spelling.
            privilege: One of `batcher.governance.PRIVILEGES`, or None for every one.

        Returns:
            The matching grants, in declaration order.

        Raises:
            PlanError: If `table` is not a non-empty string, or `privilege` is given and
                is not a known privilege.
        """
        table = self._table(table)
        want = None if privilege is None else normalize_privilege(privilege, None)
        return [
            g for g in self._grants if g.table == table and (want is None or g.privilege == want)
        ]

    def denials_on(self, table: str, privilege: str | None = None) -> list[Denial]:
        """Every denial declared on `table`, optionally narrowed to one `privilege`.

        The companion to `grants_on`, and the half an access report must not omit: a
        denial is invisible in the grant list and outranks all of it, so a report built
        from grants alone overstates what a principal can reach.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.SecurityCatalog().deny(
                ...     "contractor", on="/data/hr.parquet", select=["salary"]
                ... )
                >>> [sorted(d.columns) for d in cat.denials_on("/data/hr.parquet")]
                [['salary']]

        Args:
            table: The table name, in any spelling.
            privilege: One of `batcher.governance.PRIVILEGES`, or None for every one.

        Returns:
            The matching denials, in declaration order.

        Raises:
            PlanError: If `table` is not a non-empty string, or `privilege` is given and
                is not a known privilege.
        """
        table = self._table(table)
        want = None if privilege is None else normalize_privilege(privilege, None)
        return [
            d for d in self._denials if d.table == table and (want is None or d.privilege == want)
        ]
