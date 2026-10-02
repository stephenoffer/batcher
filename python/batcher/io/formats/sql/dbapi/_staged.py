"""Staged SQL writes: every shard into its own staging table, one transaction to publish.

A distributed write to a database table is one transaction per shard, because each shard
runs on its own worker with its own connection. For ``append`` that means a write that fails
on shard 7 of 10 has already published shards 0 to 6, and for ``overwrite`` it is refused
outright: every shard would empty the one table they all target.

Staging moves the commit point to the driver. Each shard writes its rows, in its own
transaction, into a staging table no reader knows about (``<table>__bt_stage_<token>``).
Only after every shard has succeeded does the driver's `commit` run, and it publishes the
lot in **one** transaction on the target: an ``overwrite`` empties the target with
``DELETE FROM`` and then copies every staging table in with ``INSERT ... SELECT``. A failure
anywhere before that commit leaves the target exactly as it was. The staging tables are
dropped afterwards whether the publish succeeded or not, and each one's name is logged when
it cannot be, so a crashed driver leaves named, inspectable tables rather than anonymous
debris.

The rows cross the network twice -- once into the staging table and once, inside the
database, into the target -- which is the price of a cluster-wide commit on a database that
offers no distributed transaction. The second copy is server-side and never reaches Batcher.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from contextlib import suppress
from typing import Any

from batcher._internal.errors import BackendError
from batcher._internal.logging import get_logger, log_kv
from batcher.io.formats.sql.dbapi._statements import qualified_table, quote, truncate

__all__ = ["STAGED_MODES", "drop_stages", "publish_stages", "stage_name"]

#: Modes a staged write supports. Both are whole-table operations whose result is only
#: correct if every shard lands; the keyed modes are idempotent per key and are made whole
#: by re-running them, so they write in place.
STAGED_MODES = frozenset({"append", "overwrite"})

#: Marks a staging table in its name, so one left behind by a crashed driver is findable.
STAGE_MARKER = "__bt_stage_"

_LOGGER = get_logger("io.sql")


def stage_name(table: str) -> str:
    """A fresh staging table beside `table`, in the same schema.

    Unique per call rather than per write, so shards built on different workers -- each
    constructing its own sink -- need no shared token to avoid colliding.

    Args:
        table: The destination, optionally schema-qualified.

    Returns:
        The staging table's name, qualified like `table`.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.dbapi._staged import stage_name
            >>> stage_name("public.orders").startswith("public.orders__bt_stage_")
            True
    """
    return f"{table}{STAGE_MARKER}{uuid.uuid4().hex[:16]}"


def publish_stages(
    conn: Any,
    target: str,
    stages: Sequence[str],
    columns: Sequence[str],
    *,
    overwrite: bool,
    dialect: str | None,
) -> None:
    """Copy every staging table into `target` in one transaction, then commit it.

    Args:
        conn: An open connection this write owns.
        target: The destination table.
        stages: The staging tables the shards wrote.
        columns: The columns to copy, named identically in stage and target.
        overwrite: Empty `target` first, inside the same transaction.
        dialect: The dialect whose identifier quoting applies.

    Raises:
        BackendError: If any statement fails. The transaction is rolled back first, so the
            target is left as it was.
    """
    cols = ", ".join(quote(c, dialect) for c in columns)
    statements = [truncate(target, dialect=dialect)] if overwrite else []
    statements += [
        f"INSERT INTO {qualified_table(target, dialect)} ({cols}) "
        f"SELECT {cols} FROM {qualified_table(stage, dialect)}"
        for stage in stages
    ]
    if not statements:
        return
    sql = statements[0]
    cursor = conn.cursor()
    try:
        for sql in statements:
            cursor.execute(sql)
        conn.commit()
    except Exception as exc:
        with suppress(Exception):
            conn.rollback()
        raise BackendError(
            f"staged sql write to {target!r} failed while publishing {len(stages)} staging "
            f"table(s); the target was rolled back and is unchanged: {exc}\n{sql}"
        ) from exc
    finally:
        cursor.close()


def drop_stages(conn: Any, stages: Sequence[str], *, dialect: str | None) -> None:
    """Drop each staging table, one statement and commit apiece, logging any that survive.

    One commit per drop because a ``DROP`` is DDL, and on MySQL DDL commits implicitly:
    grouping them buys nothing and a failure midway would then say less about which
    tables are left.

    Args:
        conn: An open connection this write owns.
        stages: The staging tables to remove.
        dialect: The dialect whose identifier quoting applies.
    """
    for stage in stages:
        cursor = conn.cursor()
        try:
            cursor.execute(f"DROP TABLE {qualified_table(stage, dialect)}")
            conn.commit()
        except Exception as exc:
            with suppress(Exception):
                conn.rollback()
            log_kv(
                _LOGGER,
                30,  # logging.WARNING
                "sql write: a staging table could not be dropped; drop it by hand",
                table=stage,
                error=str(exc),
            )
        finally:
            cursor.close()
