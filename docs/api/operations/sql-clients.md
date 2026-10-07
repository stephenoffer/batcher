# SQL client reference

This page is the reference for the adapters that let other tools drive a Batcher {py:class}`Session <batcher.Session>`: the PEP 249 module `batcher.dbapi`, the Flight SQL service, and the Ibis bridge. To learn them, start with {doc}`/integrations/sql-clients/index`. The SQLAlchemy dialect and the dbt adapter are reached through their tools, by URL and by profile, and are described there.

## The DB-API module

```{eval-rst}
.. currentmodule:: batcher.dbapi
```

`connect` opens a connection over a session, and the connection hands out cursors.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   connect
   Connection
   Cursor
```

The module globals PEP 249 requires:

```{eval-rst}
.. autodata:: apilevel
   :no-value:

.. autodata:: threadsafety
   :no-value:

.. autodata:: paramstyle
   :no-value:
```

### Exceptions

The PEP 249 hierarchy. Each wraps the Batcher error that caused it as `__cause__`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Error
   Warning
   InterfaceError
   DatabaseError
   DataError
   OperationalError
   IntegrityError
   InternalError
   ProgrammingError
   NotSupportedError
```

### Type objects and constructors

A cursor's `description` reports each column type as a pyarrow `DataType`, which compares equal to the matching type object. The constructors return the Python values a parameter binds.

```{eval-rst}
.. autodata:: STRING
   :no-value:

.. autodata:: BINARY
   :no-value:

.. autodata:: NUMBER
   :no-value:

.. autodata:: DATETIME
   :no-value:

.. autodata:: ROWID
   :no-value:

.. autosummary::
   :toctree: generated
   :nosignatures:

   Date
   Time
   Timestamp
   DateFromTicks
   TimeFromTicks
   TimestampFromTicks
   Binary
```

## The Flight SQL service

```{eval-rst}
.. currentmodule:: batcher.integrations.flightsql

.. autosummary::
   :toctree: generated
   :nosignatures:

   serve
```

## The Ibis bridge

```{eval-rst}
.. currentmodule:: batcher.integrations.ibis

.. autosummary::
   :toctree: generated
   :nosignatures:

   table
   to_dataset
```
