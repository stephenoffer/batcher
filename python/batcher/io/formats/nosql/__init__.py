"""`io.formats.nosql` — connectors for NoSQL / operational data stores.

Each module here registers a row-based store (MongoDB, Cassandra/Scylla,
DynamoDB, Redis, Elasticsearch, Couchbase, Neo4j, HBase) into the `SOURCES`
registry, and the five that Batcher can write into the `SINKS` registry beside it.
They share `base.ScanSource`: a `Source` that opens a per-worker
connection from never-logged connection kwargs, enumerates the store's natural
parallel unit as picklable, connection-free splits, and assembles Arrow at batch
granularity (Arrow-native where the driver supports it), and `base.BulkSink`: the
write mirror, whose ``upsert``/``append``/``overwrite``/``delete`` vocabulary matches
the SQL sink's and whose refusals are per store rather than per implementation.
Importing this package
imports every connector so the registry is populated as a side effect. Optional
drivers are deferred — a missing driver raises `BackendError` with the matching
``pip install 'batcher-engine[<extra>]'`` hint.
"""

from __future__ import annotations

from batcher.io.formats.nosql.base import STORE_WRITE_MODES, BulkSink, PartitionSpec, ScanSource
from batcher.io.formats.nosql.cassandra import CassandraSink, CassandraSource, ScyllaSource
from batcher.io.formats.nosql.couchbase import CouchbaseSource
from batcher.io.formats.nosql.dynamodb import DynamoDBSink, DynamoDBSource
from batcher.io.formats.nosql.elasticsearch import ElasticsearchSink, ElasticsearchSource
from batcher.io.formats.nosql.hbase import HBaseSink, HBaseSource
from batcher.io.formats.nosql.mongo import MongoSink, MongoSource
from batcher.io.formats.nosql.neo4j import Neo4jSource
from batcher.io.formats.nosql.redis import RedisSink, RedisSource

__all__ = [
    "STORE_WRITE_MODES",
    "BulkSink",
    "CassandraSink",
    "CassandraSource",
    "CouchbaseSource",
    "DynamoDBSink",
    "DynamoDBSource",
    "ElasticsearchSink",
    "ElasticsearchSource",
    "HBaseSink",
    "HBaseSource",
    "MongoSink",
    "MongoSource",
    "Neo4jSource",
    "PartitionSpec",
    "RedisSink",
    "RedisSource",
    "ScanSource",
    "ScyllaSource",
]
