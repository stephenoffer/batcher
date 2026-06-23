"""`io.formats.nosql` — connectors for NoSQL / operational data stores.

Each module here registers a row-based store (MongoDB, Cassandra/Scylla,
DynamoDB, Redis, Elasticsearch, Couchbase, Neo4j, HBase) into the `SOURCES`
registry. They share `base.ScanSource`: a `Source` that opens a per-worker
connection from never-logged connection kwargs, enumerates the store's natural
parallel unit as picklable, connection-free splits, and assembles Arrow at batch
granularity (Arrow-native where the driver supports it). Importing this package
imports every connector so the registry is populated as a side effect. Optional
drivers are deferred — a missing driver raises `BackendError` with the matching
``pip install 'batcher-engine[<extra>]'`` hint.
"""

from __future__ import annotations

from batcher.io.formats.nosql.base import PartitionSpec, ScanSource
from batcher.io.formats.nosql.cassandra import CassandraSource, ScyllaSource
from batcher.io.formats.nosql.couchbase import CouchbaseSource
from batcher.io.formats.nosql.dynamodb import DynamoDBSource
from batcher.io.formats.nosql.elasticsearch import ElasticsearchSource
from batcher.io.formats.nosql.hbase import HBaseSource
from batcher.io.formats.nosql.mongo import MongoSink, MongoSource
from batcher.io.formats.nosql.neo4j import Neo4jSource
from batcher.io.formats.nosql.redis import RedisSource

__all__ = [
    "CassandraSource",
    "CouchbaseSource",
    "DynamoDBSource",
    "ElasticsearchSource",
    "HBaseSource",
    "MongoSink",
    "MongoSource",
    "Neo4jSource",
    "PartitionSpec",
    "RedisSource",
    "ScanSource",
    "ScyllaSource",
]
