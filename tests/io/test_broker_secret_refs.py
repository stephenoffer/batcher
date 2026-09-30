"""A broker credential given as a secret reference is resolved where the client is built.

Kafka's ``sasl_password`` and Event Hubs' ``connection_str`` went to their client libraries
verbatim, so ``sasl_password="env:KAFKA_PASSWORD"`` authenticated with the literal string
``env:KAFKA_PASSWORD``. They now resolve through the same vocabulary every other connector
credential uses, at the point the client is constructed -- which is on the worker, so the
reference is what travels.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pytest

from batcher.io.credentials import resolve_client_secrets
from batcher.io.formats.streaming import eventhubs, kafka, kinesis
from batcher.io.formats.streaming.broker.schema import _BROKER_SECRET_HINTS
from batcher.io.formats.streaming.eventhubs import EventHubsSource
from batcher.io.formats.streaming.kafka import KafkaSource

pytestmark = pytest.mark.io


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setenv("BROKER_PW", "hunter2")


def test_kafka_resolves_a_sasl_password_reference(monkeypatch):
    seen: dict[str, Any] = {}

    class _Consumer:
        def __init__(self, config):
            seen.update(config)

        def assign(self, *_a, **_k):
            pass

        def subscribe(self, *_a, **_k):
            pass

    monkeypatch.setattr(kafka, "_import_consumer", lambda: _Consumer)
    src = KafkaSource("t", partitions=[0], sasl_username="u", sasl_password="env:BROKER_PW")
    # Assigning partitions afterwards imports `confluent_kafka`, which CI does not install;
    # the config the consumer was constructed with is what is under test.
    with contextlib.suppress(ModuleNotFoundError):
        src._client()
    assert seen["sasl.password"] == "hunter2"
    assert seen["sasl.username"] == "u"


def test_event_hubs_resolves_a_connection_string_reference(monkeypatch):
    seen: dict[str, Any] = {}

    class _Consumer:
        @classmethod
        def from_connection_string(cls, *, conn_str, consumer_group, eventhub_name):
            seen["conn_str"] = conn_str
            return cls()

    monkeypatch.setattr(eventhubs, "_import_consumer", lambda: _Consumer)
    EventHubsSource("hub", connection_str="env:BROKER_PW")._client()
    assert seen["conn_str"] == "hunter2"


def test_only_credential_keys_are_resolved():
    out = resolve_client_secrets(
        {"ssl.ca.location": "file:/etc/ca.pem", "sasl.password": "env:BROKER_PW"},
        what="x",
        hints=_BROKER_SECRET_HINTS,
    )
    assert out == {"ssl.ca.location": "file:/etc/ca.pem", "sasl.password": "hunter2"}


def test_a_literal_credential_passes_through():
    assert resolve_client_secrets(
        {"sasl.password": "plain"}, what="x", hints=_BROKER_SECRET_HINTS
    ) == {"sasl.password": "plain"}


def test_kinesis_takes_an_endpoint_and_referenced_keys(monkeypatch):
    seen: dict[str, Any] = {}

    class _Boto3:
        @staticmethod
        def client(service, **kwargs):
            seen.update(kwargs, service=service)
            return object()

    monkeypatch.setattr(kinesis, "_import_boto3", lambda: _Boto3)
    kinesis.KinesisSource(
        "s",
        endpoint_url="http://localhost:4566",
        aws_access_key_id="AKIA",
        aws_secret_access_key="env:BROKER_PW",
    )._client()
    assert seen == {
        "service": "kinesis",
        "region_name": "us-east-1",
        "endpoint_url": "http://localhost:4566",
        "aws_access_key_id": "AKIA",
        "aws_secret_access_key": "hunter2",
    }
