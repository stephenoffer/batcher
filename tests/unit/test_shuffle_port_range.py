"""The Flight shuffle listener's port range — the restricted-network escape hatch.

An ephemeral port never collides, but it obliges an operator to open the whole ephemeral
range node-to-node, which most on-prem and locked-down cloud networks will not do. A
configured range lets them open exactly what the shuffle uses. The failure mode these
tests guard is silence: a malformed range, or a range too narrow for the workers sharing a
node, must raise rather than quietly bind a port nothing outside the host can reach.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import ConfigError
from batcher.dist.flight_worker import _shuffle_port_range

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("raw", "expected"), [(None, None), ("", None), ("  ", None)])
def test_unset_range_means_ephemeral(raw, expected):
    assert _shuffle_port_range(raw) is expected


def test_range_is_parsed():
    assert _shuffle_port_range("40000-40100") == (40000, 40100)
    assert _shuffle_port_range(" 40000-40100 ") == (40000, 40100)


@pytest.mark.parametrize(
    "raw",
    [
        "bogus",  # not a range at all
        "40000",  # missing the upper bound
        "40100-40000",  # descending
        "0-10",  # port 0 is "pick an ephemeral one" — not a real bound
        "1-70000",  # above the 16-bit port space
    ],
)
def test_malformed_range_raises_rather_than_falling_back(raw):
    """A typo must not degrade to an ephemeral port: that port is outside the operator's
    firewall rule, so the shuffle hangs unreachable instead of failing."""
    with pytest.raises(ConfigError):
        _shuffle_port_range(raw)


def test_server_binds_inside_the_configured_range():
    from batcher.carbonite.transfer import FlightShuffleServer

    servers = [FlightShuffleServer(None, None, None, (41100, 41110)) for _ in range(3)]
    ports = [int(s.addr.rsplit(":", 1)[1]) for s in servers]
    assert all(41100 <= p <= 41110 for p in ports), ports
    # Workers sharing a node must not land on the same port.
    assert len(set(ports)) == len(ports), ports


def test_exhausted_range_raises_naming_the_range():
    """A range too narrow for the node's workers is a config error the operator must see —
    binding elsewhere would produce an unreachable worker, not a working one."""
    from batcher.carbonite.transfer import FlightShuffleServer

    held = FlightShuffleServer(None, None, None, (41120, 41120))  # noqa: F841 - holds the port
    with pytest.raises(OSError, match="41120"):
        FlightShuffleServer(None, None, None, (41120, 41120))


def test_ephemeral_is_still_the_default():
    from batcher.carbonite.transfer import FlightShuffleServer

    assert FlightShuffleServer().addr.startswith("127.0.0.1:")
