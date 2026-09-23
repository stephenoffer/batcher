"""Suggest a contract from the data, validate it, and ship the report as strict JSON.

`ds.dq.suggest()` reads a table and proposes the constraints it already satisfies, so the
first contract is not written from memory. The report it produces serializes to strict JSON,
even when a measurement is NaN, so a metrics sink that rejects `NaN` tokens accepts it. The
last part shows why a NaN measurement fails a relation-level bound instead of slipping past.

    python examples/quality/suggest_and_report.py
"""

from __future__ import annotations

import json

import batcher as bt

NAN = float("nan")


def readings() -> bt.Dataset:
    """Six sensor readings: a clean temperature column and a pressure column with a NaN."""
    return bt.from_pydict(
        {
            "sensor_id": [1, 2, 3, 4, 5, 6],
            "site": ["north", "south", "north", "south", "north", "north"],
            "temperature": [21.5, 22.0, 20.75, 23.25, 21.0, 22.5],
            "pressure": [1012.0, NAN, 1009.5, 1011.0, 1010.0, 1013.5],
        }
    )


def main() -> None:
    ds = readings()

    # 1. Suggest. Everything proposed holds on this data, so the report is clean.
    proposed = ds.dq.suggest()
    report = proposed.validate()
    names = list(report.violations)
    print("suggested:", names)
    assert report.ok, report.violations
    assert "unique(sensor_id)" in names
    assert "accepted_values(site)" in names
    # A float column is asked to stay finite only when it already is.
    assert "is_finite(temperature)" in names
    assert "is_finite(pressure)" not in names

    # 2. Every result reports the relation's row count, including a clean one.
    assert report.rows == ds.count() == 6
    assert all(r.rows == 6 for r in report.results)

    # 3. A NaN makes the mean NaN, and a NaN mean fails the bound.
    gate = ds.dq.mean_between("pressure", 900, 1100).mean_between("temperature", 15, 30)
    checked = gate.validate()
    pressure = checked.result("mean_between(pressure, 900, 1100)")
    temperature = checked.result("mean_between(temperature, 15, 30)")
    print("pressure mean:", pressure.value, "ok:", pressure.ok)
    assert pressure.value != pressure.value  # NaN is the only value not equal to itself
    assert (pressure.ok, pressure.pass_rate) == (False, 0.0)
    assert (temperature.ok, temperature.pass_rate) == (True, 1.0)
    assert not checked.ok

    # 4. The report is strict JSON: the NaN measurement is emitted as null.
    payload = json.dumps(checked.to_dict(), allow_nan=False)
    decoded = json.loads(payload)
    by_name = {c["name"]: c for c in decoded["constraints"]}
    assert by_name["mean_between(pressure, 900, 1100)"]["value"] is None
    assert abs(by_name["mean_between(temperature, 15, 30)"]["value"] - 131 / 6) < 1e-9
    assert decoded["ok"] is False and decoded["rows"] == 6
    print(payload)


if __name__ == "__main__":
    main()
