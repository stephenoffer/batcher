"""Straggler-speculation decision logic (pure, no Ray)."""

from __future__ import annotations

import pytest

from batcher.carbonite.resilience import SpeculationPolicy, stragglers_to_backup

pytestmark = pytest.mark.unit


def test_disabled_when_max_backups_zero():
    pol = SpeculationPolicy(max_backups=0)
    # Even a wild straggler is not backed up when speculation is off.
    assert stragglers_to_backup(4, {0: 1.0, 1: 1.0, 2: 1.0}, {3: 100.0}, pol) == []


def test_no_backup_before_min_finished_fraction():
    pol = SpeculationPolicy(max_backups=2, min_finished_frac=0.75)
    # Only 1 of 4 finished (<75%): too early to judge a straggler.
    assert stragglers_to_backup(4, {0: 1.0}, {1: 50.0, 2: 50.0, 3: 50.0}, pol) == []


def test_backs_up_slowest_running_beyond_threshold():
    pol = SpeculationPolicy(max_backups=1, min_finished_frac=0.5, straggler_factor=2.0)
    # 2 of 4 finished (median 1.0); threshold = 2.0. Tasks 2 (3x) and 3 (5x) qualify;
    # max_backups=1 → only the slowest (task 3).
    out = stragglers_to_backup(4, {0: 1.0, 1: 1.0}, {2: 3.0, 3: 5.0}, pol)
    assert out == [3]


def test_backs_up_multiple_up_to_cap():
    pol = SpeculationPolicy(max_backups=2, min_finished_frac=0.5, straggler_factor=2.0)
    out = stragglers_to_backup(4, {0: 1.0, 1: 1.0}, {2: 3.0, 3: 5.0}, pol)
    assert out == [3, 2]  # slowest first, both over threshold, within cap


def test_no_backup_when_running_within_threshold():
    pol = SpeculationPolicy(max_backups=2, min_finished_frac=0.5, straggler_factor=3.0)
    # median 1.0, threshold 3.0; the one running task at 2.5x is not a straggler.
    assert stragglers_to_backup(4, {0: 1.0, 1: 1.0, 2: 1.0}, {3: 2.5}, pol) == []
