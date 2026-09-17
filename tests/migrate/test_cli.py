"""`python -m batcher.migrate` runs every direction, with `--check` and `--report` in each."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("libcst")

from _corpus import CORPUS

from batcher._internal.errors import ConfigError
from batcher.migrate.__main__ import main


def test_a_foreign_direction_rewrites_in_place_and_reports(tmp_path, capsys) -> None:
    script = tmp_path / "job.py"
    script.write_text((CORPUS / "polars" / "mismatches" / "src.py").read_text())
    report = tmp_path / "report.json"
    code = main(
        [str(script), "--from", "polars", "--to", "batcher", "--write", "--report", str(report)]
    )
    assert code == 0
    assert script.read_text() == (CORPUS / "polars" / "mismatches" / "batcher.py").read_text()
    entry = json.loads(report.read_text())[str(script)]
    assert entry["counts"]["marked"] >= 3
    assert {s["action"] for s in entry["sites"]} >= {"rewritten", "marked"}
    assert "left `Expr.n_unique` as written" in capsys.readouterr().err


def test_check_fails_when_a_file_would_change_in_either_direction(tmp_path) -> None:
    source = tmp_path / "job.py"
    source.write_text((CORPUS / "daft" / "verbs" / "src.py").read_text())
    before = source.read_text()
    assert main([str(tmp_path), "--from", "daft", "--to", "batcher", "--check"]) == 1
    assert source.read_text() == before, "--check without --write must not touch the file"
    exported = tmp_path / "out.py"
    exported.write_text((CORPUS / "daft" / "verbs" / "batcher.py").read_text())
    source.unlink()
    assert main([str(tmp_path), "--from", "batcher", "--to", "daft", "--check"]) == 1


def test_check_passes_on_a_file_with_nothing_to_translate(tmp_path) -> None:
    (tmp_path / "plain.py").write_text("x = 1\n")
    assert main([str(tmp_path), "--from", "ray_data", "--to", "batcher", "--check"]) == 0


def test_two_foreign_engines_are_not_a_direction(tmp_path) -> None:
    with pytest.raises(ConfigError, match="one side must be batcher"):
        main([str(tmp_path), "--from", "pyspark", "--to", "polars"])
