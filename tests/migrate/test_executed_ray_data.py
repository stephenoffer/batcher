"""A translated Ray Data program returns the rows the Ray Data program returned.

Each executed corpus case runs as the original script on a fresh local Ray instance
(`RAY_ADDRESS=local`, so no shared cluster is touched) and as the translated script in Batcher,
over the same in-memory data. `map_batches` is the case this direction exists for: Ray hands a
function numpy batches by default and Batcher hands it Arrow, so a translation that dropped
`batch_format="numpy"` would fail inside the function rather than agree.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ray")
pytest.importorskip("libcst")

from _corpus import CORPUS, EXECUTED, equivalent, run


@pytest.fixture(scope="module", autouse=True)
def _local_ray():
    import ray

    ray.init(address="local", num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.mark.parametrize("case", sorted(EXECUTED["ray_data"]))
def test_translated_program_returns_the_ray_data_rows(case: str) -> None:
    source, translated = equivalent("ray_data", case)
    assert source, "a case whose program returns no rows compares nothing"
    assert translated == source


def test_the_comparison_sees_a_rewrite_that_changes_the_batch_format(tmp_path) -> None:
    # Positive control: without the numpy batch format Ray defaults to, the function is handed
    # an Arrow batch it cannot assign into, so the naive translation cannot agree.
    golden = (CORPUS / "ray_data" / "map_batches" / "batcher.py").read_text()
    naive = golden.replace('add_total, batch_format="numpy"', "add_total")
    assert naive != golden
    (tmp_path / "naive.py").write_text(naive)
    with pytest.raises(Exception):  # noqa: B017 - any failure inside the user function counts
        run(tmp_path / "naive.py")
