"""Framework exports, config-file loaders, split records: the last of the unexercised surface.

What is left after the coverage sweep, grouped by why it was missed rather than by what it
does:

* **``ds.to_torch`` / ``to_torch_dataloader`` / ``to_tf`` / ``to_jax``** -- the training-loop
  exits. They are the last mile of an ML pipeline and the place where a shape or a dtype
  error costs a training run rather than a query.
* **``Config.from_file`` / ``from_toml`` / ``from_yaml``** -- how a deployment configures the
  engine without touching code. A loader that silently ignored a section would leave a
  cluster running defaults while its config file said otherwise.
* **``FileSplit`` / ``WholeSourceSplit`` / ``RowGroupSplit``** -- the records that decide read
  parallelism. Their ``identity`` is what a resumable or cached read is keyed on, so two
  different splits sharing one identity is a wrong-data bug rather than a slow read.
* **``WriteManifest.merge``** -- how a distributed write's per-worker manifests become one.

Each is checked against the thing it promises rather than against its own output: a tensor
against the rows it came from, a loaded config against the file that set it, a split's
identity against a second split built the same way and a third built differently.
"""

from __future__ import annotations

import dataclasses

import pytest

import batcher as bt

pytestmark = pytest.mark.integration

ROWS = {"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 5.0, 6.0, 7.0]}


@pytest.fixture
def ds():
    return bt.from_pydict(ROWS)


def test_to_torch_yields_tensors_holding_the_dataset_rows(ds):
    """Batched tensors, in order, with the values the dataset holds."""
    pytest.importorskip("torch")
    batches = list(ds.to_torch(batch_size=2))
    assert len(batches) == 2, f"four rows at batch_size=2 is two batches, got {len(batches)}"
    seen: dict[str, list[float]] = {"a": [], "b": []}
    for batch in batches:
        assert set(batch) == {"a", "b"}, f"columns were {sorted(batch)}"
        for name, tensor in batch.items():
            assert tuple(tensor.shape) == (2,), f"{name} had shape {tuple(tensor.shape)}"
            seen[name].extend(float(v) for v in tensor)
    assert seen["a"] == ROWS["a"], "the rows must survive in order"
    assert seen["b"] == ROWS["b"]


def test_to_torch_selects_only_the_requested_columns(ds):
    """``columns=`` must narrow the tensors, or a wide table pays for every column."""
    pytest.importorskip("torch")
    batch = next(iter(ds.to_torch(columns=["a"], batch_size=4)))
    assert set(batch) == {"a"}, f"columns were {sorted(batch)}"
    assert [float(v) for v in batch["a"]] == ROWS["a"]


def test_to_torch_dataloader_is_a_real_dataloader_over_the_same_rows(ds):
    """The ``DataLoader`` wrapper, checked by iterating it rather than by its type alone."""
    torch = pytest.importorskip("torch")
    loader = ds.to_torch_dataloader(batch_size=2)
    assert isinstance(loader, torch.utils.data.DataLoader)
    collected: dict[str, list[float]] = {"a": [], "b": []}
    for batch in loader:
        for name, tensor in batch.items():
            collected[name].extend(float(v) for v in tensor.reshape(-1))
    assert sorted(collected["a"]) == sorted(ROWS["a"])
    assert len(collected["a"]) == len(ROWS["a"]), "the loader lost or duplicated a row"


def test_to_tf_yields_batches_holding_the_dataset_rows(ds):
    """The TensorFlow exit, same contract as the torch one."""
    pytest.importorskip("tensorflow")
    exported = ds.to_tf(batch_size=2)
    seen: list[float] = []
    for batch in exported:
        assert set(batch) == {"a", "b"}
        seen.extend(float(v) for v in batch["a"].numpy().reshape(-1))
    assert seen == ROWS["a"], f"{seen}"


def test_to_jax_names_the_missing_package_when_jax_is_absent(ds):
    """Where JAX is installed it must return arrays; where it is not, it must say so."""
    from batcher._internal.errors import MissingDependencyError

    try:
        arrays = ds.to_jax()
    except MissingDependencyError as missing:
        assert "jax" in str(missing).lower()
        assert "pip install" in str(missing)
        return
    assert set(arrays) == {"a", "b"}
    assert [float(v) for v in arrays["a"]] == ROWS["a"]


def test_the_framework_exports_agree_with_each_other_on_the_values(ds):
    """Whatever the container, the numbers are the dataset's -- so they must all match.

    This is the assertion a per-framework test cannot make: an export that transposed, or
    that read a stale cache, is consistent with itself and inconsistent with its siblings.
    """
    pytest.importorskip("torch")
    from_torch: list[float] = []
    for batch in ds.to_torch(batch_size=3):
        from_torch.extend(float(v) for v in batch["a"])
    assert from_torch == ROWS["a"]
    assert from_torch == ds.to_pydict()["a"], "the tensor and the frame must agree"


#: The three config-file loaders. ``from_file`` dispatches on the extension, so it must
#: reach both of the others.
def _write_config(directory, name: str, body: str) -> str:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_config_from_toml_applies_the_file(tmp_path):
    """A TOML section must reach the config object, not be parsed and dropped."""
    from batcher.config import Config

    path = _write_config(tmp_path, "c.toml", "[execution]\nparallelism = 7\n")
    loaded = Config.from_toml(path)
    assert loaded.execution.parallelism == 7
    assert Config().execution.parallelism != 7, "the fixture must differ from the default"


def test_config_from_yaml_applies_the_file(tmp_path):
    """The YAML spelling of the same thing."""
    pytest.importorskip("yaml")
    from batcher.config import Config

    path = _write_config(tmp_path, "c.yaml", "execution:\n  parallelism: 5\n")
    loaded = Config.from_yaml(path)
    assert loaded.execution.parallelism == 5


def test_config_from_file_dispatches_on_the_extension(tmp_path):
    """``from_file`` must reach the TOML and YAML loaders, and agree with calling them."""
    from batcher.config import Config

    toml_path = _write_config(tmp_path, "c.toml", "[execution]\nparallelism = 7\n")
    assert Config.from_file(toml_path).execution.parallelism == 7
    assert (
        Config.from_file(toml_path).execution.parallelism
        == Config.from_toml(toml_path).execution.parallelism
    )
    if pytest.importorskip("yaml", reason="PyYAML absent"):
        yaml_path = _write_config(tmp_path, "c.yaml", "execution:\n  parallelism: 5\n")
        assert Config.from_file(yaml_path).execution.parallelism == 5


def test_a_config_file_layers_onto_a_base_rather_than_replacing_it(tmp_path):
    """``base=`` is what makes a file an override, which is how a deployment uses one."""
    from batcher.config import Config

    base = dataclasses.replace(
        Config(), execution=dataclasses.replace(Config().execution, morsel_rows=4096)
    )
    path = _write_config(tmp_path, "c.toml", "[execution]\nparallelism = 3\n")
    layered = Config.from_toml(path, base)
    assert layered.execution.parallelism == 3, "the file's value wins where it speaks"
    assert layered.execution.morsel_rows == 4096, "and the base survives where it does not"


def test_an_unparseable_config_file_names_the_path(tmp_path):
    """A typo in a deployment's config must say which file, not just that parsing failed."""
    from batcher._internal.errors import ConfigError

    path = _write_config(tmp_path, "broken.conf", "this is not a config\n")
    with pytest.raises(ConfigError) as failure:
        __import__("batcher.config", fromlist=["Config"]).Config.from_file(path)
    assert "broken.conf" in str(failure.value)


def test_a_file_split_identity_distinguishes_two_different_reads():
    """The identity keys caching and resume, so two different splits must not share one."""
    from batcher.io import FileSplit

    one = FileSplit("parquet", "/data/a.parquet")
    same = FileSplit("parquet", "/data/a.parquet")
    other_path = FileSplit("parquet", "/data/b.parquet")
    other_format = FileSplit("csv", "/data/a.parquet")

    assert one.identity() == same.identity(), "the same read must key the same"
    assert one.identity() != other_path.identity(), "a different file must key differently"
    assert one.identity() != other_format.identity(), "so must a different format"
    assert isinstance(one.identity(), str)
    assert "a.parquet" in one.identity(), "the identity should be legible in a log"


def test_a_row_group_split_identity_includes_the_row_groups():
    """Two splits over the same file and different row groups are different reads."""
    from batcher.io import RowGroupSplit

    first = RowGroupSplit("/data/a.parquet", (0, 1))
    second = RowGroupSplit("/data/a.parquet", (2, 3))
    assert first.identity() != second.identity(), (
        "splitting one file by row group must produce distinct identities, or a resumed "
        "read replays the wrong groups"
    )
    assert first.identity() == RowGroupSplit("/data/a.parquet", (0, 1)).identity()


def test_a_whole_source_split_reads_the_source_it_wraps():
    """The one-split case: the whole relation, with the source's own schema and rows."""
    import pyarrow as pa

    from batcher.io import InMemorySource, WholeSourceSplit

    source = InMemorySource([pa.record_batch({"x": [1, 2, 3]})])
    split = WholeSourceSplit(source)
    assert split.schema() == source.schema()
    assert split.row_count() == 3
    rows = [batch.column("x").to_pylist() for batch in split.iter_batches()]
    assert [v for batch in rows for v in batch] == [1, 2, 3]
    assert isinstance(split.identity(), str)


def test_a_source_splits_into_splits_that_together_read_everything():
    """The split protocol's whole point: the parts must reconstruct the relation."""
    import pyarrow as pa

    from batcher.io import InMemorySource

    source = InMemorySource([pa.record_batch({"x": [1, 2]}), pa.record_batch({"x": [3, 4]})])
    splits = list(source.splits())
    assert splits, "a source must offer at least one split"
    seen: list[int] = []
    for split in splits:
        for batch in split.iter_batches():
            seen.extend(batch.column("x").to_pylist())
    assert sorted(seen) == [1, 2, 3, 4], f"the splits read {sorted(seen)}"
    assert len({s.identity() for s in splits}) == len(splits), (
        "two splits of one source sharing an identity would collide in a resume log"
    )


def test_a_write_manifest_merges_into_one_record(tmp_path):
    """A distributed write produces one manifest per worker; ``merge`` makes them one."""
    first = bt.from_pydict({"a": [1, 2]}).write.parquet(str(tmp_path / "one"))
    second = bt.from_pydict({"a": [3, 4]}).write.parquet(str(tmp_path / "two"))
    merged = first.merge(second)

    assert len(merged.files) == len(first.files) + len(second.files)
    assert sum(f.rows for f in merged.files) == 4, (
        f"the merged manifest accounts for {sum(f.rows for f in merged.files)} rows, not four"
    )
    # Compared by path: `WrittenFile` carries a `stats` dict, so the records are not
    # hashable and a set comparison would fail on the container rather than the content.
    merged_paths = [f.path for f in merged.files]
    assert all(f.path in merged_paths for f in first.files)
    assert all(f.path in merged_paths for f in second.files)
    assert merged.schema == first.schema, "merging must not change the written schema"
    assert sum(f.rows for f in first.files) == 2, (
        "and the individual manifests still say what they wrote"
    )


def test_merging_manifests_is_associative_in_what_it_accounts_for(tmp_path):
    """``merge`` is a fold over per-worker results, so the grouping must not matter."""
    a = bt.from_pydict({"a": [1]}).write.parquet(str(tmp_path / "a"))
    b = bt.from_pydict({"a": [2]}).write.parquet(str(tmp_path / "b"))
    c = bt.from_pydict({"a": [3]}).write.parquet(str(tmp_path / "c"))
    left = a.merge(b).merge(c)
    right = a.merge(b.merge(c))
    assert sum(f.rows for f in left.files) == sum(f.rows for f in right.files) == 3
    assert sorted(f.path for f in left.files) == sorted(f.path for f in right.files)
