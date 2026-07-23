"""Path, format-detection, and option-alias ergonomics at the IO boundary.

These cover the vocabulary a pandas/Polars/Spark user arrives with — `pathlib.Path`,
``~``, a list of files, ``events.csv.gz``, `usecols`, `nrows` — and the actionable
errors that fire when a spelling has no Batcher equivalent.
"""

from __future__ import annotations

import gzip
import os
import pathlib

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import FormatError, IOError, SchemaError
from batcher.io.base._options import BASE_SOURCE_OPTIONS, OptionSpec
from batcher.io.base._paths import normalize_path, normalize_source_path
from batcher.io.detect import compression_for_path, detect_format

pytestmark = pytest.mark.unit


@pytest.fixture
def csv_file(tmp_path: pathlib.Path) -> str:
    p = tmp_path / "t.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    return str(p)


# --------------------------------------------------------------------------- paths


def test_normalize_path_accepts_pathlib(tmp_path: pathlib.Path) -> None:
    assert normalize_path(tmp_path / "a.csv") == str(tmp_path / "a.csv")


def test_normalize_path_accepts_os_pathlike(tmp_path: pathlib.Path) -> None:
    class P:
        def __fspath__(self) -> str:
            return "/data/a.csv"

    assert normalize_path(P()) == "/data/a.csv"


def test_normalize_path_expands_tilde() -> None:
    assert normalize_path("~/x.csv") == os.path.join(os.path.expanduser("~"), "x.csv")


def test_normalize_path_leaves_tilde_inside_a_uri_key() -> None:
    """``~`` is a legal object-store key character, so expanding it would rewrite the key."""
    assert normalize_path("s3://bucket/~keep/a.csv") == "s3://bucket/~keep/a.csv"


def test_normalize_path_rejects_bytes_and_junk() -> None:
    with pytest.raises(IOError, match="Decode it first"):
        normalize_path(b"/data/a.csv")
    with pytest.raises(IOError, match=r"str, pathlib\.Path, or os\.PathLike"):
        normalize_path(42)


def test_normalize_source_path_list_becomes_root_plus_files() -> None:
    root, files = normalize_source_path(["/data/a.csv", "/data/b.csv"])
    assert root == "/data"
    assert files == ["/data/a.csv", "/data/b.csv"]


def test_normalize_source_path_single_has_no_file_list() -> None:
    assert normalize_source_path("/data/a.csv") == ("/data/a.csv", None)


def test_normalize_source_path_rejects_empty_list() -> None:
    with pytest.raises(IOError, match="empty"):
        normalize_source_path([])


def test_read_accepts_pathlib_path(csv_file: str) -> None:
    assert bt.read.csv(pathlib.Path(csv_file)).to_pydict() == {"a": [1, 3], "b": [2, 4]}


def test_read_autodetect_accepts_pathlib_path(csv_file: str) -> None:
    assert bt.read(pathlib.Path(csv_file)).to_pydict() == {"a": [1, 3], "b": [2, 4]}


def test_read_accepts_a_list_of_files(tmp_path: pathlib.Path) -> None:
    (tmp_path / "a.csv").write_text("a,b\n1,2\n")
    (tmp_path / "b.csv").write_text("a,b\n3,4\n")
    got = bt.read.csv([str(tmp_path / "a.csv"), str(tmp_path / "b.csv")]).to_pydict()
    assert sorted(got["a"]) == [1, 3]


def test_a_pinned_file_list_gets_its_own_statistics_identity(tmp_path: pathlib.Path) -> None:
    """A subset is a different relation, so it must not inherit the directory's stats."""
    for name in ("a.csv", "b.csv"):
        (tmp_path / name).write_text("a,b\n1,2\n")
    from batcher.io import CSVSource

    whole = CSVSource(str(tmp_path)).identity()
    subset = CSVSource([str(tmp_path / "a.csv"), str(tmp_path / "b.csv")]).identity()
    assert whole != subset


def test_write_accepts_pathlib_path(tmp_path: pathlib.Path) -> None:
    from batcher.io import ParquetSink

    out = ParquetSink().write(pa.table({"x": [1, 2]}), tmp_path / "o.parquet")
    assert out.path == str(tmp_path / "o.parquet")
    assert isinstance(out.path, str)


# ------------------------------------------------------------------ format detection


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("d/t.parquet", "parquet"),
        ("d/t.pq", "parquet"),
        ("d/t.parq", "parquet"),
        ("d/t.csv", "csv"),
        ("d/t.tsv", "csv"),
        ("d/t.tab", "csv"),
        ("d/t.jsonl", "json"),
        ("d/t.ndjson", "json"),
        ("d/t.orc", "orc"),
        ("d/t.arrow", "arrow"),
        ("d/t.mcap", "mcap"),
        ("d/t.mf4", "mdf"),
    ],
)
def test_detect_format_from_extension(name: str, expected: str) -> None:
    assert detect_format(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("d/t.csv.gz", "csv"),
        ("d/t.jsonl.gz", "json"),
        ("d/t.csv.zst", "csv"),
        ("d/t.csv.bz2", "csv"),
        ("d/t.ndjson.xz", "json"),
    ],
)
def test_detect_format_strips_the_compression_suffix(name: str, expected: str) -> None:
    """``events.csv.gz`` is a CSV — the ``.gz`` says how it is packed, not what it means."""
    assert detect_format(name) == expected


@pytest.mark.parametrize(
    ("name", "codec"),
    [("a.csv.gz", "gzip"), ("a.csv.zst", "zstd"), ("a.csv.xz", "lzma"), ("a.csv", None)],
)
def test_compression_for_path(name: str, codec: str | None) -> None:
    assert compression_for_path(name) == codec


def test_unknown_extension_error_lists_what_is_recognized() -> None:
    with pytest.raises(FormatError) as exc:
        detect_format("d/t.zzz")
    assert "parquet" in str(exc.value)
    assert "format=" in str(exc.value)


def test_unknown_extension_error_suggests_a_near_miss() -> None:
    with pytest.raises(FormatError, match=r"(?i)did you mean"):
        detect_format("d/t.parquett")


def test_unknown_explicit_format_suggests_the_real_one() -> None:
    with pytest.raises(FormatError, match=r"(?i)did you mean 'parquet'"):
        detect_format("d/t.data", explicit="parquett")


def test_explicit_format_still_wins_over_the_extension() -> None:
    assert detect_format("d/t.parquet", explicit="csv") == "csv"


def test_reading_a_gzipped_csv_works_end_to_end(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.csv.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("a,b\n1,2\n3,4\n")
    assert bt.read(str(p)).to_pydict() == {"a": [1, 3], "b": [2, 4]}


def test_reading_a_gzipped_ndjson_works_end_to_end(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.jsonl.gz"
    with gzip.open(p, "wt") as fh:
        fh.write('{"a": 1}\n{"a": 2}\n')
    assert bt.read(str(p)).to_pydict() == {"a": [1, 2]}


# ------------------------------------------------------------------- option aliases


def test_option_spec_folds_an_alias_to_its_canonical_name() -> None:
    spec = OptionSpec("csv", canonical=("delimiter",), aliases={"sep": "delimiter"})
    assert spec.resolve({"sep": ";"}) == {"delimiter": ";"}


def test_option_spec_drops_an_ignored_option() -> None:
    spec = OptionSpec("csv", canonical=(), ignored={"low_memory": "no-op here"})
    assert spec.resolve({"low_memory": True}) == {}


def test_option_spec_rejects_two_spellings_of_one_option() -> None:
    spec = OptionSpec("csv", canonical=("delimiter",), aliases={"sep": "delimiter"})
    with pytest.raises(FormatError, match="two spellings"):
        spec.resolve({"sep": ";", "delimiter": "|"})


def test_option_spec_explains_an_unsupported_option() -> None:
    spec = OptionSpec("csv", unsupported={"index_col": "Batcher has no row index."})
    with pytest.raises(FormatError, match="no row index"):
        spec.resolve({"index_col": 0})


def test_option_spec_suggests_a_near_miss() -> None:
    spec = OptionSpec("csv", canonical=("delimiter",), aliases={"sep": "delimiter"})
    with pytest.raises(FormatError, match=r"(?i)did you mean"):
        spec.resolve({"seperator": ";"})


def test_option_spec_passes_base_options_through() -> None:
    """A format's spec validates its extras; the base class still owns its own keywords."""
    spec = OptionSpec("csv", canonical=("delimiter",), base=BASE_SOURCE_OPTIONS)
    assert spec.resolve({"on_error": "skip", "n_rows": 3}) == {"on_error": "skip", "n_rows": 3}


# ------------------------------------------------- columns / n_rows across formats


@pytest.fixture
def three_col(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path


@pytest.mark.parametrize("fmt", ["parquet", "csv", "json", "arrow", "orc"])
def test_columns_narrows_every_format(fmt: str, tmp_path: pathlib.Path) -> None:
    out = str(tmp_path / f"t.{fmt}")
    ds = bt.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    getattr(ds.write, fmt)(out)
    got = getattr(bt.read, fmt)(out, columns=["a", "c"])
    assert got.schema.names == ["a", "c"]
    assert got.to_pydict() == {"a": [1, 2, 3], "c": [7, 8, 9]}


@pytest.mark.parametrize("fmt", ["parquet", "csv", "json", "arrow", "orc"])
def test_n_rows_caps_every_format(fmt: str, tmp_path: pathlib.Path) -> None:
    out = str(tmp_path / f"t.{fmt}")
    bt.from_pydict({"a": list(range(50))}).write(out, fmt)
    assert getattr(bt.read, fmt)(out, n_rows=4).to_pydict() == {"a": [0, 1, 2, 3]}


def test_n_rows_is_reflected_in_the_row_count(tmp_path: pathlib.Path) -> None:
    """`count()` must answer for the capped relation, not the file behind it."""
    out = str(tmp_path / "t.parquet")
    bt.from_pydict({"a": list(range(100))}).write.parquet(out)
    assert bt.read.parquet(out, n_rows=5).count() == 5


def test_a_capped_read_does_not_poison_the_uncapped_one(tmp_path: pathlib.Path) -> None:
    """A cap makes it a different relation, so it needs its own statistics identity.

    Sharing the path's key let a `n_rows=5` read persist "5 rows" under it, and the next
    full read of the same path answered `count()` as 5 from the cache.
    """
    out = str(tmp_path / "t.parquet")
    bt.from_pydict({"a": list(range(100))}).write.parquet(out)
    assert bt.read.parquet(out, n_rows=5).count() == 5
    assert bt.read.parquet(out).count() == 100
    assert bt.read.parquet(out, n_rows=7).count() == 7
    assert bt.read.parquet(out).count() == 100


def test_n_rows_and_columns_compose(tmp_path: pathlib.Path) -> None:
    out = str(tmp_path / "t.parquet")
    bt.from_pydict({"a": list(range(20)), "b": list(range(20))}).write.parquet(out)
    assert bt.read.parquet(out, columns=["b"], n_rows=3).to_pydict() == {"b": [0, 1, 2]}


def test_unknown_column_names_the_ones_that_exist(tmp_path: pathlib.Path) -> None:
    out = str(tmp_path / "t.parquet")
    bt.from_pydict({"alpha": [1], "beta": [2]}).write.parquet(out)
    with pytest.raises(SchemaError, match=r"(?i)did you mean 'alpha'"):
        _ = bt.read.parquet(out, columns=["alpah"]).schema


def test_negative_n_rows_is_rejected(tmp_path: pathlib.Path) -> None:
    out = str(tmp_path / "t.parquet")
    bt.from_pydict({"a": [1]}).write.parquet(out)
    with pytest.raises(ValueError, match="n_rows"):
        _ = bt.read.parquet(out, n_rows=-1).schema


def test_a_capped_source_advertises_no_footer_statistics(tmp_path: pathlib.Path) -> None:
    """Truncation invalidates the min/max bounds, not just the row count."""
    from batcher.io import ParquetSource

    out = str(tmp_path / "t.parquet")
    bt.from_pydict({"a": list(range(100))}).write.parquet(out)
    assert ParquetSource(out).statistics() is not None
    assert ParquetSource(out, n_rows=5).statistics() is None


def test_a_capped_source_reads_as_one_split(tmp_path: pathlib.Path) -> None:
    """Independent splits would each honor the cap, returning n_rows x len(splits)."""
    from batcher.io import ParquetSource

    out = str(tmp_path / "t.parquet")
    bt.from_pydict({"a": list(range(100))}).write.parquet(out)
    assert len(ParquetSource(out, n_rows=5).splits()) == 1


# ------------------------------------------------------------------------ JSON opts


def test_json_accepts_lines_true(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.jsonl"
    p.write_text('{"a": 1}\n')
    assert bt.read.json(str(p), lines=True).to_pydict() == {"a": [1]}


def test_json_lines_false_is_refused_not_ignored(tmp_path: pathlib.Path) -> None:
    """Ignoring it would read an array file as malformed records and report a bad schema."""
    p = tmp_path / "t.jsonl"
    p.write_text('{"a": 1}\n')
    with pytest.raises(FormatError, match="newline-delimited"):
        bt.read.json(str(p), lines=False)


def test_json_orient_names_the_conversion(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.jsonl"
    p.write_text('{"a": 1}\n')
    with pytest.raises(FormatError, match="orient='records', lines=True"):
        bt.read.json(str(p), orient="split")


def test_json_index_col_says_there_is_no_index(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.jsonl"
    p.write_text('{"a": 1}\n')
    with pytest.raises(FormatError, match="no row index"):
        bt.read.json(str(p), index_col=0)


# ------------------------------------------- pandas spellings of the base options


@pytest.mark.parametrize("fmt", ["parquet", "csv", "json", "orc", "arrow"])
def test_usecols_is_accepted_on_every_format(fmt: str, tmp_path: pathlib.Path) -> None:
    """`usecols` is `columns`; it works everywhere because `FileSource` owns both."""
    out = str(tmp_path / f"t.{fmt}")
    bt.from_pydict({"a": [1, 2], "b": [3, 4]}).write(out, fmt)
    assert getattr(bt.read, fmt)(out, usecols=["b"]).to_pydict() == {"b": [3, 4]}


@pytest.mark.parametrize("fmt", ["parquet", "csv", "json", "orc", "arrow"])
def test_nrows_is_accepted_on_every_format(fmt: str, tmp_path: pathlib.Path) -> None:
    out = str(tmp_path / f"t.{fmt}")
    bt.from_pydict({"a": list(range(10))}).write(out, fmt)
    assert getattr(bt.read, fmt)(out, nrows=2).to_pydict() == {"a": [0, 1]}


def test_a_typo_on_a_format_without_its_own_spec_still_suggests(tmp_path: pathlib.Path) -> None:
    """Parquet has no `__init__` of its own, so the base is what must catch this."""
    out = str(tmp_path / "t.parquet")
    bt.from_pydict({"a": [1]}).write.parquet(out)
    with pytest.raises(FormatError, match=r"(?i)did you mean 'usecols'"):
        bt.read.parquet(out, usecolz=["a"])


def test_passing_both_spellings_of_one_base_option_is_refused(tmp_path: pathlib.Path) -> None:
    out = str(tmp_path / "t.parquet")
    bt.from_pydict({"a": [1], "b": [2]}).write.parquet(out)
    with pytest.raises(FormatError, match="two spellings"):
        bt.read.parquet(out, usecols=["a"], columns=["b"])
