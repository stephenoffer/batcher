"""The `.str` namespace beyond `ops-strings`: split, regex rewrite and extract, hashing.

`ops-strings` times `LIKE`, `length`, `upper`, `substring` and `replace` through SQL. These are
the string calls that cost the most per row and that a DataFrame user reaches for when parsing
text: splitting on a delimiter, rewriting every regex match, pulling out a capture group, and
hashing. All run over `lineitem`'s 6M free-text `l_comment` values (or the short
`l_shipinstruct`), each reduced to a handful of rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from registry import suite

from ._engines import frame_case

if TYPE_CHECKING:
    from context import Context

text = suite("ops-frame-text", dataset="operators")


@text.case("op-str-split-part")
def split_part(ctx: Context):
    """Line count per first word of `l_shipinstruct` (`split_part(s, ' ', 1)`, 4 groups)."""
    import batcher as bt

    def batcher(ds):
        word = bt.col("l_shipinstruct").str.split_part(" ", 1).alias("w")
        return ds.with_columns(word).group_by("w").agg(n=bt.col("w").count()).to_arrow()

    def polars(lf):
        import polars as pl

        word = pl.col("l_shipinstruct").str.split(" ").list.get(0).alias("w")
        return lf.group_by(word).agg(pl.len().alias("n"))

    def pyarrow(t):
        word = pc.list_element(pc.split_pattern(t["l_shipinstruct"], " "), 0)
        out = pa.table({"w": word}).group_by("w").aggregate([("w", "count")])
        return pa.table({"w": out["w"], "n": out["w_count"]})

    sql = "SELECT split_part(l_shipinstruct, ' ', 1) AS w, count(*) AS n FROM {t} GROUP BY 1"
    return frame_case(ctx, batcher=batcher, polars=polars, sql=sql, pyarrow=pyarrow)


@text.case("op-str-regex-replace-all")
def regex_replace_all(ctx: Context):
    """Total length of `l_comment` with every vowel removed (a regex replace-all, per row)."""
    import batcher as bt

    def batcher(ds):
        stripped = bt.col("l_comment").str.replace_all("[aeiou]", "")
        return ds.agg(total=stripped.str.len_chars().sum()).to_arrow()

    def polars(lf):
        import polars as pl

        stripped = pl.col("l_comment").str.replace_all("[aeiou]", "")
        return lf.select(stripped.str.len_chars().cast(pl.Int64).sum().alias("total"))

    def pyarrow(t):
        stripped = pc.replace_substring_regex(t["l_comment"], "[aeiou]", "")
        return pa.table({"total": [pc.sum(pc.utf8_length(stripped)).as_py()]})

    sql = "SELECT sum(length(regexp_replace(l_comment, '[aeiou]', '', 'g'))) AS total FROM {t}"
    return frame_case(ctx, batcher=batcher, polars=polars, sql=sql, pyarrow=pyarrow)


@text.case("op-str-regex-extract")
def regex_extract(ctx: Context):
    """How many comments hold an `-ly` adverb, and how many distinct ones (a capture group)."""
    import batcher as bt

    def batcher(ds):
        adverb = bt.col("l_comment").str.extract("([a-z]+)ly", 1).alias("adv")
        hits = ds.with_columns(adverb).filter(bt.col("adv") != "")
        return hits.agg(n=bt.col("adv").count(), nd=bt.col("adv").count_distinct()).to_arrow()

    def polars(lf):
        import polars as pl

        adverb = pl.col("l_comment").str.extract("([a-z]+)ly", 1)
        hits = lf.select(adverb.alias("adv")).filter(pl.col("adv").is_not_null())
        return hits.select(pl.len().alias("n"), pl.col("adv").n_unique().cast(pl.Int64).alias("nd"))

    def pyarrow(t):
        found = pc.struct_field(pc.extract_regex(t["l_comment"], r"(?P<adv>[a-z]+)ly"), "adv")
        hits = pc.filter(found, pc.is_valid(found))
        return pa.table({"n": [len(hits)], "nd": [pc.count_distinct(hits).as_py()]})

    sql = (
        "SELECT count(*) AS n, count(DISTINCT adv) AS nd FROM "
        "(SELECT regexp_extract(l_comment, '([a-z]+)ly', 1) AS adv FROM {t}) WHERE adv <> ''"
    )
    return frame_case(ctx, batcher=batcher, polars=polars, sql=sql, pyarrow=pyarrow)


@text.case("op-str-md5")
def md5(ctx: Context):
    """Distinct MD5 digests of `l_comment` -- a per-row cryptographic hash, then a distinct."""
    import batcher as bt

    def batcher(ds):
        return ds.agg(n=bt.col("l_comment").str.md5().count_distinct()).to_arrow()

    # Polars and PyArrow have no MD5 function (Polars' `hash` is a different function whose
    # values a gate could not compare), so the case is Batcher's API against DuckDB's `md5`.
    sql = "SELECT count(DISTINCT md5(l_comment)) AS n FROM {t}"
    return frame_case(ctx, batcher=batcher, sql=sql)
