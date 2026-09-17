# Expression recipes

This page assembles the expression language into the jobs people actually reach for it for: porting a pandas or Polars script, building features for a model, and curating a text corpus for training.

Read {doc}`/user-guide/transform/columns/expressions` and {doc}`/user-guide/transform/columns/expression-accessors` first. The blocks below build on each other in order.

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "name": ["Ann", "bob", "CARL"],
        "price": [10.0, 20.0, 30.0],
        "qty": [1, 2, 3],
    }
)
```

## Migrating from Polars or pandas

Batcher has one spelling for each operation, so a script ported from Polars or pandas often calls a method by a name Batcher doesn't use. That call raises an `AttributeError` naming the Batcher spelling, such as `use .is_null` for `isna`, so a port fails at the first unported name instead of running on a guess. Many Polars names are already Batcher's own. On `.str` they include `count_matches`, `extract`, `extract_all`, `replace_all`, `len_chars`, `strip_chars_start`, `strip_chars_end`, and `slice`. On `.dt` they include `weekday`, `date`, `month_start`, `normalize`, and the sub-second `millisecond` / `microsecond` / `nanosecond`. An expression takes `arcsin`/`arccos`/`arctan`/`arcsinh`/`arccosh`/`arctanh` and `skew`, alongside the top-level {py:func}`bt.arctan2(y, x) <batcher.arctan2>`.

The accessor names that differ map as follows, grouped by accessor:

| Accessor | Polars or pandas | Batcher |
|---|---|---|
| `.str` | `to_lowercase`, `to_uppercase` | `lower`, `upper` |
| `.str` | `title` | `to_titlecase` |
| `.str` | `pad_start`, `rjust` | `lpad` |
| `.str` | `pad_end`, `ljust` | `rpad` |
| `.str` | `strip`, `strip_chars` | `trim` |
| `.str` | `head`, `tail` | `left`, `right` |
| `.str` | `len_bytes` | `octet_length` |
| `.str` | `removeprefix`, `removesuffix` | {py:meth}`strip_prefix <batcher.plan.expr_ir.namespaces.strings._StrNamespace.strip_prefix>`, {py:meth}`strip_suffix <batcher.plan.expr_ir.namespaces.strings._StrNamespace.strip_suffix>` |
| `.str` | `isdigit`, `isalpha`, `isalnum`, `isspace` | `is_numeric`, `is_alpha`, `is_alnum`, `is_space` |
| `.dt` | `day_of_week` | `dayofweek` |
| `.dt` | `day_of_year`, `ordinal_day` | `dayofyear` |
| `.dt` | `week_of_year`, `weekofyear` | `week` |
| `.dt` | `day_name`, `month_name` | `dayname`, `monthname` |
| `.dt` | `daysinmonth` | `days_in_month` |
| `.dt` | `month_end` | `last_day` |
| `.dt` | `to_string` | `strftime(format)` |
| `.dt` | `floor(unit)` | `truncate(unit)` |
| `.list` | `lengths` | `len` |
| `.list` | `element_at` | `get` |
| `.list` | `argmin`, `argmax` | `arg_min`, `arg_max` |
| `.list` | `set_union`, `set_intersection`, `set_difference` | `union`, `intersect`, `difference` |

```python
migrate = bt.from_pydict({"name": ["  Ann  "], "code": ["7"]})
out = migrate.select(
    clean=bt.col("name").str.trim().str.upper(),
    padded=bt.col("code").str.lpad(4, "0"),
)
print(out.to_pydict())
# {'clean': ['ANN'], 'padded': ['0007']}
```

On an expression, the pandas method names map to Batcher's names, and each pandas operator method maps to the Python operator it stands for:

| pandas | Batcher |
|---|---|
| `astype` | `cast` |
| `isna`, `isnull` | `is_null` |
| `notna`, `notnull` | `is_not_null` |
| `fillna` | `fill_null` |
| `isin` | `is_in` |
| `nunique` | `count_distinct` |
| `rename` | `alias` |
| `kurt` | `kurtosis` |
| `prod` | `product` |
| `any`, `all` | `bool_or`, `bool_and` |
| `log`, numpy's natural logarithm | `ln` |
| `cumsum`, `cummax`, `cummin`, `cumcount` | `cum_sum`, `cum_max`, `cum_min`, `cum_count` |
| `clip_min(x)`, `clip_max(x)` | `clip(lower=x)`, `clip(upper=x)` |
| `is_between` | `between` |
| `add`, `sub`, `mul`, `truediv`, `div`, `floordiv`, `mod` | `+`, `-`, `*`, `/`, `/`, `//`, `%` |
| `eq`, `ne`, `lt`, `le`, `gt`, `ge` | `==`, `!=`, `<`, `<=`, `>`, `>=` |
| `and_`, `or_`, `not_`, `xor` | `&`, `\|`, `~`, and `!=` for a boolean XOR (`^` is bitwise and returns integers) |

```python
sales = bt.from_pydict({"units": [2, None, 5], "price": [10, 20, 30]})
print(
    sales.select(
        revenue=(bt.col("units").fill_null(0) * bt.col("price")),
        missing=bt.col("units").is_null(),
    ).to_pydict()
)
# {'revenue': [20, 0, 150], 'missing': [False, True, False]}
```

The {py:class}`Dataset <batcher.Dataset>` itself keeps the pandas names `isna`, `notna`, `round`, `abs`, `clip`, `shape`, `size`, `nunique`, and `select_dtypes`, plus `drop_constant_columns`. The frame methods that differ map as follows:

| pandas | Batcher |
|---|---|
| `fillna`, `dropna` | `fill_null`, `drop_nulls` |
| `astype` | `cast` |
| `assign` | `with_columns` |
| `groupby` | `group_by` |
| `merge` | `join` |
| `sort_values` | `sort` |
| `nlargest`, `nsmallest` | `top_k`, `bottom_k` |
| `head(n)` | `limit(n)` |
| `drop_duplicates` | `distinct` |
| `melt` | `unpivot` |
| `sample_frac` | `sample` |

```python
records = bt.from_pydict({"code": ["123", "a1"], "tags": [[3, 1, 2], [5]]})
print(
    records.select(
        numeric=bt.col("code").str.is_numeric(),
        n=bt.col("tags").list.len(),
        smallest=bt.col("tags").list.arg_min(),
        second=bt.col("tags").list.get(1),
    ).to_pydict()
)
# {'numeric': [True, False], 'n': [3, 1], 'smallest': [1, 0], 'second': [1, None]}
```

A few ecosystem names are deliberately missing, because they don't mean the same thing here. `str.find` and `str.index` are absent because `position` is 1-based and returns 0 when the substring is absent, where pandas returns a 0-based index and -1. `str.islower` and `str.isupper` are absent because Batcher's {py:meth}`is_lower <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_lower>` / {py:meth}`is_upper <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_upper>` are true for a string with no cased characters, where Python's are false. Use `str.slice` rather than a `substring` alias, and `str.count_matches` for pandas' regex `count`.

## Feature engineering for data science

Feature engineering runs in the engine, not in pandas, so it scales with the data and never round-trips rows through Python. The expression layer carries the transforms a model pipeline needs, and the scaling and encoding functions each accept `partition_by=` to fit per group: `zscore`, `minmax_scale`, `maxabs_scale`, `mean_center`, `label_encode`, and `hash_bucket` for a reproducible split key. Activations such as `sigmoid`, `logit`, `relu`, and `softplus`, share and ratio features such as {py:meth}`pct_of_total <batcher.plan.expr_ir.core.Expr.pct_of_total>`, {py:meth}`cumulative_pct <batcher.plan.expr_ir.core.Expr.cumulative_pct>`, {py:meth}`normalize_l1 <batcher.plan.expr_ir.core.Expr.normalize_l1>`, {py:meth}`rank_pct <batcher.plan.expr_ir.core.Expr.rank_pct>`, {py:meth}`safe_divide <batcher.plan.expr_ir.core.Expr.safe_divide>`, and the expanding statistics {py:meth}`expanding_mean <batcher.plan.expr_ir.core.Expr.expanding_mean>`, {py:meth}`expanding_var <batcher.plan.expr_ir.core.Expr.expanding_var>`, {py:meth}`expanding_std <batcher.plan.expr_ir.core.Expr.expanding_std>` round it out. Value predicates {py:meth}`is_positive <batcher.plan.expr_ir.core.Expr.is_positive>`, {py:meth}`is_negative <batcher.plan.expr_ir.core.Expr.is_negative>`, {py:meth}`is_zero <batcher.plan.expr_ir.core.Expr.is_zero>`, {py:meth}`is_even <batcher.plan.expr_ir.core.Expr.is_even>`, {py:meth}`is_odd <batcher.plan.expr_ir.core.Expr.is_odd>`, and {py:meth}`is_outlier <batcher.plan.expr_ir.core.Expr.is_outlier>` read as filters.

```python
model = bt.from_pydict({"g": ["a", "a", "b", "b"], "v": [1.0, 3.0, 10.0, 20.0]})
out = model.select(
    z=bt.col("v").zscore(["g"]).round(4),
    scaled=bt.col("v").minmax_scale(["g"]),
    activated=bt.col("v").sigmoid().round(4),
)
print(out.to_pydict())
# {'z': [-0.7071, 0.7071, -0.7071, 0.7071], 'scaled': [0.0, 1.0, 0.0, 1.0], 'activated': [0.7311, 0.9526, 1.0, 1.0]}
```

Calendar features come off `.dt`: {py:meth}`is_weekend <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.is_weekend>` / {py:meth}`is_business_day <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.is_business_day>`, {py:meth}`is_month_start <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.is_month_start>` / {py:meth}`is_month_end <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.is_month_end>`, {py:meth}`is_quarter_start <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.is_quarter_start>` / {py:meth}`is_quarter_end <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.is_quarter_end>`, {py:meth}`is_year_start <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.is_year_start>` / {py:meth}`is_year_end <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.is_year_end>`, plus {py:meth}`quarter_start <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.quarter_start>`, {py:meth}`year_start <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.year_start>`, {py:meth}`days_in_year <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.days_in_year>`, and {py:meth}`week_of_month <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.week_of_month>`, the period closes {py:meth}`quarter_end <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.quarter_end>` and {py:meth}`year_end <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.year_end>`, and the elapsed-time features {py:meth}`seconds_between <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.seconds_between>`, {py:meth}`minutes_between <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.minutes_between>`, {py:meth}`hours_between <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.hours_between>`, {py:meth}`days_between <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.days_between>`, and {py:meth}`weeks_between <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.weeks_between>`. Text features come off `.str`: {py:meth}`word_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.word_count>`, {py:meth}`digit_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.digit_count>`, {py:meth}`contains_all <batcher.plan.expr_ir.namespaces.strings._StrNamespace.contains_all>`, {py:meth}`count_char <batcher.plan.expr_ir.namespaces.strings._StrNamespace.count_char>`, `capitalize`, `remove_punctuation`, and the character-class checks `is_alpha`, {py:meth}`is_numeric <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_numeric>`, {py:meth}`is_alnum <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_alnum>`, {py:meth}`is_space <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_space>`, {py:meth}`is_upper <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_upper>`, {py:meth}`is_lower <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_lower>`.

```python
import datetime as dt

events = bt.from_pydict({"d": [dt.datetime(2024, 2, 3)], "note": ["Hi, there!"]})
out = events.select(
    weekend=bt.col("d").dt.is_weekend(),
    week=bt.col("d").dt.week_of_month(),
    words=bt.col("note").str.word_count(),
    clean=bt.col("note").str.remove_punctuation(),
)
print(out.to_pydict())
# {'weekend': [True], 'week': [1], 'words': [2], 'clean': ['Hi there']}
```

For column profiling, {py:func}`bt.q1 <batcher.q1>` / {py:func}`bt.q3 <batcher.q3>` / {py:func}`bt.iqr <batcher.iqr>` give the spread an outlier cannot move, {py:func}`bt.value_range <batcher.value_range>` the full spread, {py:func}`bt.null_rate <batcher.null_rate>` / {py:func}`bt.non_null_rate <batcher.non_null_rate>` completeness, and {py:func}`bt.nunique_ratio <batcher.nunique_ratio>` the cardinality ratio that separates identifiers from categoricals.

```python
prof = bt.from_pydict({"x": [1.0, None, 3.0, 4.0]})
out = prof.agg(
    spread=bt.iqr("x"),
    rng=bt.value_range("x"),
    missing=bt.null_rate("x"),
    card=bt.nunique_ratio("x"),
)
print(out.to_pydict())
# {'spread': [1.5], 'rng': [3.0], 'missing': [0.25], 'card': [0.75]}
```

## Curating an AI training corpus

Filtering a pretraining corpus is a per-row scan, so it belongs in the engine rather than in a Python loop. The `.str` namespace carries the Gopher and C4 style quality heuristics: the character-class ratios {py:meth}`alpha_ratio <batcher.plan.expr_ir.namespaces.strings._StrNamespace.alpha_ratio>`, {py:meth}`digit_ratio <batcher.plan.expr_ir.namespaces.strings._StrNamespace.digit_ratio>`, {py:meth}`uppercase_ratio <batcher.plan.expr_ir.namespaces.strings._StrNamespace.uppercase_ratio>`, {py:meth}`lowercase_ratio <batcher.plan.expr_ir.namespaces.strings._StrNamespace.lowercase_ratio>`, {py:meth}`punctuation_ratio <batcher.plan.expr_ir.namespaces.strings._StrNamespace.punctuation_ratio>`, {py:meth}`whitespace_ratio <batcher.plan.expr_ir.namespaces.strings._StrNamespace.whitespace_ratio>`, {py:meth}`non_ascii_ratio <batcher.plan.expr_ir.namespaces.strings._StrNamespace.non_ascii_ratio>`, and {py:meth}`alnum_ratio <batcher.plan.expr_ir.namespaces.strings._StrNamespace.alnum_ratio>`, plus the shape statistics {py:meth}`line_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.line_count>`, {py:meth}`mean_line_length <batcher.plan.expr_ir.namespaces.strings._StrNamespace.mean_line_length>`, {py:meth}`avg_word_length <batcher.plan.expr_ir.namespaces.strings._StrNamespace.avg_word_length>`, {py:meth}`sentence_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.sentence_count>`, {py:meth}`non_ascii_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.non_ascii_count>`, {py:meth}`url_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.url_count>`, and {py:meth}`email_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.email_count>`. Thresholding a couple of these removes most boilerplate, link dumps, and machine-generated text.

```python
corpus = bt.from_pydict({"text": ["Real prose, with sentences and words.", "AAA 111 &&& ||| ###"]})
kept = corpus.filter(
    (bt.col("text").str.alpha_ratio() > 0.6)
    & (bt.col("text").str.avg_word_length().between(3, 10))
)
print(kept.to_pydict())
# {'text': ['Real prose, with sentences and words.']}
```

Length is not the only signal. A second measure asks not what a document says but how much of it repeats. A page that is mostly a navigation menu, a template header, or the same sentence emitted twice by a scraper is repetitive rather than short, so a length threshold never catches it. {py:meth}`duplicate_line_ratio <batcher.plan.expr_ir.namespaces.strings._StrNamespace.duplicate_line_ratio>` and {py:meth}`duplicate_paragraph_ratio <batcher.plan.expr_ir.namespaces.strings._StrNamespace.duplicate_paragraph_ratio>` report the share of the document taken up by repeated lines and by repeated blank-line-separated paragraphs. Both weigh by *characters* rather than by count, following Gopher, so one repeated long paragraph counts for more than one repeated word.

```python
docs = bt.from_pydict(
    {
        "body": [
            "unique one\nunique two\nunique three\nunique four",
            "same line\nsame line\nsame line\nsame line",
        ]
    }
)
out = docs.select(dup=bt.col("body").str.duplicate_line_ratio())
print(out.to_pydict())
# {'dup': [0.0, 0.75]}
```

Each is null where the document has nothing to measure, so an empty extraction fails a threshold rather than sliding under it. The corpus-level aggregates of the same properties are in {doc}`/api/models/metrics`.

Document shape adds {py:meth}`paragraph_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.paragraph_count>`, {py:meth}`is_single_line <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_single_line>`, {py:meth}`ends_with_punctuation <batcher.plan.expr_ir.namespaces.strings._StrNamespace.ends_with_punctuation>`, {py:meth}`has_repeated_punctuation <batcher.plan.expr_ir.namespaces.strings._StrNamespace.has_repeated_punctuation>`, {py:meth}`quote_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.quote_count>`, {py:meth}`paren_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.paren_count>`, {py:meth}`digit_to_word_ratio <batcher.plan.expr_ir.namespaces.strings._StrNamespace.digit_to_word_ratio>`, and the code detectors {py:meth}`code_fence_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.code_fence_count>` and {py:meth}`looks_like_code <batcher.plan.expr_ir.namespaces.strings._StrNamespace.looks_like_code>`. Further signals include {py:meth}`uppercase_word_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.uppercase_word_count>`, {py:meth}`long_word_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.long_word_count>`, {py:meth}`symbol_to_word_ratio <batcher.plan.expr_ir.namespaces.strings._StrNamespace.symbol_to_word_ratio>`, {py:meth}`hashtag_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.hashtag_count>`, {py:meth}`mention_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.mention_count>`, {py:meth}`phone_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.phone_count>`, and {py:meth}`has_phone <batcher.plan.expr_ir.namespaces.strings._StrNamespace.has_phone>`. Cleaning and PII scrubbing use {py:meth}`remove_urls <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_urls>`, {py:meth}`remove_emails <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_emails>`, {py:meth}`remove_phones <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_phones>`, the shape-preserving {py:meth}`mask_emails <batcher.plan.expr_ir.namespaces.strings._StrNamespace.mask_emails>` / {py:meth}`mask_urls <batcher.plan.expr_ir.namespaces.strings._StrNamespace.mask_urls>`, {py:meth}`remove_non_ascii <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_non_ascii>`, {py:meth}`remove_digits <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_digits>`, and {py:meth}`remove_html_tags <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_html_tags>`. `left(n)` caps a row at a character budget, and {py:meth}`truncate_words <batcher.plan.expr_ir.namespaces.strings._StrNamespace.truncate_words>` caps it at a word budget without cutting mid-word. The detection predicates {py:meth}`has_url <batcher.plan.expr_ir.namespaces.strings._StrNamespace.has_url>`, {py:meth}`has_email <batcher.plan.expr_ir.namespaces.strings._StrNamespace.has_email>`, {py:meth}`has_non_ascii <batcher.plan.expr_ir.namespaces.strings._StrNamespace.has_non_ascii>`, {py:meth}`has_digits <batcher.plan.expr_ir.namespaces.strings._StrNamespace.has_digits>`, {py:meth}`has_html <batcher.plan.expr_ir.namespaces.strings._StrNamespace.has_html>`, {py:meth}`is_ascii_only <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_ascii_only>`, {py:meth}`is_blank <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_blank>`, {py:meth}`starts_with_bullet <batcher.plan.expr_ir.namespaces.strings._StrNamespace.starts_with_bullet>`, and {py:meth}`looks_like_json <batcher.plan.expr_ir.namespaces.strings._StrNamespace.looks_like_json>` read as filters. For context windows, {py:meth}`estimate_tokens <batcher.plan.expr_ir.namespaces.strings._StrNamespace.estimate_tokens>` and {py:meth}`fits_token_budget <batcher.plan.expr_ir.namespaces.strings._StrNamespace.fits_token_budget>` give a tokenizer-free size estimate.

```python
raw = bt.from_pydict({"text": ["Mail bob@x.com or see http://y.io for more"]})
print(raw.select(clean=bt.col("text").str.remove_emails().str.remove_urls()).to_pydict())
# {'clean': ['Mail  or see  for more']}
```

Counts and predicates round it out: {py:meth}`newline_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.newline_count>`, {py:meth}`tab_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.tab_count>`, {py:meth}`space_count <batcher.plan.expr_ir.namespaces.strings._StrNamespace.space_count>`, {py:meth}`word_char_ratio <batcher.plan.expr_ir.namespaces.strings._StrNamespace.word_char_ratio>`, {py:meth}`avg_sentence_length <batcher.plan.expr_ir.namespaces.strings._StrNamespace.avg_sentence_length>`, {py:meth}`is_short <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_short>` / {py:meth}`is_long <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_long>`, {py:meth}`is_question <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_question>`, {py:meth}`is_exclamation <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_exclamation>`, {py:meth}`starts_with_capital <batcher.plan.expr_ir.namespaces.strings._StrNamespace.starts_with_capital>`, {py:meth}`is_all_caps <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_all_caps>`, {py:meth}`has_currency <batcher.plan.expr_ir.namespaces.strings._StrNamespace.has_currency>`, and the whole-string {py:meth}`is_url <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_url>` / {py:meth}`is_email <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_email>`. Extraction gives {py:meth}`extract_urls <batcher.plan.expr_ir.namespaces.strings._StrNamespace.extract_urls>`, {py:meth}`extract_emails <batcher.plan.expr_ir.namespaces.strings._StrNamespace.extract_emails>`, {py:meth}`extract_numbers <batcher.plan.expr_ir.namespaces.strings._StrNamespace.extract_numbers>`, {py:meth}`extract_hashtags <batcher.plan.expr_ir.namespaces.strings._StrNamespace.extract_hashtags>`, {py:meth}`extract_mentions <batcher.plan.expr_ir.namespaces.strings._StrNamespace.extract_mentions>`, {py:meth}`first_sentence <batcher.plan.expr_ir.namespaces.strings._StrNamespace.first_sentence>`, `first_word`, and `last_word`. Normalization gives `slugify`, `remove_bullets`, {py:meth}`remove_repeated_punctuation <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_repeated_punctuation>`, {py:meth}`remove_markdown_links <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_markdown_links>`, {py:meth}`remove_code_blocks <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_code_blocks>`, {py:meth}`remove_stopwords <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_stopwords>`, and {py:meth}`truncate_sentences <batcher.plan.expr_ir.namespaces.strings._StrNamespace.truncate_sentences>`.

An embedding is a list column, so its vector methods live on {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>` alongside the reductions above: `len` for the dimension, `mean` and `max` for mean and max pooling, `is_zero_vector`, `sum_squares`, `l2_norm`, `is_unit_norm` to assert normalization before a cosine search, {py:meth}`l2_distance <batcher.plan.expr_ir.namespaces.collections._ListNamespace.l2_distance>`, and {py:meth}`angular_distance <batcher.plan.expr_ir.namespaces.collections._ListNamespace.angular_distance>`. Preparing the training set itself uses {py:meth}`ds.shuffle(seed=) <batcher.Dataset.shuffle>`, {py:meth}`ds.stratified_split(label, test_size) <batcher.Dataset.stratified_split>`, {py:meth}`ds.sample_per_group(by, n) <batcher.Dataset.sample_per_group>`, {py:meth}`ds.class_balance(label) <batcher.Dataset.class_balance>`, and {py:meth}`ds.class_weights(label) <batcher.Dataset.class_weights>`.

```python
labelled = bt.from_pydict({"y": ["a"] * 6 + ["b"] * 2, "x": list(range(8))})
train, test = labelled.stratified_split("y", 0.25, seed=5)
print(labelled.class_weights("y").sort("y").to_pydict())
# {'y': ['a', 'b'], 'weight': [0.6666666666666666, 2.0]}
```

## See also

- {doc}`/user-guide/transform/columns/expressions`: the core language these recipes are built from.
- {doc}`/user-guide/transform/columns/expression-accessors`: the {py:class}`.str <batcher.plan.expr_ir.namespaces.strings._StrNamespace>`, {py:class}`.dt <batcher.plan.expr_ir.namespaces.temporal._DtNamespace>`, {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>`, {py:class}`.struct <batcher.plan.expr_ir.namespaces.collections._StructNamespace>`, and {py:class}`.json <batcher.plan.expr_ir.namespaces.collections._JsonNamespace>` methods.
- {doc}`/ml/preparing/preprocessors/index`: fitted, reusable versions of the feature steps here.
- {doc}`/getting-started/migration/transforming`: the verb-by-verb table behind the porting section.
- {doc}`/cookbook/expressions/index`: the same territory, one self-contained script per topic.
