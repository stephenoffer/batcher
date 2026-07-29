# Expression recipes

This page assembles the expression language into the jobs people actually reach for it
for: porting a pandas or Polars script, building features for a model, and curating a
text corpus for training.

Read {doc}`expressions` and {doc}`expression-accessors` first. Every example here
runs against the engine, and blocks share one namespace and execute in order.

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

Coming from another DataFrame library, the operation you know by its Polars or pandas
name is usually available under that name too, delegating to Batcher's SQL-style
primary. On `.str`: `to_lowercase`, `to_uppercase`, `to_titlecase`, `pad_start`,
`pad_end`, `ljust`, `rjust`, `count_matches`, `extract`, `extract_all`, `replace_all`,
`len_chars`, `len_bytes`, `strip_chars`, `strip_chars_start`, `strip_chars_end`, `head`,
`tail`, and `slice`. On `.dt`: `weekday`, `ordinal_day`, `to_string`, `date`,
`month_start`, `month_end`, and the sub-second `millisecond` / `microsecond` /
`nanosecond`. On `.list`: `set_union`, `set_intersection`, `set_difference`. On an
expression: `arcsin`/`arccos`/`arctan`/`arcsinh`/`arccosh`/`arctanh`, `clip_min` /
`clip_max`, and `is_between`; plus top-level `bt.arctan2(y, x)`.

The pandas spellings are there too. On `.str`: `strip`, `startswith`, `endswith`,
`match`, `title`, and Python's `removeprefix` / `removesuffix`; on `.dt`: `day_name`,
`month_name`, `daysinmonth`, `weekofyear`, `normalize`, and `floor(unit)`. On the
`Dataset` itself: `fillna`, `dropna`, `isna`, `notna`, `astype`, `assign`, `groupby`,
`merge`, `sort_values`, `nlargest`, `nsmallest`, `round`, `abs`, `clip`, `shape`,
`size`, plus `nunique`, `select_dtypes`, `sample_frac`, and `drop_constant_columns`.

```python
migrate = bt.from_pydict({"name": ["  Ann  "], "code": ["7"]})
out = migrate.select(
    clean=bt.col("name").str.strip_chars().str.to_uppercase(),
    padded=bt.col("code").str.rjust(4, "0"),
)
print(out.to_pydict())
# {'clean': ['ANN'], 'padded': ['0007']}
```

An expression carries the pandas names as well, so a ported column computation runs
without a find-and-replace pass: `astype`, `isna`, `isnull`, `notna`, `notnull`,
`fillna`, `isin`, `nunique`, `rename`, `skew`, `kurt`, `prod`, `any`, `all`, `log`
(numpy's natural logarithm), and the cumulative `cumsum`, `cummax`, `cummin`,
`cumcount`. Each operator has its pandas method form too (`add`,
`sub`, `mul`, `truediv`, `div`, `floordiv`, `mod`, `eq`, `ne`, `lt`, `le`, `gt`, `ge`),
along with `and_`, `or_`, `not_`, and `xor` for the boolean operators that Python's
keywords can't express.

```python
sales = bt.from_pydict({"units": [2, None, 5], "price": [10, 20, 30]})
print(sales.select(
    revenue=bt.col("units").fillna(0).mul(bt.col("price")),
    missing=bt.col("units").isna(),
).to_pydict())
# {'revenue': [20, 0, 150], 'missing': [False, True, False]}
```

The typed accessors follow the same rule. `.str` answers to Python's own string
predicates `isdigit`, `isalpha`, `isalnum`, and `isspace`, and to Polars'
`strip_prefix` / `strip_suffix`. `.dt` takes the snake_case `day_of_week`,
`day_of_year`, and `week_of_year`. `.list` takes `lengths`, PySpark's `element_at`, and
numpy's `argmin` / `argmax`.

```python
records = bt.from_pydict({"code": ["123", "a1"], "tags": [[3, 1, 2], [5]]})
print(records.select(
    numeric=bt.col("code").str.isdigit(),
    n=bt.col("tags").list.lengths(),
    smallest=bt.col("tags").list.argmin(),
    second=bt.col("tags").list.element_at(1),
).to_pydict())
# {'numeric': [True, False], 'n': [3, 1], 'smallest': [1, 0], 'second': [1, None]}
```

A few ecosystem names are deliberately missing, because they don't mean the same thing
here. `str.find` and `str.index` are absent because `position` is 1-based and returns 0
when the substring is absent, where pandas returns a 0-based index and -1. `str.islower`
and `str.isupper` are absent because Batcher's `is_lower` / `is_upper` are true for a
string with no cased characters, where Python's are false. Use `str.slice` rather than a
`substring` alias, and `str.regexp_count` for pandas' regex `count`.

## Feature engineering for data science

The expression layer carries the transforms a model pipeline needs, so feature
engineering runs in the engine rather than in pandas. The scaling and encoding functions
each accept `partition_by=` to fit per group: `zscore`, `minmax_scale`, `maxabs_scale`,
`mean_center`, `label_encode`, and `hash_bucket` for a reproducible split key. Activations (`sigmoid`, `logit`, `relu`, `softplus`), share/ratio features
(`pct_of_total`, `cumulative_pct`, `normalize_l1`, `rank_pct`, `safe_divide`), and the
expanding statistics (`expanding_mean`, `expanding_var`, `expanding_std`) round it out.
Value predicates `is_positive`, `is_negative`, `is_zero`, `is_even`, `is_odd`, and
`is_outlier` read as filters.

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

Calendar features come off `.dt`: `is_weekend` / `is_weekday` (spelled `is_business_day` if you are coming from Polars), `is_month_start` /
`is_month_end`, `is_quarter_start` / `is_quarter_end`, `is_year_start` /
`is_year_end`, plus `quarter_start`, `year_start`, `days_in_year`, and
`week_of_month`, the period closes `quarter_end` and `year_end`, and the elapsed-time
features `seconds_between`, `minutes_between`, `hours_between`, `days_between`, and
`weeks_between`. Text features come off `.str`: `word_count`, `digit_count`,
`contains_all`, `count_char`,
`capitalize`, `remove_punctuation`, and the character-class checks `is_alpha`,
`is_numeric`, `is_alnum`, `is_space`, `is_upper`, `is_lower`.

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

For column profiling, `bt.q1` / `bt.q3` / `bt.iqr` give the robust spread,
`bt.value_range` the full spread, `bt.null_rate` / `bt.non_null_rate` completeness, and
`bt.nunique_ratio` the cardinality ratio that separates identifiers from categoricals.

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

Filtering a pretraining corpus is a per-row scan, so it runs in the engine. The `.str`
namespace carries the Gopher / C4-style quality heuristics: the character-class ratios
`alpha_ratio`, `digit_ratio`, `uppercase_ratio`, `lowercase_ratio`,
`punctuation_ratio`, `whitespace_ratio`, `non_ascii_ratio`, and `alnum_ratio`, plus the
shape statistics `line_count`, `mean_line_length`, `avg_word_length`, `sentence_count`,
`non_ascii_count`, `url_count`, and `email_count`. Thresholding a couple of these
removes most boilerplate, link dumps, and machine-generated text.

```python
corpus = bt.from_pydict(
    {"text": ["Real prose, with sentences and words.", "AAA 111 &&& ||| ###"]}
)
kept = corpus.filter(
    (bt.col("text").str.alpha_ratio() > 0.6)
    & (bt.col("text").str.avg_word_length().is_between(3, 10))
)
print(kept.to_pydict())
# {'text': ['Real prose, with sentences and words.']}
```

Document shape adds `paragraph_count`, `is_single_line`, `ends_with_punctuation`,
`has_repeated_punctuation`, `quote_count`, `paren_count`, `digit_to_word_ratio`, and the
code detectors `code_fence_count` and `looks_like_code`.
Further signals include `uppercase_word_count`, `long_word_count`,
`symbol_to_word_ratio`, `hashtag_count`, `mention_count`, `phone_count`, and
`has_phone`. Cleaning and PII scrubbing use `remove_urls`, `remove_emails`,
`remove_phones`, the shape-preserving `mask_emails` / `mask_urls`, `remove_non_ascii`,
`remove_digits`, and `remove_html_tags`; `truncate_chars` and `truncate_words` cap a row
to a budget without cutting mid-word. The detection predicates `has_url`, `has_email`,
`has_non_ascii`, `has_digits`, `has_html`, `is_ascii_only`, `is_blank`,
`starts_with_bullet`, and `looks_like_json` read as filters. For context windows,
`estimate_tokens` and `fits_token_budget` give a tokenizer-free size estimate.

```python
raw = bt.from_pydict({"text": ["Mail bob@x.com or see http://y.io for more"]})
print(raw.select(clean=bt.col("text").str.remove_emails().str.remove_urls()).to_pydict())
# {'clean': ['Mail  or see  for more']}
```

Counts and predicates round it out: `newline_count`, `tab_count`, `space_count`,
`word_char_ratio`, `avg_sentence_length`, `is_short` / `is_long`, `is_question`,
`is_exclamation`, `starts_with_capital`, `is_all_caps`, `has_currency`, and the
whole-string `is_url` / `is_email`. Extraction gives `extract_urls`, `extract_emails`,
`extract_numbers`, `extract_hashtags`, `extract_mentions`, `first_sentence`,
`first_word`, and `last_word`; normalization gives `slugify`, `remove_bullets`,
`remove_repeated_punctuation`, `remove_markdown_links`, `remove_code_blocks`,
`remove_stopwords`, and `truncate_sentences`.

An embedding is a list column, so its vector methods live on `.list` alongside the
reductions above: `dim`, `is_zero_vector`, `sum_squares`, `mean_pool`, `max_pool`,
`magnitude`, `is_unit_norm` (assert normalization before a cosine search),
`euclidean_distance`, and `angular_distance`. Preparing the training set itself
uses `ds.shuffle(seed=)`, `ds.stratified_split(label, test_size)`,
`ds.sample_per_group(by, n)`, `ds.class_balance(label)`, and `ds.class_weights(label)`.

```python
labelled = bt.from_pydict({"y": ["a"] * 6 + ["b"] * 2, "x": list(range(8))})
train, test = labelled.stratified_split("y", 0.25, seed=5)
print(labelled.class_weights("y").sort("y").to_pydict())
# {'y': ['a', 'b'], 'weight': [0.6666666666666666, 2.0]}
```

## See also

:::{seealso}
- {doc}`expressions`: the core language these recipes are built from.
- {doc}`expression-accessors`: the `.str`, `.dt`, `.list`, `.struct`, and `.json` methods.
- {doc}`../ml/preprocessors/index`: fitted, reusable versions of the feature steps here.
- {doc}`../migration/transforming`: the verb-by-verb table behind the porting section.
:::
