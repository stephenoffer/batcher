# Expression accessors API

Breadth on `Expr` lives on accessor namespaces rather than on the expression itself, so
a hundred string functions sit behind `.str` instead of on every expression. This page
enumerates every method on every namespace.

The `Expr` methods these hang off are in {doc}`/api/relational/expressions`.

```python
import batcher as bt
```

## Accessor namespaces

Each namespace and the methods it carries:

| Namespace | Covers |
| --- | --- |
| `.str` | `upper`, `lower`, `trim(chars=None)`, `lstrip`/`rstrip(chars=None)`, `len`, `contains`, `starts_with`, `ends_with`, `like`, `ilike`, `substr`, `left`, `right`, `split`, `split_part(delim, n)`, `strip_html()` (markup → prose; drops `<script>`/`<style>` bodies and decodes entities), `chunk(size, overlap=0, boundary="char")` (RAG document splitter; `boundary` may be `"char"`/`"word"`/`"sentence"`/`"line"` so a chunk never ends mid-word), `token_ngrams(n)` (every window of `n` whitespace tokens, the unit BLEU and ROUGE-N are defined on), `minhash(num_perm=128, ngram=5)` (fuzzy-dedup signature), `replace`, `regexp_replace`, `regexp_replace_all`, `regexp_extract`, `initcap`, `hex`, `base64`, `translate`, `zfill(width)` (zero-pad numeric strings), `contains_any([...])` (true if any literal substring is present), and more |
| `.dt` | `year`, `month`, `day`, `hour`, `minute`, `second`, `quarter`, `week`, `dayofweek`, `dayofyear`, `dayname`, `monthname`, `epoch`, `epoch_ms()` / `epoch_us()` / `epoch_ns()` (integer epoch at ms/µs/ns resolution), `iso_year`, `is_leap_year`, `days_in_month`, `truncate(unit)`, `strftime(fmt)`, `offset_by("1mo15d")`, `convert_timezone(from_tz, to_tz)` (DST-aware), `to_string(fmt)` (ISO-8601 by default), `timestamp(unit)` (the Polars epoch reader), `is_business_day()` (the weekday test, Polars' spelling of `is_weekday`), and more |
| `.list` | `len`, `sum`, `min`, `max`, `mean`, `median`, `std`, `var`, `product`, `n_unique`, `l2_norm`, `l1_norm` (Manhattan magnitude), `max_abs` (MaxAbs-scaling divisor), `normalize`, `softmax` (per-row logits→probabilities), `log_softmax` (the log-domain form, which stays finite where `softmax` underflows), `entropy` (per-row Shannon entropy in nats, the uncertainty of a score or probability vector), `arg_sort` (indices sorting ascending, so reverse for top-k), `cum_sum` (cumulative sum), `diff` (first difference with a leading null, for delta features), `sort`, `reverse`, `unique`, `flatten`, `get(i)` (negative ok), `first()`, `last()`, `slice`, `head(n)`, `contains(v)`, `position(v)`, `gather(indices)` (take each row's elements at the given positions — what makes `arg_sort` usable, so a rerank stays in the engine), `intersect(o)`, `difference(o)`, `union(o)`, `concat(o)` (append, keeping duplicates, as in DuckDB `list_concat` rather than a set op), `has_all(o)` / `has_any(o)` (containment tests, DuckDB `list_has_all`/`list_has_any`), `transform(element()-expr)`, `filter(element()-pred)`, `drop_nulls()` (the null-dropping filter, Polars `list.drop_nulls`), `join(sep)`, `add(o)` / `subtract(o)` / `multiply(o)` (element-wise vector arithmetic → List<Float64>); vector ops `dot(o)`, `cosine_similarity(o)`, `cosine_distance(o)`, `l2_distance(o)`, `l1_distance(o)` (Manhattan), `hamming_distance(o)` (differing positions, for binary or quantized embeddings), `jaccard(o)` (agreement rate; the MinHash/SimHash similarity estimate), `multiset_overlap(o)` (clipped bag intersection size, counting repeats — the BLEU/ROUGE-N numerator), `lcs_length(o)` (longest common *subsequence* length — the one overlap measure that reads order, and the ROUGE-L numerator; `O(n*m)` per row), `simhash(num_bits=64, seed=0)` (random-hyperplane LSH signature, the blocking key for a vector similarity join) |
| `.struct` | `field(name)`, `get(name)`, `keys()` |
| `.json` | `extract_string(path)` |
| `.map` | `get(key)`, `keys()`, `values()`, `len()` (entry count, DuckDB `cardinality`), `contains(key)` (DuckDB `map_contains`), which read a `Map`-typed column |
| `.image` | `decode()` (header-only struct `{width, height, channels, mode}` from a single read), `to_tensor(width, height)`, `to_tensor_f32(width, height, mean=, std=, channels_first=)` (model-ready `float32` tensor: scale to `[0,1]`, per-channel normalize, HWC/CHW), `center_crop(width, height)` (centered crop, torchvision-style zero-pad when smaller), `to_grayscale(width, height)` (decode+resize to a single Rec.601 luma channel), `resize(width, height)` (re-encode to PNG bytes), `crop(x, y, width, height)` (an arbitrary window as PNG bytes, clipped at the edge rather than padded), `encode(format)` (re-encode as `png`/`jpeg`/`bmp`/`gif`), `convert(mode)` (change color mode to `L`/`LA`/`RGB`/`RGBA`; Rec. 601 luma, the same weighting `to_grayscale` and `dhash` use), `brightness()` (mean luma in `[0, 1]`, the blank-tile detector), `sharpness()` (normalized Laplacian variance, the focus measure that finds a corpus's blurred tail), `dhash()` (64-bit perceptual hash for near-duplicate detection) |
| `.audio` | `decode()`, `to_waveform()` (decode to a mono PCM `List<Float>` signal), `resample(rate)` (decode + band-limited resample to `rate` Hz, the 16 kHz audio-ML preprocessing step), `trim_silence(threshold_db=-40)` (drop the leading and trailing quiet; a clip silent throughout trims to an empty list, which is how a silent recording is filtered out), `peak_normalize()` (scale so the loudest sample sits at full scale, the level-matching step before batching clips from different sources), `zero_crossing_rate()` (the voiced/unvoiced descriptor, in `[0, 1]`), `mel_spectrogram(rate, n_fft=400, hop_length=160, n_mels=80)` (the speech-model mel power-spectrogram front end; matches `torchaudio.transforms.MelSpectrogram`), `mfcc(rate, n_fft=400, hop_length=160, n_mels=128, n_mfcc=40)` (Mel-Frequency Cepstral Coefficients; matches `torchaudio.transforms.MFCC`) |
| `.video` | `decode()` |

### More `.str` methods

The table above lists the common string operations. These are the rest, covering padding,
casing, encoding, and the similarity measures:

| Method | Description |
| --- | --- |
| `.lpad(width, fill=" ")` / `.rpad(width, fill=" ")` | pad to `width` characters with `fill` (cycled); truncate if longer |
| `.repeat(n)` | repeat the string `n` times (`n` ≤ 0 → empty) |
| `.normalize_whitespace()` | collapse every run of whitespace to a single space and trim the ends |
| `.position(pattern)` | 1-based index of `pattern`, 0 if absent (→ Int64) |
| `.regexp_matches(pattern)` | true where the regex matches anywhere (→ Bool) |
| `.ascii()` | Unicode codepoint of the first character, 0 if empty (→ Int64) |
| `.bit_length()` / `.octet_length()` | number of bits / UTF-8 bytes in the string (→ Int64) |
| `.from_base64()` | decode standard base64 to a UTF-8 string; null if invalid |
| `.unhex()` | decode pairs of hex digits to a UTF-8 string; null if invalid |
| `.md5()` / `.sha1()` / `.sha256()` | cryptographic digest as lowercase hex; null → null |
| `.crc32()` | CRC-32 (IEEE) checksum of the UTF-8 bytes (Spark `crc32`, → Int64) |
| `.hash64()` | deterministic FNV-1a 64-bit hash, stable across partitions and machines, so it's a surrogate-key building block (→ Int64) |
| `.xxhash64()` | fast non-cryptographic 64-bit xxHash; the standard bucketing/sharding hash (→ Int64) |
| `.substring_index(delimiter, count)` | substring before the `count`-th `delimiter` (Spark) |
| `.overlay(replacement, pos, length=None)` | replace `length` chars from 1-based `pos` (SQL `OVERLAY`) |
| `.regexp_extract_all(pattern)` | every regex match as a `List<Utf8>` (DuckDB `regexp_extract_all`) |
| `.regexp_count(pattern)` | number of non-overlapping regex matches (→ Int64) |
| `.regexp_split(pattern)` | split on every regex match into a `List<Utf8>` (DuckDB `regexp_split_to_array`) |
| `.levenshtein(target)` | edit distance to the constant `target` (DuckDB `levenshtein`, → Int64) |
| `.damerau_levenshtein(target)` | edit distance to `target` counting an adjacent-swap as one edit (DuckDB `damerau_levenshtein`, → Int64), which handles typos better |
| `.jaro_similarity(target)` | Jaro similarity to `target`, `[0,1]` (DuckDB `jaro_similarity`, → Float64), for fuzzy matching and record linkage |
| `.jaro_winkler_similarity(target)` | Jaro-Winkler similarity to `target`, `[0,1]` (DuckDB `jaro_winkler_similarity`, → Float64), prefix-weighted for name matching |
| `.soundex()` | American Soundex phonetic code, a 4-character key (→ Utf8) |
| `.hamming(target)` | positions at which the value and `target` differ (DuckDB `hamming`/`mismatches`, → Int64); the lengths must be equal |
| `.jaccard(target)` | Jaccard similarity of the two values' character *sets*, `[0,1]` (DuckDB `jaccard`, → Float64) |
| `.url_encode()` / `.url_decode()` | percent-encode a URL *component* and its inverse (DuckDB `url_encode`/`url_decode`) |
| `.regexp_escape()` | escape the regex metacharacters, so a value can be embedded in a pattern as a literal (DuckDB `regexp_escape`) |
| `.escape_regex()` | the Polars spelling of `regexp_escape` |
| `.join(sep)` | concatenate every value of the group into one string, as an aggregate (SQL `string_agg`, Polars `str.join`) |
| `.parse_filename()` | the final component of a path (DuckDB `parse_filename`) |
| `.parse_dirname()` | the *first* component of a path, which is `/` for an absolute one (DuckDB `parse_dirname`) |
| `.parse_dirpath()` | everything before the last separator, i.e. the containing directory (DuckDB `parse_dirpath`) |
| `.parse_path()` | the path's components as a `List<Utf8>`, keeping a leading `/` (DuckDB `parse_path`) |
| `.to_binary()` / `.from_binary()` | the UTF-8 bytes as `0`/`1` text and back; undecodable input is null (DuckDB `to_binary`/`from_binary`) |
| `.to_date(format="%Y-%m-%d")` | parse into a Date with a strftime format; unmatched → NULL (→ Date32) |
| `.to_datetime(format)` | parse into a Timestamp (DuckDB `try_strptime`); unmatched → NULL (→ Timestamp(us)) |
| `.to_case(style)` | re-case an identifier into `style`, one of `snake`, `upper_snake`, `camel`, `pascal`, `kebab`, `upper_kebab`, `title`, `sentence`, `dot`, or `train` |
| `.compress(codec)` | compress the raw bytes with `gzip`, `zlib`, `deflate`, `zstd`, `brotli`, or `lz4` (→ Binary) |
| `.decompress(codec)` | the inverse; a frame that isn't valid for `codec` is null, not an error |

### More `.dt` methods

`.century()`, `.decade()`, `.isodow()` (ISO day of week), `.last_day()` (last day
of the month), and `.millennium()`. Each extracts the named field of a date/time column (→ Int64).

### More `.json` methods

These extract a typed value at a JSON path, or describe the document's shape when you do
not yet know it:

| Method | Description |
| --- | --- |
| `.extract_int(path)` | the integer value at JSON `path`; null if absent or non-integral (→ Int64) |
| `.extract_float(path)` | the numeric value at JSON `path` as a float; null if absent or non-numeric |
| `.extract_bool(path)` | the boolean value at JSON `path`; null if absent or non-boolean |
| `.array_length(path="$")` | number of elements in the array at `path`; null if it isn't an array (→ Int64) |
| `.keys(path="$")` | the object's keys at `path`, in source order (→ List<Utf8>) |
| `.values(path="$")` | the array's elements at `path`, each rendered as `extract_string` renders a leaf (→ List<Utf8>) |
| `.type_of(path="$")` | the JSON type at `path`: `object`, `array`, `string`, `number`, `boolean`, or `null` |
| `.exists(path)` | whether a value is present at `path`; a JSON `null` counts as present (→ Bool) |
| `.value(path)` | the **scalar** at `path` as its JSON token, null for an object or array (DuckDB `json_value`) |
| `.contains(value)` | whether the document holds `value` (itself JSON) as an element or field value (→ Bool) |
| `.pretty()` | the document re-rendered with two-space indentation (DuckDB `json_pretty`) |
| `.structure()` | the document's shape with each leaf replaced by its type name (DuckDB `json_structure`) |

`.value` and `.extract_string` answer different questions on purpose, as they do in
DuckDB: `.value` returns the JSON token of a *scalar* and null for a container, where
`.extract_string` unquotes a string and renders a container as compact JSON.
`.structure()` is the one to group by when you are finding out what shapes a JSON column
actually holds.

The shape methods inspect a document's *shape* rather than pulling one leaf out of it, which
is what a schema-on-read pipeline needs before it can extract anything. `type_of` routes
a field whose type varies row to row; `exists` separates an absent key from a key whose
value is JSON `null`, a distinction the `extract_*` methods can't express because both
come back as SQL NULL; and `values` turns a JSON array into a list column, so
{py:meth}`~batcher.Dataset.explode` and the whole `.list` namespace apply to it.

```python
docs = bt.from_pydict({"j": ['{"tags": ["a", "b"], "meta": null}']})
out = docs.select(
    n=bt.col("j").json.array_length("$.tags"),
    fields=bt.col("j").json.keys(),
    has_meta=bt.col("j").json.exists("$.meta"),
    kind=bt.col("j").json.type_of("$.tags"),
).to_pydict()
print(out)
# {'n': [2], 'fields': [['tags', 'meta']], 'has_meta': [True], 'kind': ['array']}
```

For retrieval / RAG, the vector ops score each row's embedding against a query
vector (a broadcast `array(...)` literal): `bt.col("emb").list.cosine_similarity(
bt.array(*[bt.lit(x) for x in query]))`.

```python
words = bt.from_pydict({"name": ["Ann", "bob"], "tags": [["x", "y"], ["z"]]})
out = words.select(
    upper=bt.col("name").str.upper(),
    n_tags=bt.col("tags").list.len(),
)
print(out.to_pydict())
# {'upper': ['ANN', 'BOB'], 'n_tags': [2, 1]}
```

## AI data-pipeline toolkit

Curating a training corpus, scrubbing PII, and budgeting context windows are all
per-row scans, so they belong in the engine rather than a Python loop. These score a
whole corpus in one vectorized pass.

**Corpus quality heuristics** on `.str` are the character-class ratios and shape statistics that Gopher, C4, and RefinedWeb-style filters threshold on to drop boilerplate and machine-generated text: `.alpha_ratio()`, `.digit_ratio()`, `.uppercase_ratio()`,
`.lowercase_ratio()`, `.punctuation_ratio()`, `.whitespace_ratio()`,
`.non_ascii_ratio()`, `.alnum_ratio()`, plus `.non_ascii_count()`, `.line_count()`,
`.mean_line_length()`, `.avg_word_length()`, `.sentence_count()`, `.url_count()`, and
`.email_count()`.

Document-shape signals: `.paragraph_count()`, `.is_single_line()`,
`.ends_with_punctuation()` (catches truncated crawls),
`.has_repeated_punctuation()`, `.quote_count()`, `.paren_count()`,
`.digit_to_word_ratio()`, and the code detectors `.code_fence_count()` /
`.looks_like_code()` (route code out of a prose corpus, or keep only code).

More corpus signals: `.uppercase_word_count()` (shouting/headers),
`.long_word_count(n)`, `.symbol_to_word_ratio()` (markup and ASCII art),
`.hashtag_count()` / `.mention_count()` (social-media provenance), and
`.phone_count()`.

**Cleaning and PII scrubbing**: `.remove_urls()`, `.remove_emails()`,
`.remove_phones()`, `.has_phone()`, and the shape-preserving `.mask_emails(token)` /
`.mask_urls(token)` (preferred over deletion for training data),
`.remove_non_ascii()`, `.remove_digits()`, `.remove_html_tags()`, and the budget guards
`.truncate_chars(n)` / `.truncate_words(n)` (which never cut mid-word).

**Detection predicates** for filtering: `.has_url()`, `.has_email()`,
`.has_non_ascii()`, `.has_digits()`, `.has_html()`, `.is_ascii_only()`, `.is_blank()`,
`.starts_with_bullet()`, and `.looks_like_json()` (a cheap shape check before decoding
LLM structured output).

Counts and shape predicates: `.newline_count()`, `.tab_count()`, `.space_count()`,
`.word_char_ratio()`, `.avg_sentence_length()`, `.is_short(n)` / `.is_long(n)`,
`.is_question()`, `.is_exclamation()`, `.starts_with_capital()`, `.is_all_caps()`,
`.has_currency()`, `.is_url()`, and `.is_email()` (whole-string forms, stricter than
`has_url`/`has_email`).

Extraction into `List<Utf8>`: `.extract_urls()`, `.extract_emails()`,
`.extract_numbers()`, `.extract_hashtags()`, `.extract_mentions()`, plus the scalar
`.first_sentence()`, `.first_word()`, and `.last_word()`.

Normalization for dedup keys and prose corpora: `.slugify()`, `.remove_bullets()`,
`.remove_repeated_punctuation()`, `.remove_markdown_links()`, `.remove_code_blocks()`,
`.remove_stopwords(words)`, and `.truncate_sentences(n)`.

**Token budgeting**: `.estimate_tokens(chars_per_token=4.0)` and
`.fits_token_budget(budget)` give the tokenizer-free estimate used to size context windows without paying to tokenize the corpus.

Embedding sanity and pooling on `.list`: `.dim()` (the embedding dimension),
`.is_zero_vector()` (the failed-encoder check), `.sum_squares()`, `.mean_pool()`, and
`.max_pool()`.

**Embedding helpers** on `.list`: `.magnitude()`, `.is_unit_norm(tol)` (assert the
normalization invariant held), `.euclidean_distance(o)`, and `.angular_distance(o)`. Angular distance is a true metric, unlike `1 - cosine`, and that is what nearest-neighbour indexes require.

```python
docs_ds = bt.from_pydict({"text": ["Real prose here, with sentences.", "AAA 111 &&& http://x.co"]})
scored = docs_ds.select(
    alpha=bt.col("text").str.alpha_ratio().round(3),
    toks=bt.col("text").str.estimate_tokens(),
    linky=bt.col("text").str.has_url(),
)
print(scored.to_pydict())
# {'alpha': [0.813, 0.435], 'toks': [8, 6], 'linky': [False, True]}
```

At the dataset level, `ds.shuffle(seed=)`, `ds.stratified_split(label, test_size)`
(preserves each class's proportion, value-hashed so it is identical distributed),
`ds.sample_per_group(by, n)`, `ds.class_balance(label)`, and `ds.class_weights(label)`
cover the train-set preparation steps.

## See also

- {doc}`/api/relational/expressions`: the core `Expr` surface these namespaces extend.
- {doc}`/user-guide/transform/expression-accessors`: the same namespaces taught with examples.
- {doc}`/user-guide/transform/expression-recipes`: text-corpus and feature recipes built on them.
- {doc}`/cookbook/expressions/index`: runnable recipes for the namespaces on this page.
