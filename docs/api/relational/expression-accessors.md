# Expression accessors API

Breadth on {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` lives on accessor namespaces rather than on the expression itself, so
a hundred string functions sit behind {py:class}`.str <batcher.plan.expr_ir.namespaces.strings._StrNamespace>` instead of on every expression. This page
enumerates every method on every namespace.

The `Expr` methods these hang off are in {doc}`/api/relational/expressions`.

```python
import batcher as bt
```

## Accessor namespaces

Each namespace and the methods it carries:

| Namespace | Covers |
| --- | --- |
| `.str` | `upper`, `lower`, `trim(chars=None)`, `lstrip`/`rstrip(chars=None)`, `len`, `contains`, `starts_with`, `ends_with`, `like`, `ilike`, `substr`, `left`, `right`, `split`, `split_part(delim, n)`, `strip_html()` (markup → prose; drops `<script>`/`<style>` bodies and decodes entities), `chunk(size, overlap=0, boundary="char")` (RAG document splitter; `boundary` may be `"char"`/`"word"`/`"sentence"`/`"line"` so a chunk never ends mid-word), `token_ngrams(n)` (every window of `n` whitespace tokens, the unit BLEU and ROUGE-N are defined on), `squad_normalize()` (lowercase, drop standalone articles, delete punctuation, collapse whitespace, trim — the one tokenization every word-level metric agrees on, in a single pass), `minhash(num_perm=128, ngram=5)` (fuzzy-dedup signature), `replace`, `regexp_replace`, `regexp_replace_all`, `regexp_extract`, `initcap`, `hex`, `base64`, `translate`, `zfill(width)` (zero-pad numeric strings), `contains_any([...])` (true if any literal substring is present), and more |
| `.dt` | `year`, `month`, `day`, `hour`, `minute`, `second`, `quarter`, `week`, `dayofweek`, `dayofyear`, `dayname`, `monthname`, `epoch`, `epoch_ms()` / `epoch_us()` / `epoch_ns()` (integer epoch at ms/µs/ns resolution), `iso_year`, `is_leap_year`, `days_in_month`, `truncate(unit)` / `floor(unit)` / `ceil(unit)` / `round(unit)`, `time_of_day()` (microseconds since midnight) and `is_between_time('09:00', '17:00')` (a clock-time window, wrap-past-midnight aware), `strftime(fmt)`, `offset_by("1mo15d")`, `convert_timezone(from_tz, to_tz)` (DST-aware), `to_string(fmt)` (ISO-8601 by default), `timestamp(unit)` (the Polars epoch reader), `is_business_day()` (the weekday test, Polars' spelling of `is_weekday`), and more |
| `.list` | `len`, `sum`, `min`, `max`, `mean`, `median`, `std`, `var`, `product`, `n_unique`, `l2_norm`, `l1_norm` (Manhattan magnitude), `max_abs` (MaxAbs-scaling divisor), `normalize`, `softmax` (per-row logits→probabilities), `log_softmax` (the log-domain form, which stays finite where `softmax` underflows), `entropy` (per-row Shannon entropy in nats, the uncertainty of a score or probability vector), `arg_sort` (indices sorting ascending, so reverse for top-k), `cum_sum` (cumulative sum), `diff` (first difference with a leading null, for delta features), `sort`, `sort_desc` (descending, and *not* `sort().reverse()`: ascending puts nulls last, so reversing lifts them to the front, where DuckDB's `list_reverse_sort` keeps them at the back), `reverse`, `unique`, `flatten`, `get(i)` (negative ok), `first()`, `last()`, `slice`, `head(n)`, `contains(v)`, `position(v)`, `gather(indices)` (take each row's elements at the given positions — what makes `arg_sort` usable, so a rerank stays in the engine), `intersect(o)`, `difference(o)`, `union(o)`, `concat(o)` (append, keeping duplicates, as in DuckDB `list_concat` rather than a set op), `has_all(o)` / `has_any(o)` (containment tests, DuckDB `list_has_all`/`list_has_any`), `transform(element()-expr)`, `filter(element()-pred)`, `drop_nulls()` (the null-dropping filter, Polars `list.drop_nulls`), `join(sep)`, `add(o)` / `subtract(o)` / `multiply(o)` (element-wise vector arithmetic → List<Float64>); vector ops `dot(o)`, `cosine_similarity(o)`, `cosine_distance(o)`, `l2_distance(o)`, `l1_distance(o)` (Manhattan), `hamming_distance(o)` (differing positions, for binary or quantized embeddings), `jaccard(o)` (agreement rate; the MinHash/SimHash similarity estimate), `multiset_overlap(o)` (clipped bag intersection size, counting repeats — the BLEU/ROUGE-N numerator), `lcs_length(o)` (longest common *subsequence* length — the one overlap measure that reads order, and the ROUGE-L numerator; `O(n*m)` per row), `simhash(num_bits=64, seed=0)` (random-hyperplane LSH signature, the blocking key for a vector similarity join) |
| `.str` (document quality) | The LLM pretraining-corpus filters, per row: `word_count()`, `mean_word_length()` (Gopher keeps 3-10), `symbol_ratio()` (`#` and ellipses over words; drop above 0.1), `alpha_word_ratio()` (words containing a letter; drop below 0.8), `stopword_count()` (of Gopher's eight, counted distinctly; fewer than two is not prose), `bullet_line_ratio()` (drop above 0.9 — a navigation menu), `ellipsis_line_ratio()` (drop above 0.3 — a listing page of teasers), `duplicate_line_ratio()` / `duplicate_paragraph_ratio()` (weighed by *characters*, following Gopher), `top_ngram_ratio(n)` (the most frequent word n-gram's footprint; keyword-stuffed SEO), `duplicate_ngram_ratio(n)` (every repeated n-gram; boilerplate assembly), `char_entropy()` (the gibberish and base64-blob detector — not a Gopher rule, so choose its threshold from your own corpus). Each is null where the document has nothing to measure, so an empty extraction fails a threshold rather than sliding under it. The corpus-level *aggregates* of the same properties are in {doc}`/api/models/metrics`. |
| `.struct` | `field(name)`, `get(name)`, `keys()` |
| {py:class}`.json <batcher.plan.expr_ir.namespaces.collections._JsonNamespace>` | {py:meth}`extract_string(path) <batcher.plan.expr_ir.namespaces.collections._JsonNamespace.extract_string>` |
| `.map` | `get(key)`, `keys()`, `values()`, `entries()` (key/value structs, DuckDB `map_entries`), `len()` (entry count, DuckDB `cardinality`), `contains(key)` (DuckDB `map_contains`), which read a `Map`-typed column |
| `.image` | **Decode and tensors:** `decode()` (header-only struct `{width, height, channels, mode}` from a single read), `to_tensor(width, height)`, `to_tensor_f32(width, height, mean=, std=, channels_first=)` (model-ready `float32` tensor: scale to `[0,1]`, per-channel normalize, HWC/CHW), `center_crop(width, height)` (centered crop, torchvision-style zero-pad when smaller), `to_grayscale(width, height)` (decode+resize to a single Rec.601 luma channel). **Geometry and containers** — each takes `format=`/`quality=`, so a resize can stay JPEG instead of inflating a corpus into PNG: `resize(width, height)` (stretches to the exact size), `thumbnail(max_size)` (aspect-preserving downscale to a longest side, never upscaling — the one to use when `resize` would squash a mixed-orientation corpus), `letterbox(width, height, fill=114)` (aspect-preserving fit onto a padded canvas — the object-detection preprocessing, since stretching moves every predicted box and cropping throws away where the missed detections are), `pad(width, height, fill=0)` (centre on a canvas **without** scaling, so every surviving pixel keeps its exact value), `crop(x, y, width, height)` (an arbitrary window, clipped at the edge rather than padded; **each bound may be a column**, which is what makes cutting a detector's per-row bounding boxes out of their frames an engine operation), `rotate(degrees)` (a multiple of 90 — a free rotation would resample every pixel and pad the corners), `flip_horizontal()`, `flip_vertical()`, `encode(format)` (re-encode as `png`/`jpeg`/`bmp`/`gif`), `convert(mode)` (change color mode to `L`/`LA`/`RGB`/`RGBA`; Rec. 601 luma, the same weighting `to_grayscale` and `dhash` use), `auto_orient()` (apply the EXIF orientation a camera recorded instead of rotating its sensor data, so a portrait phone photo decodes upright rather than a quarter turn off). **Photometric adjustment**, on the `PIL.ImageEnhance` convention where `1.0` is the identity: `adjust_brightness(factor)`, `adjust_contrast(factor)`, `adjust_saturation(factor)`, `adjust_hue(degrees)`, `blur(sigma)`, `sharpen(amount)`, `invert()`, `posterize(bits)`, `solarize(threshold)`, `equalize()`, `autocontrast(cutoff)` (the AutoAugment/RandAugment primitives, plus the two histogram fixes that leave a flat image alone rather than turning it into noise). **Perceptual fingerprints**, all Int64 so `bitwise_xor(...).bit_count()` is a Hamming distance: `dhash()`, `phash()` (the DCT hash — the most robust to rescaling and re-encoding), `ahash()` (the cheapest pre-filter). **Curation measures and header facts:** `brightness()` (mean luma in `[0, 1]`, the blank-tile detector), `sharpness()` (normalized Laplacian variance, the focus measure that finds a corpus's blurred tail), `entropy()` (luma-histogram Shannon entropy in bits, which separates a mid-grey placeholder from a foggy photograph where brightness cannot), `colorfulness()` (Hasler-Süsstrunk, which finds the sepia and line-art rows no luma measure can see), `mean_color()` (struct `{r, g, b}`), `is_grayscale()` (three identical channels, the fact no header carries), `exif_orientation()` (the EXIF code 1-8, so you can measure how much of a corpus needs orienting), `aspect_ratio()`, `has_alpha()`, `format()` (the container sniffed from the magic bytes, not the file extension — how a corpus of `.jpg` files that are really PNGs gets found; the name is the one `encode()` accepts, so it round-trips) |
| `.audio` | **Decode and resample:** `decode()`, `to_waveform()` (decode to a mono PCM `List<Float>` signal), `resample(rate)` (decode + band-limited resample to `rate` Hz, the 16 kHz audio-ML preprocessing step). **Level and hygiene measures:** `rms()` (root-mean-square amplitude, which tracks loudness where a peak does not), `dbfs()` and `peak_dbfs()` (the same in decibels below full scale; null rather than `-inf` for digital silence, so a threshold cannot silently accept it), `clipping_ratio(threshold=0.99)` (distortion no normalization can undo), `silence_ratio(threshold_db=-40)` (how much of the clip is dead air), `zero_crossing_rate()` (the voiced/unvoiced descriptor, in `[0, 1]`). **Waveform shaping** — these and the level measures also accept a *waveform* column, so they chain without re-decoding: `trim_silence(threshold_db=-40)` (drop the leading and trailing quiet; a clip silent throughout trims to an empty list, which is how a silent recording is filtered out), `peak_normalize()` (scale so the loudest sample sits at full scale), `rms_normalize(target_db=-20)` (loudness match, usually the one you want — peak normalization leaves a clip with one loud click quiet everywhere else), `pre_emphasis(coefficient=0.97)` (the first-order high-pass every classical ASR front end runs before framing), `pad_or_trim(duration_secs, rate)` (force every row to the same length — Whisper's fixed 30 seconds, and what makes a clip corpus batchable at all), `slice(offset_secs, duration_secs)` (a region, with a window past the end giving an empty list rather than null), `encode_wav(rate=None)` (a mono 16-bit PCM WAV container, so a cleaned corpus can be written back out as audio). **Spectral front ends and descriptors:** `mel_spectrogram(rate, n_fft=400, hop_length=160, n_mels=80)` (matches `torchaudio.transforms.MelSpectrogram`), `mfcc(rate, n_fft=400, hop_length=160, n_mels=128, n_mfcc=40)` (matches `torchaudio.transforms.MFCC`), `spectrogram(rate, n_fft=, hop_length=)` (the **linear** power spectrogram, for the music and bioacoustic models a mel warp throws frequencies away for), `spectral_centroid(rate)` (brightness), `spectral_rolloff(rate, percentile=0.85)` (where the usable band ends — how an 8 kHz recording upsampled to 16 kHz is caught), `spectral_bandwidth(rate)`, `spectral_flatness(rate)` (tonality: near 0 for a tone, near 1 for noise) |
| `.video` | `decode()` (header-only struct `{width, height, num_frames, duration_secs, fps}`), `frames(num_frames, width, height)` (evenly-spaced frames as one fixed-shape `(num_frames, height, width, 3)` uint8 tensor — the video training-ingest kernel), `thumbnail(max_size)` (the clip's middle frame as PNG bytes, aspect preserved), `frame_at(second, max_size)` (the frame shown at `second`, as PNG bytes; **`second` may be a column**, so a row that already carries a timestamp can name the moment it wants) |
| `.seq` | `complement()`, `reverse_complement()` (IUPAC-correct and case-preserving, so soft-masked repeats survive), `transcribe()` / `back_transcribe()` (DNA↔RNA), `gc_content()` (the G+C fraction of the *unambiguous* bases, so a run of `N` does not read as AT-rich), `gc_skew()` (the replication-strand signal), `base_counts()` (struct `{a, c, g, t, u, n, other}` from one pass), `max_homopolymer()` (the nanopore/PacBio error signature), `is_valid(alphabet)` (`dna`/`rna`/`dna_iupac`/`rna_iupac`/`protein`), `translate(frame=0, to_stop=False)` (NCBI genetic code 1; ambiguous codons → `X`, a trailing partial codon dropped rather than padded), `kmers(k)`, `canonical_kmers(k)` (folded with the reverse complement, so both strands agree — the Jellyfish/KMC rule), `minimizers(k, window)` (the seed-and-extend sketch; two sequences sharing `window + k - 1` bases are guaranteed to share one), `melting_temp()` (SantaLucia 1998 nearest-neighbour at 50 mM Na⁺ / 500 nM strand), `molecular_weight(alphabet)`, `gravy()` (Kyte-Doolittle hydropathy), `isoelectric_point()`, `phred_quality(offset=33)` / `mean_quality(offset=33)` / `expected_errors(offset=33)` (FASTQ decoding; `expected_errors` is the `fastq_maxee` filter and is additive where a mean is not), `find_motif(motif)` / `count_motif(motif)` (IUPAC-degenerate, overlapping, 1-based positions) |

### More `.str` methods

The table above lists the common string operations. These are the rest, covering padding,
casing, encoding, and the similarity measures:

| Method | Description |
| --- | --- |
| `.lpad(width, fill=" ")` / `.rpad(width, fill=" ")` | pad to `width` characters with `fill` (cycled); truncate if longer |
| {py:meth}`.repeat(n) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.repeat>` | repeat the string `n` times (`n` ≤ 0 → empty) |
| {py:meth}`.normalize_whitespace() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.normalize_whitespace>` | collapse every run of whitespace to a single space and trim the ends |
| `.position(pattern)` | 1-based index of `pattern`, 0 if absent (→ Int64) |
| {py:meth}`.regexp_matches(pattern) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.regexp_matches>` | true where the regex matches anywhere (→ Bool) |
| {py:meth}`.ascii() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.ascii>` | Unicode codepoint of the first character, 0 if empty (→ Int64) |
| {py:meth}`.bit_length() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.bit_length>` / {py:meth}`.octet_length() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.octet_length>` | number of bits / UTF-8 bytes in the string (→ Int64) |
| {py:meth}`.from_base64() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.from_base64>` | decode standard base64 to a UTF-8 string; null if invalid |
| {py:meth}`.unhex() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.unhex>` | decode pairs of hex digits to a UTF-8 string; null if invalid |
| {py:meth}`.md5() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.md5>` / {py:meth}`.sha1() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.sha1>` / {py:meth}`.sha256() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.sha256>` | cryptographic digest as lowercase hex; null → null |
| `.crc32()` | CRC-32 (IEEE) checksum of the UTF-8 bytes (Spark `crc32`, → Int64) |
| `.mime_type()` | what the leading bytes say the payload is (`image/png`, `video/mp4`, `application/pdf`, …), null when unrecognized — the way to identify bytes that never came from a file read, such as a `ds.ml.download` or a blob column |
| {py:meth}`.hash64() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.hash64>` | deterministic FNV-1a 64-bit hash, stable across partitions and machines, so it's a surrogate-key building block (→ Int64) |
| {py:meth}`.xxhash64() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.xxhash64>` | fast non-cryptographic 64-bit xxHash; the standard bucketing/sharding hash (→ Int64) |
| `.substring_index(delimiter, count)` | substring before the `count`-th `delimiter` (Spark) |
| {py:meth}`.overlay(replacement, pos, length=None) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.overlay>` | replace `length` chars from 1-based `pos` (SQL `OVERLAY`) |
| {py:meth}`.regexp_extract_all(pattern) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.regexp_extract_all>` | every regex match as a `List<Utf8>` (DuckDB {py:meth}`regexp_extract_all <batcher.plan.expr_ir.namespaces.strings._StrNamespace.regexp_extract_all>`) |
| {py:meth}`.regexp_count(pattern) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.regexp_count>` | number of non-overlapping regex matches (→ Int64) |
| {py:meth}`.regexp_split(pattern) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.regexp_split>` | split on every regex match into a `List<Utf8>` (DuckDB `regexp_split_to_array`) |
| `.levenshtein(target)` | edit distance to the constant `target` (DuckDB `levenshtein`, → Int64) |
| {py:meth}`.damerau_levenshtein(target) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.damerau_levenshtein>` | edit distance to `target` counting an adjacent-swap as one edit (DuckDB {py:meth}`damerau_levenshtein <batcher.plan.expr_ir.namespaces.strings._StrNamespace.damerau_levenshtein>`, → Int64), which handles typos better |
| {py:meth}`.jaro_similarity(target) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.jaro_similarity>` | Jaro similarity to `target`, `[0,1]` (DuckDB {py:meth}`jaro_similarity <batcher.plan.expr_ir.namespaces.strings._StrNamespace.jaro_similarity>`, → Float64), for fuzzy matching and record linkage |
| {py:meth}`.jaro_winkler_similarity(target) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.jaro_winkler_similarity>` | Jaro-Winkler similarity to `target`, `[0,1]` (DuckDB {py:meth}`jaro_winkler_similarity <batcher.plan.expr_ir.namespaces.strings._StrNamespace.jaro_winkler_similarity>`, → Float64), prefix-weighted for name matching |
| {py:meth}`.soundex() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.soundex>` | American Soundex phonetic code, a 4-character key (→ Utf8) |
| `.hamming(target)` | positions at which the value and `target` differ (DuckDB `hamming`/`mismatches`, → Int64); the lengths must be equal |
| `.jaccard(target)` | Jaccard similarity of the two values' character *sets*, `[0,1]` (DuckDB `jaccard`, → Float64) |
| {py:meth}`.url_encode() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.url_encode>` / {py:meth}`.url_decode() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.url_decode>` | percent-encode a URL *component* and its inverse (DuckDB {py:meth}`url_encode <batcher.plan.expr_ir.namespaces.strings._StrNamespace.url_encode>`/{py:meth}`url_decode <batcher.plan.expr_ir.namespaces.strings._StrNamespace.url_decode>`) |
| {py:meth}`.regexp_escape() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.regexp_escape>` | escape the regex metacharacters, so a value can be embedded in a pattern as a literal (DuckDB {py:meth}`regexp_escape <batcher.plan.expr_ir.namespaces.strings._StrNamespace.regexp_escape>`) |
| {py:meth}`.escape_regex() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.escape_regex>` | the Polars spelling of {py:meth}`regexp_escape <batcher.plan.expr_ir.namespaces.strings._StrNamespace.regexp_escape>` |
| `.join(sep)` | concatenate every value of the group into one string, as an aggregate (SQL `string_agg`, Polars `str.join`) |
| {py:meth}`.parse_filename() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.parse_filename>` | the final component of a path (DuckDB {py:meth}`parse_filename <batcher.plan.expr_ir.namespaces.strings._StrNamespace.parse_filename>`) |
| {py:meth}`.parse_dirname() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.parse_dirname>` | the *first* component of a path, which is `/` for an absolute one (DuckDB {py:meth}`parse_dirname <batcher.plan.expr_ir.namespaces.strings._StrNamespace.parse_dirname>`) |
| {py:meth}`.parse_dirpath() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.parse_dirpath>` | everything before the last separator, i.e. the containing directory (DuckDB {py:meth}`parse_dirpath <batcher.plan.expr_ir.namespaces.strings._StrNamespace.parse_dirpath>`) |
| {py:meth}`.parse_path() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.parse_path>` | the path's components as a `List<Utf8>`, keeping a leading `/` (DuckDB {py:meth}`parse_path <batcher.plan.expr_ir.namespaces.strings._StrNamespace.parse_path>`) |
| {py:meth}`.to_binary() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.to_binary>` / {py:meth}`.from_binary() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.from_binary>` | the UTF-8 bytes as `0`/`1` text and back; undecodable input is null (DuckDB {py:meth}`to_binary <batcher.plan.expr_ir.namespaces.strings._StrNamespace.to_binary>`/{py:meth}`from_binary <batcher.plan.expr_ir.namespaces.strings._StrNamespace.from_binary>`) |
| {py:meth}`.to_date(format="%Y-%m-%d") <batcher.plan.expr_ir.namespaces.strings._StrNamespace.to_date>` | parse into a Date with a strftime format; unmatched → NULL (→ Date32) |
| {py:meth}`.to_datetime(format) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.to_datetime>` | parse into a Timestamp (DuckDB `try_strptime`); unmatched → NULL (→ Timestamp(us)) |
| `.to_case(style)` | re-case an identifier into `style`, one of `snake`, `upper_snake`, `camel`, `pascal`, `kebab`, `upper_kebab`, `title`, `sentence`, `dot`, or `train` |
| {py:meth}`.compress(codec) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.compress>` | compress the raw bytes with `gzip`, `zlib`, `deflate`, `zstd`, `brotli`, or `lz4` (→ Binary) |
| {py:meth}`.decompress(codec) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.decompress>` | the inverse; a frame that isn't valid for `codec` is null, not an error |

### More `.dt` methods

{py:meth}`.century() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.century>`, {py:meth}`.decade() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.decade>`, {py:meth}`.isodow() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.isodow>` (ISO day of week), {py:meth}`.last_day() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.last_day>` (last day
of the month), and {py:meth}`.millennium() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.millennium>`. Each extracts the named field of a date/time column (→ Int64).

### More `.json` methods

These extract a typed value at a JSON path, or describe the document's shape when you do
not yet know it:

| Method | Description |
| --- | --- |
| {py:meth}`.extract_int(path) <batcher.plan.expr_ir.namespaces.collections._JsonNamespace.extract_int>` | the integer value at JSON `path`; null if absent or non-integral (→ Int64) |
| {py:meth}`.extract_float(path) <batcher.plan.expr_ir.namespaces.collections._JsonNamespace.extract_float>` | the numeric value at JSON `path` as a float; null if absent or non-numeric |
| {py:meth}`.extract_bool(path) <batcher.plan.expr_ir.namespaces.collections._JsonNamespace.extract_bool>` | the boolean value at JSON `path`; null if absent or non-boolean |
| {py:meth}`.array_length(path="$") <batcher.plan.expr_ir.namespaces.collections._JsonNamespace.array_length>` | number of elements in the array at `path`; null if it isn't an array (→ Int64) |
| `.keys(path="$")` | the object's keys at `path`, in source order (→ List<Utf8>) |
| `.values(path="$")` | the array's elements at `path`, each rendered as `extract_string` renders a leaf (→ List<Utf8>) |
| `.type_of(path="$")` | the JSON type at `path`: `object`, `array`, `string`, `number`, `boolean`, or `null` |
| {py:meth}`.exists(path) <batcher.plan.expr_ir.namespaces.collections._JsonNamespace.exists>` | whether a value is present at `path`; a JSON `null` counts as present (→ Bool) |
| {py:meth}`.value(path) <batcher.plan.expr_ir.namespaces.collections._JsonNamespace.value>` | the **scalar** at `path` as its JSON token, null for an object or array (DuckDB `json_value`) |
| `.contains(value)` | whether the document holds `value` (itself JSON) as an element or field value (→ Bool) |
| {py:meth}`.pretty() <batcher.plan.expr_ir.namespaces.collections._JsonNamespace.pretty>` | the document re-rendered with two-space indentation (DuckDB `json_pretty`) |
| {py:meth}`.structure() <batcher.plan.expr_ir.namespaces.collections._JsonNamespace.structure>` | the document's shape with each leaf replaced by its type name (DuckDB `json_structure`) |

`.value` and `.extract_string` answer different questions on purpose, as they do in
DuckDB: `.value` returns the JSON token of a *scalar* and null for a container, where
{py:meth}`.extract_string <batcher.plan.expr_ir.namespaces.collections._JsonNamespace.extract_string>` unquotes a string and renders a container as compact JSON.
`.structure()` is the one to group by when you are finding out what shapes a JSON column
actually holds.

The shape methods inspect a document's *shape* rather than pulling one leaf out of it, which
is what a schema-on-read pipeline needs before it can extract anything. {py:meth}`type_of <batcher.plan.expr_ir.namespaces.collections._JsonNamespace.type_of>` routes
a field whose type varies row to row; `exists` separates an absent key from a key whose
value is JSON `null`, a distinction the `extract_*` methods can't express because both
come back as SQL NULL; and `values` turns a JSON array into a list column, so
{py:meth}`~batcher.Dataset.explode` and the whole {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>` namespace apply to it.

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
vector (a broadcast {py:func}`array(...) <batcher.array>` literal): `bt.col("emb").list.cosine_similarity(
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

**Corpus quality heuristics** on {py:class}`.str <batcher.plan.expr_ir.namespaces.strings._StrNamespace>` are the character-class ratios and shape statistics that Gopher, C4, and RefinedWeb-style filters threshold on to drop boilerplate and machine-generated text: {py:meth}`.alpha_ratio() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.alpha_ratio>`, {py:meth}`.digit_ratio() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.digit_ratio>`, {py:meth}`.uppercase_ratio() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.uppercase_ratio>`,
{py:meth}`.lowercase_ratio() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.lowercase_ratio>`, {py:meth}`.punctuation_ratio() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.punctuation_ratio>`, {py:meth}`.whitespace_ratio() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.whitespace_ratio>`,
{py:meth}`.non_ascii_ratio() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.non_ascii_ratio>`, {py:meth}`.alnum_ratio() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.alnum_ratio>`, plus {py:meth}`.non_ascii_count() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.non_ascii_count>`, {py:meth}`.line_count() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.line_count>`,
{py:meth}`.mean_line_length() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.mean_line_length>`, {py:meth}`.avg_word_length() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.avg_word_length>`, {py:meth}`.sentence_count() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.sentence_count>`, {py:meth}`.url_count() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.url_count>`, and
{py:meth}`.email_count() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.email_count>`.

Document-shape signals: {py:meth}`.paragraph_count() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.paragraph_count>`, {py:meth}`.is_single_line() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_single_line>`,
{py:meth}`.ends_with_punctuation() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.ends_with_punctuation>` (catches truncated crawls),
{py:meth}`.has_repeated_punctuation() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.has_repeated_punctuation>`, {py:meth}`.quote_count() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.quote_count>`, {py:meth}`.paren_count() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.paren_count>`,
{py:meth}`.digit_to_word_ratio() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.digit_to_word_ratio>`, and the code detectors {py:meth}`.code_fence_count() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.code_fence_count>` /
{py:meth}`.looks_like_code() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.looks_like_code>` (route code out of a prose corpus, or keep only code).

More corpus signals: {py:meth}`.uppercase_word_count() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.uppercase_word_count>` (shouting/headers),
{py:meth}`.long_word_count(n) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.long_word_count>`, {py:meth}`.symbol_to_word_ratio() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.symbol_to_word_ratio>` (markup and ASCII art),
{py:meth}`.hashtag_count() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.hashtag_count>` / {py:meth}`.mention_count() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.mention_count>` (social-media provenance), and
{py:meth}`.phone_count() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.phone_count>`.

**Cleaning and PII scrubbing**: {py:meth}`.remove_urls() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_urls>`, {py:meth}`.remove_emails() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_emails>`,
{py:meth}`.remove_phones() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_phones>`, {py:meth}`.has_phone() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.has_phone>`, and the shape-preserving {py:meth}`.mask_emails(token) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.mask_emails>` /
{py:meth}`.mask_urls(token) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.mask_urls>` (preferred over deletion for training data),
{py:meth}`.remove_non_ascii() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_non_ascii>`, {py:meth}`.remove_digits() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_digits>`, {py:meth}`.remove_html_tags() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_html_tags>`, and the budget guards
{py:meth}`.truncate_chars(n) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.truncate_chars>` / {py:meth}`.truncate_words(n) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.truncate_words>` (which never cut mid-word).

**Detection predicates** for filtering: {py:meth}`.has_url() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.has_url>`, {py:meth}`.has_email() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.has_email>`,
{py:meth}`.has_non_ascii() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.has_non_ascii>`, {py:meth}`.has_digits() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.has_digits>`, {py:meth}`.has_html() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.has_html>`, {py:meth}`.is_ascii_only() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_ascii_only>`, {py:meth}`.is_blank() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_blank>`,
{py:meth}`.starts_with_bullet() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.starts_with_bullet>`, and {py:meth}`.looks_like_json() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.looks_like_json>` (a cheap shape check before decoding
LLM structured output).

Counts and shape predicates: {py:meth}`.newline_count() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.newline_count>`, {py:meth}`.tab_count() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.tab_count>`, {py:meth}`.space_count() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.space_count>`,
{py:meth}`.word_char_ratio() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.word_char_ratio>`, {py:meth}`.avg_sentence_length() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.avg_sentence_length>`, {py:meth}`.is_short(n) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_short>` / {py:meth}`.is_long(n) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_long>`,
{py:meth}`.is_question() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_question>`, {py:meth}`.is_exclamation() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_exclamation>`, {py:meth}`.starts_with_capital() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.starts_with_capital>`, {py:meth}`.is_all_caps() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_all_caps>`,
{py:meth}`.has_currency() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.has_currency>`, {py:meth}`.is_url() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_url>`, and {py:meth}`.is_email() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_email>` (whole-string forms, stricter than
{py:meth}`has_url <batcher.plan.expr_ir.namespaces.strings._StrNamespace.has_url>`/{py:meth}`has_email <batcher.plan.expr_ir.namespaces.strings._StrNamespace.has_email>`).

Extraction into `List<Utf8>`: {py:meth}`.extract_urls() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.extract_urls>`, {py:meth}`.extract_emails() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.extract_emails>`,
{py:meth}`.extract_numbers() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.extract_numbers>`, {py:meth}`.extract_hashtags() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.extract_hashtags>`, {py:meth}`.extract_mentions() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.extract_mentions>`, plus the scalar
{py:meth}`.first_sentence() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.first_sentence>`, {py:meth}`.first_word() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.first_word>`, and {py:meth}`.last_word() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.last_word>`.

Normalization for dedup keys and prose corpora: {py:meth}`.slugify() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.slugify>`, {py:meth}`.remove_bullets() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_bullets>`,
{py:meth}`.remove_repeated_punctuation() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_repeated_punctuation>`, {py:meth}`.remove_markdown_links() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_markdown_links>`, {py:meth}`.remove_code_blocks() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_code_blocks>`,
{py:meth}`.remove_stopwords(words) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.remove_stopwords>`, and {py:meth}`.truncate_sentences(n) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.truncate_sentences>`.

**Token budgeting**: {py:meth}`.estimate_tokens(chars_per_token=4.0) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.estimate_tokens>` and
{py:meth}`.fits_token_budget(budget) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.fits_token_budget>` give the tokenizer-free estimate used to size context windows without paying to tokenize the corpus.

Embedding sanity and pooling on {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>`: {py:meth}`.dim() <batcher.plan.expr_ir.namespaces.collections._ListNamespace.dim>` (the embedding dimension),
{py:meth}`.is_zero_vector() <batcher.plan.expr_ir.namespaces.collections._ListNamespace.is_zero_vector>` (the failed-encoder check), {py:meth}`.sum_squares() <batcher.plan.expr_ir.namespaces.collections._ListNamespace.sum_squares>`, {py:meth}`.mean_pool() <batcher.plan.expr_ir.namespaces.collections._ListNamespace.mean_pool>`, and
{py:meth}`.max_pool() <batcher.plan.expr_ir.namespaces.collections._ListNamespace.max_pool>`.

**Embedding helpers** on {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>`: {py:meth}`.magnitude() <batcher.plan.expr_ir.namespaces.collections._ListNamespace.magnitude>`, {py:meth}`.is_unit_norm(tol) <batcher.plan.expr_ir.namespaces.collections._ListNamespace.is_unit_norm>` (assert the
normalization invariant held), {py:meth}`.euclidean_distance(o) <batcher.plan.expr_ir.namespaces.collections._ListNamespace.euclidean_distance>`, and {py:meth}`.angular_distance(o) <batcher.plan.expr_ir.namespaces.collections._ListNamespace.angular_distance>`. Angular distance is a true metric, unlike `1 - cosine`, and that is what nearest-neighbour indexes require.

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

At the dataset level, {py:meth}`ds.shuffle(seed=) <batcher.Dataset.shuffle>`, {py:meth}`ds.stratified_split(label, test_size) <batcher.Dataset.stratified_split>`
(preserves each class's proportion, value-hashed so it is identical distributed),
{py:meth}`ds.sample_per_group(by, n) <batcher.Dataset.sample_per_group>`, {py:meth}`ds.class_balance(label) <batcher.Dataset.class_balance>`, and {py:meth}`ds.class_weights(label) <batcher.Dataset.class_weights>`
cover the train-set preparation steps.

## See also

- {doc}`/api/relational/expressions`: the core `Expr` surface these namespaces extend.
- {doc}`/user-guide/transform/columns/expression-accessors`: the same namespaces taught with examples.
- {doc}`/user-guide/transform/columns/expression-recipes`: text-corpus and feature recipes built on them.
- {doc}`/cookbook/expressions/index`: runnable recipes for the namespaces on this page.
