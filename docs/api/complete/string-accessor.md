# String accessor reference

This page is the generated reference for the `.str` accessor namespace, which you reach from any string expression as `col("x").str`. Each method has its own page, and the tables below group them by the job they do.

## The `.str` namespace

Every method returns a new lazy expression, and a null input gives a null output.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.strings

.. autoclass:: _StrNamespace
   :no-members:
```

### Case and trimming

Change a string's case, trim characters from its ends, and collapse its whitespace.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.strings

.. autosummary::
   :toctree: generated
   :nosignatures:

   _StrNamespace.upper
   _StrNamespace.lower
   _StrNamespace.capitalize
   _StrNamespace.to_titlecase
   _StrNamespace.to_case
   _StrNamespace.trim
   _StrNamespace.strip_chars_start
   _StrNamespace.strip_chars_end
   _StrNamespace.strip_prefix
   _StrNamespace.strip_suffix
   _StrNamespace.normalize_whitespace
```

### Search and match

Test a string for a literal substring or a SQL `LIKE` pattern, or find where one occurs.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.strings

.. autosummary::
   :toctree: generated
   :nosignatures:

   _StrNamespace.contains
   _StrNamespace.contains_any
   _StrNamespace.contains_all
   _StrNamespace.starts_with
   _StrNamespace.ends_with
   _StrNamespace.like
   _StrNamespace.ilike
   _StrNamespace.position
   _StrNamespace.count_char
```

### Slicing, padding, and replacement

Take part of a string, pad it to a width, or replace characters in it.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.strings

.. autosummary::
   :toctree: generated
   :nosignatures:

   _StrNamespace.substr
   _StrNamespace.slice
   _StrNamespace.left
   _StrNamespace.right
   _StrNamespace.substring_index
   _StrNamespace.overlay
   _StrNamespace.replace
   _StrNamespace.translate
   _StrNamespace.lpad
   _StrNamespace.rpad
   _StrNamespace.zfill
   _StrNamespace.repeat
   _StrNamespace.reverse
```

### Split, join, and paths

Split a string into a list, join values into one string, or take a path apart.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.strings

.. autosummary::
   :toctree: generated
   :nosignatures:

   _StrNamespace.split
   _StrNamespace.find_in_set
   _StrNamespace.split_part
   _StrNamespace.join
   _StrNamespace.parse_path
   _StrNamespace.parse_filename
   _StrNamespace.parse_dirpath
   _StrNamespace.parse_dirname
```

### Regular expressions

Match, extract, count, replace, and split with a regex pattern.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.strings

.. autosummary::
   :toctree: generated
   :nosignatures:

   _StrNamespace.regexp_matches
   _StrNamespace.match
   _StrNamespace.extract
   _StrNamespace.extract_all
   _StrNamespace.count_matches
   _StrNamespace.replace_all
   _StrNamespace.regexp_replace
   _StrNamespace.regexp_split
   _StrNamespace.escape_regex
```

### Length and character classes

Measure a string in characters, bytes, bits, or words, and test which characters it holds.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.strings

.. autosummary::
   :toctree: generated
   :nosignatures:

   _StrNamespace.len_chars
   _StrNamespace.octet_length
   _StrNamespace.bit_length
   _StrNamespace.word_count
   _StrNamespace.ascii
   _StrNamespace.is_alpha
   _StrNamespace.is_alnum
   _StrNamespace.is_numeric
   _StrNamespace.is_space
   _StrNamespace.is_upper
   _StrNamespace.is_lower
```

### Encoding, compression, and parsing

Encode and decode a string's bytes, compress them, sniff a payload's type, or parse a date from text.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.strings

.. autosummary::
   :toctree: generated
   :nosignatures:

   _StrNamespace.to_date
   _StrNamespace.to_datetime
   _StrNamespace.base64
   _StrNamespace.from_base64
   _StrNamespace.hex
   _StrNamespace.unhex
   _StrNamespace.to_binary
   _StrNamespace.from_binary
   _StrNamespace.url_encode
   _StrNamespace.url_decode
   _StrNamespace.compress
   _StrNamespace.decompress
   _StrNamespace.mime_type
```

### Similarity and phonetic matching

Score how close a string is to another, for fuzzy joins and deduplication.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.strings

.. autosummary::
   :toctree: generated
   :nosignatures:

   _StrNamespace.levenshtein
   _StrNamespace.damerau_levenshtein
   _StrNamespace.hamming
   _StrNamespace.jaro_similarity
   _StrNamespace.jaro_winkler_similarity
   _StrNamespace.jaccard
   _StrNamespace.soundex
```

### Hashing and fingerprints

Hash a string's bytes into a checksum, a digest, or a MinHash signature.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.strings

.. autosummary::
   :toctree: generated
   :nosignatures:

   _StrNamespace.hash64
   _StrNamespace.xxhash64
   _StrNamespace.crc32
   _StrNamespace.md5
   _StrNamespace.sha1
   _StrNamespace.sha256
   _StrNamespace.minhash
```

### Text quality ratios

Compute the character-class ratios and averages that corpus quality filters threshold on.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.strings

.. autosummary::
   :toctree: generated
   :nosignatures:

   _StrNamespace.alpha_ratio
   _StrNamespace.alnum_ratio
   _StrNamespace.digit_ratio
   _StrNamespace.uppercase_ratio
   _StrNamespace.lowercase_ratio
   _StrNamespace.punctuation_ratio
   _StrNamespace.whitespace_ratio
   _StrNamespace.non_ascii_ratio
   _StrNamespace.word_char_ratio
   _StrNamespace.alpha_word_ratio
   _StrNamespace.symbol_ratio
   _StrNamespace.symbol_to_word_ratio
   _StrNamespace.digit_to_word_ratio
   _StrNamespace.char_entropy
   _StrNamespace.avg_word_length
   _StrNamespace.mean_word_length
   _StrNamespace.avg_sentence_length
   _StrNamespace.mean_line_length
```

### Counts and repetition

Count a document's lines, sentences, and marks, and measure how much of it repeats.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.strings

.. autosummary::
   :toctree: generated
   :nosignatures:

   _StrNamespace.line_count
   _StrNamespace.paragraph_count
   _StrNamespace.sentence_count
   _StrNamespace.newline_count
   _StrNamespace.tab_count
   _StrNamespace.space_count
   _StrNamespace.digit_count
   _StrNamespace.non_ascii_count
   _StrNamespace.quote_count
   _StrNamespace.paren_count
   _StrNamespace.code_fence_count
   _StrNamespace.uppercase_word_count
   _StrNamespace.long_word_count
   _StrNamespace.stopword_count
   _StrNamespace.bullet_line_ratio
   _StrNamespace.ellipsis_line_ratio
   _StrNamespace.duplicate_line_ratio
   _StrNamespace.duplicate_paragraph_ratio
   _StrNamespace.duplicate_ngram_ratio
   _StrNamespace.top_ngram_ratio
```

### Detection predicates

Boolean checks on a document's shape and content, for use in a `filter`.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.strings

.. autosummary::
   :toctree: generated
   :nosignatures:

   _StrNamespace.is_blank
   _StrNamespace.is_short
   _StrNamespace.is_long
   _StrNamespace.is_single_line
   _StrNamespace.is_ascii_only
   _StrNamespace.has_non_ascii
   _StrNamespace.has_digits
   _StrNamespace.has_html
   _StrNamespace.has_currency
   _StrNamespace.has_repeated_punctuation
   _StrNamespace.is_question
   _StrNamespace.is_exclamation
   _StrNamespace.is_all_caps
   _StrNamespace.starts_with_capital
   _StrNamespace.starts_with_bullet
   _StrNamespace.ends_with_punctuation
   _StrNamespace.looks_like_code
   _StrNamespace.looks_like_json
```

### URLs, emails, and PII

Detect, count, extract, remove, or mask URLs, email addresses, phone numbers, hashtags, and mentions.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.strings

.. autosummary::
   :toctree: generated
   :nosignatures:

   _StrNamespace.has_url
   _StrNamespace.has_email
   _StrNamespace.has_phone
   _StrNamespace.is_url
   _StrNamespace.is_email
   _StrNamespace.url_count
   _StrNamespace.parse_url
   _StrNamespace.email_count
   _StrNamespace.phone_count
   _StrNamespace.hashtag_count
   _StrNamespace.mention_count
   _StrNamespace.extract_urls
   _StrNamespace.extract_emails
   _StrNamespace.extract_numbers
   _StrNamespace.extract_hashtags
   _StrNamespace.extract_mentions
   _StrNamespace.remove_urls
   _StrNamespace.remove_emails
   _StrNamespace.remove_phones
   _StrNamespace.mask_emails
   _StrNamespace.mask_urls
```

### Cleaning and normalization

Strip markup, symbols, and boilerplate from text, and normalize it into a comparison key.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.strings

.. autosummary::
   :toctree: generated
   :nosignatures:

   _StrNamespace.strip_html
   _StrNamespace.remove_html_tags
   _StrNamespace.remove_markdown_links
   _StrNamespace.remove_code_blocks
   _StrNamespace.remove_bullets
   _StrNamespace.remove_punctuation
   _StrNamespace.remove_repeated_punctuation
   _StrNamespace.remove_non_ascii
   _StrNamespace.remove_digits
   _StrNamespace.remove_stopwords
   _StrNamespace.slugify
   _StrNamespace.squad_normalize
   _StrNamespace.truncate_words
   _StrNamespace.truncate_sentences
```

### Tokens, chunks, and snippets

Estimate a token budget, split a document into chunks or n-grams, and take a short snippet.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.strings

.. autosummary::
   :toctree: generated
   :nosignatures:

   _StrNamespace.estimate_tokens
   _StrNamespace.fits_token_budget
   _StrNamespace.chunk
   _StrNamespace.token_ngrams
   _StrNamespace.first_sentence
   _StrNamespace.first_word
   _StrNamespace.last_word
```

## See also

- {doc}`/api/relational/expression-accessors`: every accessor method enumerated in one curated page.
- {doc}`expressions`: the `Expr` class these namespaces hang off.
- {doc}`/user-guide/transform/columns/string-accessor`: how to use the string accessor.
