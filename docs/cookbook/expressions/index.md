# Expression cookbook

Recipes for the expression API: strings, temporal, lists, JSON, selectors, and the scalar algebra. Every one runs in Rust over whole columns rather than row by row.

Every page here embeds a complete, self-contained script from the
[`examples/expressions/`](https://github.com/batcher/batcher/tree/main/examples/expressions) directory.
The scripts build their own in-memory data and assert on their own output, and
`tests/docs/test_examples.py` runs all of them, so a page that stops matching the
engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`aggregates` | The aggregate vocabulary: counts, positions, quantiles, and approximations |
| {doc}`column_selectors` | Selectors: naming columns by type or pattern instead of one at a time |
| {doc}`conditionals` | Branching inside an expression: when/then/otherwise, and the SQL null helpers |
| {doc}`horizontal` | Horizontal functions: reducing across columns instead of down rows |
| {doc}`json_columns` | Reading JSON held in a string column, without parsing it in Python |
| {doc}`lists_aggregate` | Reducing a list column to one value per row |
| {doc}`lists_basics` | List columns: indexing, slicing, joining, and flattening |
| {doc}`lists_set_operations` | Treating two list columns as sets: union, intersection, difference, overlap |
| {doc}`lists_transforms` | Transforming inside a list column, without exploding it first |
| {doc}`lists_vectors` | Embedding vectors as list columns: similarity, distance, and normalization |
| {doc}`nulls_and_casting` | Nulls and type casting: the two places a pipeline quietly changes its answer |
| {doc}`numeric_math` | Arithmetic and math functions on numeric columns |
| {doc}`sorting_and_ranking` | Sorting and ranking, including the edge cases that hide bugs |
| {doc}`strings_case` | String case: normalizing capitalization before you compare or group |
| {doc}`strings_chunking` | Splitting long documents into overlapping chunks for a RAG index |
| {doc}`strings_cleaning` | Cleaning scraped text: strip markup, URLs, emails, and stray punctuation |
| {doc}`strings_counts` | Counting structure in text: words, lines, sentences, and entities |
| {doc}`strings_extraction` | Pulling entities and leading fragments out of free text |
| {doc}`strings_hashing` | Hashing and encoding a string column: keys, checksums, and safe transport |
| {doc}`strings_padding` | String padding and trimming: fixed-width keys and cleaning stray whitespace |
| {doc}`strings_paths` | Parsing file paths held in a column |
| {doc}`strings_predicates` | Boolean text predicates: the screen in front of an expensive stage |
| {doc}`strings_ratios` | Character-class ratios: cheap quality signals for a text corpus |
| {doc}`strings_regex` | Regular expressions over a column: extract, replace, and count |
| {doc}`strings_search` | String search: substring tests, multi-pattern tests, and match counting |
| {doc}`strings_similarity` | Fuzzy string matching against a reference value |
| {doc}`strings_slicing` | String slicing: taking a fixed piece of every value |
| {doc}`structs_and_maps` | Struct and map columns: nested records without flattening the table |
| {doc}`temporal_business_days` | Weekend and business-day predicates, and formatting a timestamp for output |
| {doc}`temporal_differences` | Durations between two timestamp columns, and shifting a timestamp |
| {doc}`temporal_parts` | Pulling calendar parts out of a timestamp column |
| {doc}`temporal_timezones` | Time zones: converting between them, and the reporting-boundary trap |
| {doc}`temporal_truncation` | Bucketing timestamps: truncate to a period, or snap to a period boundary |
| {doc}`window_functions` | Window functions: per-row values computed from a window of related rows |

```{toctree}
:hidden:

aggregates
column_selectors
conditionals
horizontal
json_columns
lists_aggregate
lists_basics
lists_set_operations
lists_transforms
lists_vectors
nulls_and_casting
numeric_math
sorting_and_ranking
strings_case
strings_chunking
strings_cleaning
strings_counts
strings_extraction
strings_hashing
strings_padding
strings_paths
strings_predicates
strings_ratios
strings_regex
strings_search
strings_similarity
strings_slicing
structs_and_maps
temporal_business_days
temporal_differences
temporal_parts
temporal_timezones
temporal_truncation
window_functions
```
