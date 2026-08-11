# Expressions

This page covers the scripts that exercise the expression language: the operators, the
accessor namespaces, and the type rules that decide what an expression returns.

## The accessor namespaces

Breadth lives on accessors rather than on `Expr` itself, so the fluent builder stays thin.
`.str`, `.dt`, `.list`, `.struct`, `.map` and `.json` each carry their family.

```python
import batcher as bt
from batcher import col

logs = bt.from_pydict(
    {
        "line": [
            "GET /users/42 200 13ms",
            "POST /orders/8812 500 220ms",
        ]
    }
)

parsed = logs.select(
    verb=col("line").str.extract(r"^([A-Z]+) "),
    status=col("line").str.extract(r" (\d{3}) ").cast("int64"),
    anonymized=col("line").str.replace_all(r"/\d+", "/<id>"),
)
result = parsed.to_pydict()

assert result["verb"] == ["GET", "POST"]
assert result["status"] == [200, 500]
assert result["anonymized"][0] == "GET /users/<id> 200 13ms"
```

Two spellings in the string family behave differently from their Python counterparts, and
both have their own script. `strip` removes spaces rather than all whitespace, so a leading
tab survives it; pass the character set to `strip_chars` when you mean all of it. And
`levenshtein` and the other comparison functions take a plan-time constant rather than
another column, because the target is lowered into the plan and compiled once.

## Conditionals and nulls

A CASE builder needs a terminating `otherwise`. An unfinished one raises rather than
implying a null, which catches the most common way these go wrong.

```python
orders = bt.from_pydict({"total": [10_000.0, 90_000.0, 400_000.0]})

banded = orders.with_columns(
    band=bt.when(col("total") < 50_000)
    .then(bt.lit("small"))
    .when(col("total") < 150_000)
    .then(bt.lit("medium"))
    .otherwise(bt.lit("large"))
)
assert banded.to_pydict()["band"] == ["small", "medium", "large"]
```

There is no null literal to reach for. `bt.nullif(a, b)` is the expression that produces one,
returning null where the two sides are equal, and `bt.coalesce` is how you consume nulls by
supplying a fallback chain.

## Selectors and horizontal folds

A selector resolves against the schema at plan time, which is what makes a generic cleanup
step possible without reflection in Python. The `*_horizontal` family is the row-wise
counterpart to an aggregate, for when a value is spread across columns rather than rows.

```python
readings = bt.from_pydict(
    {"sensor": ["a", "b"], "morning": [1.0, 3.0], "evening": [2.0, 4.0]}
)

folded = readings.select(
    "sensor",
    total=bt.sum_horizontal(col("morning"), col("evening")),
    peak=bt.max_horizontal(col("morning"), col("evening")),
)
assert folded.to_pydict()["total"] == [3.0, 7.0]
assert folded.to_pydict()["peak"] == [2.0, 4.0]

numeric = readings.select(bt.numeric())
assert numeric.columns == ["morning", "evening"]
```

`count_horizontal` counts non-null arguments rather than true ones, mirroring
`col(x).count()`. To count satisfied predicates, cast them to integers and sum.

## Types

Mixed arithmetic widens to the type that can hold both. True division always widens and
floor division does not, which is the difference that turns a count into a fraction without
anyone noticing.

```python
counts = bt.from_pydict({"hits": [10, 5], "total": [10, 20]})

typed = counts.select(
    exact=col("hits") / col("total"),
    floored=col("hits") // col("total"),
)
types = {name: str(dtype) for name, dtype in zip(typed.columns, typed.dtypes)}
assert types["exact"] == "double"
assert types["floored"] == "int64"
```

A cast to an integer rounds to nearest rather than truncating, which is the opposite of both
C-style casting and Python's `int()`. Call `floor()` before the cast when you mean to
truncate.

## Every script on this page

The table below lists the expression scripts in path order.

<!-- library-table: expressions,expr_text,expr_numeric,expr_temporal,expr_collections,expr_logic,expr_vectors -->
| Script | Shows |
| --- | --- |
| `examples/expressions/aggregates.py` | The aggregate vocabulary: counts, positions, quantiles, and approximations |
| `examples/expressions/column_selectors.py` | Selectors: naming columns by type or pattern instead of one at a time |
| `examples/expressions/conditionals.py` | Branching inside an expression: when/then/otherwise, and the SQL null helpers |
| `examples/expressions/genomics_assembly_stats.py` | Judging genome assemblies: N50, N90, L50, and auN as mergeable aggregates |
| `examples/expressions/genomics_files.py` | FASTA and FASTQ files as tables: read, measure, filter, write |
| `examples/expressions/genomics_intervals.py` | Intervals, annotations, and variants: BED, GFF, and VCF as joinable tables |
| `examples/expressions/genomics_reads.py` | Sequencing reads as a table: quality filtering, k-mer sketching, and primer design |
| `examples/expressions/genomics_sequences.py` | Nucleotide sequences as a column: complementing, measuring, translating, and searching |
| `examples/expressions/horizontal.py` | Horizontal functions: reducing across columns instead of down rows |
| `examples/expressions/json_columns.py` | Reading JSON held in a string column, without parsing it in Python |
| `examples/expressions/lists_aggregate.py` | Reducing a list column to one value per row |
| `examples/expressions/lists_basics.py` | List columns: indexing, slicing, joining, and flattening |
| `examples/expressions/lists_set_operations.py` | Treating two list columns as sets: union, intersection, difference, overlap |
| `examples/expressions/lists_transforms.py` | Transforming inside a list column, without exploding it first |
| `examples/expressions/lists_vectors.py` | Embedding vectors as list columns: similarity, distance, and normalization |
| `examples/expressions/nulls_and_casting.py` | Nulls and type casting: the two places a pipeline quietly changes its answer |
| `examples/expressions/numeric_math.py` | Arithmetic and math functions on numeric columns |
| `examples/expressions/sorting_and_ranking.py` | Sorting and ranking, including the edge cases that hide bugs |
| `examples/expressions/strings_case.py` | String case: normalizing capitalization before you compare or group |
| `examples/expressions/strings_chunking.py` | Splitting long documents into overlapping chunks for a RAG index |
| `examples/expressions/strings_cleaning.py` | Cleaning scraped text: strip markup, URLs, emails, and stray punctuation |
| `examples/expressions/strings_counts.py` | Counting structure in text: words, lines, sentences, and entities |
| `examples/expressions/strings_extraction.py` | Pulling entities and leading fragments out of free text |
| `examples/expressions/strings_hashing.py` | Hashing and encoding a string column: keys, checksums, and safe transport |
| `examples/expressions/strings_padding.py` | String padding and trimming: fixed-width keys and cleaning stray whitespace |
| `examples/expressions/strings_paths.py` | Parsing file paths held in a column |
| `examples/expressions/strings_predicates.py` | Boolean text predicates: the screen in front of an expensive stage |
| `examples/expressions/strings_ratios.py` | Character-class ratios: cheap quality signals for a text corpus |
| `examples/expressions/strings_regex.py` | Regular expressions over a column: extract, replace, and count |
| `examples/expressions/strings_search.py` | String search: substring tests, multi-pattern tests, and match counting |
| `examples/expressions/strings_similarity.py` | Fuzzy string matching against a reference value |
| `examples/expressions/strings_slicing.py` | String slicing: taking a fixed piece of every value |
| `examples/expressions/structs_and_maps.py` | Struct and map columns: nested records without flattening the table |
| `examples/expressions/temporal_business_days.py` | Weekend and business-day predicates, and formatting a timestamp for output |
| `examples/expressions/temporal_differences.py` | Durations between two timestamp columns, and shifting a timestamp |
| `examples/expressions/temporal_parts.py` | Pulling calendar parts out of a timestamp column |
| `examples/expressions/temporal_timezones.py` | Time zones: converting between them, and the reporting-boundary trap |
| `examples/expressions/temporal_truncation.py` | Bucketing timestamps: truncate to a period, or snap to a period boundary |
| `examples/expressions/text_quality_filters.py` | Filtering an LLM pretraining corpus with the Gopher document-quality rules |
| `examples/expressions/window_functions.py` | Window functions: per-row values computed from a window of related rows |
| `examples/expr_text/case_and_padding.py` | Case conversion and padding, for fixed-width output |
| `examples/expr_text/cleaning_pipeline.py` | Normalizing free text before anything downstream reads it |
| `examples/expr_text/document_shape.py` | Counting the structure of a document: words, sentences, lines, paragraphs |
| `examples/expr_text/encoding_and_hashing.py` | Encoding bytes and hashing values |
| `examples/expr_text/extracting_entities.py` | Pulling structured values out of free text |
| `examples/expr_text/language_and_script.py` | Detecting non-ASCII content and mixed scripts |
| `examples/expr_text/multiline_and_markdown.py` | Handling text with structure: lines, paragraphs, code fences and markdown |
| `examples/expr_text/normalization_for_matching.py` | Normalizing text so two spellings of the same thing compare equal |
| `examples/expr_text/path_parsing.py` | Pulling the pieces out of a file path or URI |
| `examples/expr_text/pii_detection_and_masking.py` | Finding and masking personal data in a text column |
| `examples/expr_text/quality_ratios.py` | Character-class ratios as a document-quality signal |
| `examples/expr_text/search_and_replace.py` | Finding and rewriting substrings, literally and by pattern |
| `examples/expr_text/similarity_measures.py` | Comparing a column against a target string: edit distance and phonetic keys |
| `examples/expr_text/tokens_and_chunking.py` | Splitting long text into model-sized chunks |
| `examples/expr_text/trimming_and_splitting.py` | Trimming whitespace and splitting on a delimiter |
| `examples/expr_text/word_and_sentence_shaping.py` | Truncating text to a budget, by characters, words or sentences |
| `examples/expr_numeric/absolute_and_sign.py` | Magnitude and sign, and the clipping that bounds a column |
| `examples/expr_numeric/aggregating_ratios_safely.py` | Ratios that survive being aggregated |
| `examples/expr_numeric/cumulative_and_differences.py` | Differences and cumulative sums as expressions |
| `examples/expr_numeric/integer_arithmetic.py` | Integer division, modulo, and the bucketing they give you |
| `examples/expr_numeric/logs_exponents_and_powers.py` | Logarithms, exponentials and powers over a real numeric column |
| `examples/expr_numeric/normalization_expressions.py` | Normalizing a column with expressions rather than a fitted preprocessor |
| `examples/expr_numeric/rounding_and_precision.py` | Rounding: to a place, toward zero, and away from it |
| `examples/expr_numeric/safe_division.py` | Dividing without producing an infinity or a null you did not plan for |
| `examples/expr_numeric/trigonometry_and_geometry.py` | Trigonometric and geometric helpers |
| `examples/expr_temporal/boundaries_and_flags.py` | Is this date the start of something? The boundary predicates |
| `examples/expr_temporal/business_days.py` | Weekdays, weekends, and business-day predicates |
| `examples/expr_temporal/date_arithmetic.py` | Shifting a date by an interval |
| `examples/expr_temporal/date_differences.py` | Measuring the gap between two dates |
| `examples/expr_temporal/date_parts.py` | Pulling the components out of a date column |
| `examples/expr_temporal/date_sequences.py` | Generating a date range, and using it to find the gaps in a series |
| `examples/expr_temporal/duration_bucketing.py` | Bucketing elapsed times into service-level bands |
| `examples/expr_temporal/epochs_and_timestamps.py` | Converting between dates, timestamps and epoch integers |
| `examples/expr_temporal/fiscal_calendars.py` | Reporting on a fiscal year that does not start in January |
| `examples/expr_temporal/formatting_and_parsing.py` | Dates to text and back |
| `examples/expr_temporal/relative_dates.py` | Windows relative to a reference date rather than to today |
| `examples/expr_temporal/timezones.py` | Time zones: converting, and the ambiguity that conversion cannot remove |
| `examples/expr_temporal/truncation_and_periods.py` | Rounding a date down to a period, which is how you group a time series |
| `examples/expr_collections/exploding_and_regrouping.py` | Explode, transform, regroup: the round trip through a flat relation |
| `examples/expr_collections/json_arrays.py` | JSON documents that hold arrays, and getting them into rows |
| `examples/expr_collections/json_columns.py` | Querying JSON held in a string column |
| `examples/expr_collections/list_aggregation_and_stats.py` | Reducing a list column without exploding it |
| `examples/expr_collections/list_basics.py` | Working with a list column: length, indexing, and membership |
| `examples/expr_collections/list_transforms.py` | Reshaping a list column: slicing, flattening, joining, and set operations |
| `examples/expr_collections/map_columns.py` | Map columns: key-value pairs in one column |
| `examples/expr_collections/nested_structs.py` | Structs inside structs, and reaching into them |
| `examples/expr_collections/set_operations_on_lists.py` | Treating list columns as sets, per row |
| `examples/expr_collections/struct_columns.py` | Struct columns: packing several values into one, and reading them back |
| `examples/expr_logic/boolean_algebra.py` | Boolean columns: combining them, and counting what they say |
| `examples/expr_logic/coalesce_and_defaults.py` | Supplying defaults: coalesce, nullif, and the fallback chain |
| `examples/expr_logic/column_selectors.py` | Choosing columns by type or by name pattern instead of listing them |
| `examples/expr_logic/comparison_chains.py` | Range checks, and comparing columns to each other |
| `examples/expr_logic/conditionals.py` | Branching in an expression: when/then/otherwise, and its shorthands |
| `examples/expr_logic/expression_reuse.py` | Defining an expression once and using it several times |
| `examples/expr_logic/horizontal_folds.py` | Reducing across columns in a row, not down a column |
| `examples/expr_logic/null_handling.py` | Nulls: detecting them, filling them, and the arithmetic they poison |
| `examples/expr_logic/predicate_pushdown_shapes.py` | Which predicate shapes can be pushed into a scan, and which cannot |
| `examples/expr_logic/type_coercion.py` | What happens when an expression mixes types |
| `examples/expr_logic/window_free_ranking.py` | Ranking without a window: `top_k`, `nlargest` and a self-join |
| `examples/expr_vectors/distance_measures.py` | Distances between embedding vectors held in a list column |
| `examples/expr_vectors/normalization_and_pooling.py` | Vector shape: magnitudes, unit norm, and pooling many vectors into one |
<!-- /library-table -->
