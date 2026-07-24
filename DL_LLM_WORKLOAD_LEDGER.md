# Deep Learning & LLM workload improvements — ledger

Goal: 400+ distinct, tested, documented improvements/features/integrations for deep-learning and
LLM data workloads (new features, easier integration, better perf/scalability).

Verification: hand-written references (no rouge_score/sacrebleu/jiwer installed), behavior tests,
and scikit-learn/numpy where a numeric oracle exists. Every relational/expression addition follows
the control-plane discipline (no O(rows) driver work; per-row work stays in Rust/expressions).

## Cluster A — Generation-quality metrics (LLM output eval)

- [x] A.1-A.7 `exact_match`, `normalized_exact_match` (SQuAD normalization), `token_set_precision`,
  `token_set_recall`, `token_set_f1`, `token_set_jaccard`, `length_ratio` — expression-based
  generation-eval metrics that compare a generated column to a reference column and aggregate to a
  corpus score in one scan (column-vs-column via the list set operations; no per-row Python).
  Verified against hand-written SQuAD-normalized set-overlap references. Surfaced through the `bt.`
  metrics facade.

## Cluster B — Embedding-quality metrics

- [x] B.1-B.6 `mean_cosine_similarity`, `mean_euclidean_distance`, `mean_dot_product`,
  `mean_embedding_norm`, `unit_norm_rate`, `zero_vector_rate` — per-row vector operations over
  Arrow list columns that aggregate to a corpus score (retrieval alignment, magnitude drift,
  degenerate-embedding detection). The vector math runs in Rust; matched to numpy. `bt.` facade.

## Cluster C — LLM output parsing (generated text → typed columns, no second model call)

- [x] C.1-C.11 `extract_json`, `extract_json_array`, `extract_code_block`, `extract_first_number`,
  `extract_tag`, `extract_reasoning`, `strip_reasoning`, `extract_after`, `extract_between`,
  `is_refusal`, `extract_choice` — vectorized regex extractors that turn a model's prose-wrapped
  output into a typed column (JSON blob, fenced code, `<answer>` tag, `<think>` trace, marker
  slices, multiple-choice letter, refusal flag) in the same scan, no GPU and no second inference
  pass. Each degrades to an empty string / null where the fragment is absent. Surfaced as `bt.*`
  free functions; tested in `tests/unit/test_llm_output_parsing.py` against hand references and
  taught in `docs/ml/llm.md` ("Parsing without a second model call"). The complement to the
  model-in-the-loop `ds.ml.extract` / `ds.ml.classify`.

## Cluster D — Reference-free generation-quality metrics (corpus signals, single output column)

- [x] D.1-D.5 `distinct_token_ratio` (Distinct-1 diversity / degeneration detector),
  `mean_output_tokens` (verbosity / token-bill sizing), `empty_generation_rate`,
  `refusal_rate` (safety eval; reuses `is_refusal`), `truncation_rate` (max-token-cutoff proxy) —
  single-column mergeable aggregates that score an output column with no gold reference, the
  numbers a generation-at-scale team dashboards. Compose with `group_by` for per-model/per-day
  breakdowns. Tested in `tests/unit/test_generation_quality_metrics.py` against hand references;
  taught in `docs/ml/llm.md` ("Scoring generations without a reference"). `bt.` metrics facade.

## Cluster E — Character n-gram overlap metrics (chrF-style, language-agnostic generation eval)

- [x] E.1-E.4 `char_ngram_precision`, `char_ngram_recall`, `char_ngram_f1` (chrF-style),
  `char_ngram_jaccard` — set-based character n-gram overlap between a generated and a reference
  column, built on `.str.chunk(n, overlap=n-1)` for the n-grams and the list set-ops for the
  overlap. Unlike the whitespace token-set metrics they need no word boundaries, so they score
  Chinese/Japanese/inflected output (CJK case tested). Tested in
  `tests/unit/test_char_ngram_metrics.py` against hand-computed bigram sets; taught in
  `docs/ml/llm.md`. `bt.` metrics facade.

## Cluster F — Text-quality & safety monitors (single-column corpus rates)

- [x] F.1-F.9 `all_caps_rate`, `repeated_punctuation_rate`, `non_ascii_rate`, `url_rate`,
  `code_block_rate`, `long_output_rate`, `short_output_rate`, `mean_sentence_count`,
  `mean_word_length` — corpus-rate monitors for what a generation should not do at scale (shouting,
  degenerate punctuation, encoding/language drift, hallucinated links/injection, code leakage,
  length distribution, structural/lexical drift). Each a single mergeable aggregate over an existing
  `.str` primitive. Tested in `tests/unit/test_text_quality_monitors.py`; taught in `docs/ml/llm.md`.
  `bt.` metrics facade.

## Cluster G — Embedding distance metrics (the distance spaces ANN indexes use)

- [x] G.1-G.4 `mean_cosine_distance` (1 − cos, the cosine-index drift number), `mean_manhattan_distance`
  (L1, robust to a dominant dimension), `mean_angular_distance` (true-metric normalized angle),
  `mean_hamming_distance` (bit disagreement for binary/product-quantized vectors) — per-row paired
  vector distances aggregated to a corpus mean, computed in Rust over the Arrow list. Extends the
  embedding-metrics family (Cluster B) across the metrics a real ANN index ranks by. Verified against
  numpy in `tests/unit/test_embedding_distance_metrics.py`; taught in `docs/ml/embeddings.md`.

## Cluster H — Token & cost aggregates (LLM capacity/cost planning)

- [x] H.1-H.3 `total_token_estimate` (corpus token bill), `token_budget_exceed_rate` (context-overflow
  fraction), `token_estimate_quantile` (the token-length tail that sizes the window, via a mergeable
  quantile sketch) — mergeable aggregates over the tokenizer-free `estimate_tokens` heuristic that
  size an LLM run before it is paid for, on either the prompt or output column. Tested in
  `tests/unit/test_token_cost_metrics.py`; taught in `docs/ml/llm.md`. `bt.` metrics facade.

## Cluster I — Structured-output compliance metrics (JSON-mode / tagged-answer reliability)

- [x] I.1-I.3 `valid_json_rate` (strict: whole output is JSON), `json_present_rate` (lenient: a JSON
  object is extractable from prose), `tagged_answer_rate` (a non-empty ``<tag>...</tag>`` block is
  present) — corpus compliance rates that measure whether a model returned the requested shape, the
  silent-failure axis a JSON-mode or tool-use pipeline depends on. Tested in
  `tests/unit/test_structured_output_compliance.py`; taught in `docs/ml/llm.md`. `bt.` metrics facade.

## Cluster J — Answer-format compliance metrics (benchmark/QA output gradeability)

- [x] J.1-J.3 `numeric_answer_rate` (a number is parseable — math/counting tasks), `choice_answer_rate`
  (a standalone A-H letter — multiple choice), `boxed_answer_rate` (a LaTeX ``\boxed{}`` answer — the
  MATH convention) — corpus rates measuring whether a benchmark run's outputs are in a gradeable
  answer format, the axis that separates "answering in prose the grader can't read" from "getting it
  wrong". Built on the Cluster C parsers. Tested in `tests/unit/test_structured_output_compliance.py`;
  taught in `docs/ml/llm.md`. `bt.` metrics facade.

## Cluster K — Prompt construction (row-wise builders, ease-of-use)

- [x] K.1-K.3 `render_template` (named `{placeholder}` interpolation from columns), `wrap_tag`
  (wrap a field in `<tag>...</tag>` for structured prompts), `truncate_to_token_budget` (trim a
  column to fit a context window) — row-wise string builders for assembling prompts in the data
  plane. Taught in `docs/ml/llm.md`; tested in `tests/unit/test_prompt_construction.py`. Surfaced
  via `plan/functions/__init__.py` + the `bt.*` facade.

## Cluster L — Readability metrics (built by subagent, integrated + gated by parent)

- [x] L.1-L.5 `automated_readability_index` (ARI), `mean_words_per_sentence`, `mean_chars_per_word`,
  `long_word_rate`, `mean_paragraph_count` — score how complex a generated column reads, for
  matching a target reading level. `tests/unit/test_readability_metrics.py`.

## Cluster M — RAG grounding metrics (subagent-built, parent-integrated)

- [x] M.1-M.5 `answer_groundedness`, `context_utilization`, `unsupported_token_rate`,
  `fully_grounded_rate`, `citation_rate` — compare a generated answer column to its retrieved
  context column (token-set groundedness / hallucination proxies for RAG).
  `tests/unit/test_retrieval_grounding_metrics.py`.

## Cluster N — PII & safety monitors (subagent-built, parent-integrated)

- [x] N.1-N.6 `email_rate`, `phone_rate`, `pii_rate`, `ssn_like_rate`, `credit_card_like_rate`,
  `contains_any_rate` — corpus rates flagging leaked contact details, structured identifiers, or
  blocklisted terms in generated text. `tests/unit/test_pii_safety_metrics.py`.

## Cluster O — Repetition / degeneration metrics (subagent-built, parent-integrated)

- [x] O.1-O.5 `distinct_char_ngram_ratio`, `char_repetition_rate`, `word_type_token_ratio`,
  `repeated_line_rate`, `compression_ratio_proxy` — detect a model looping or repeating n-grams
  (character, word, and line level). `tests/unit/test_repetition_metrics.py`.

## Method note

Clusters L-O were built in parallel by 4 subagents, each owning ONE new self-contained module +
test file and explicitly forbidden from touching the shared facades (`api/functions.py`,
`metrics/__init__.py`) or docs — the parent agent did all facade/doc wiring and the full gate
serially, so parallel subagents never collided on the funnel files. This is the scalable pattern
for facade-bottlenecked additions.

## Cluster P — Markdown formatting metrics (subagent-built, parent-integrated)

- [x] P.1-P.6 `heading_rate`, `bullet_list_rate`, `numbered_list_rate`, `markdown_link_rate`,
  `table_rate`, `code_block_present_rate` — whether generated text used the Markdown elements a task
  asked for. `tests/unit/test_formatting_metrics.py`.

## Cluster Q — Tone & style metrics (subagent-built, parent-integrated)

- [x] Q.1-Q.6 `question_rate`, `exclamation_rate`, `hedge_rate`, `first_person_rate`,
  `politeness_rate`, `contains_phrase_rate` — register/tone monitors (deflection, over-excitement,
  hedging, voice, configurable phrase list). `tests/unit/test_tone_metrics.py`.

## Cluster R — Script / language metrics (subagent-built, parent-integrated)

- [x] R.1-R.5 `cjk_rate`, `cyrillic_rate`, `arabic_rate`, `emoji_rate`, `latin_only_rate` —
  character-set composition, for language drift and emoji spam. Uses the Rust regex `\p{Script}`
  Unicode classes. `tests/unit/test_script_metrics.py`.

## Progress
- 96 distinct new public names so far (A: 7, B: 6, C: 11, D: 5, E: 4, F: 9, G: 4, H: 3, I: 3, J: 3,
  K: 3, L: 5, M: 5, N: 6, O: 5, P: 6, Q: 6, R: 5).

## Note on the buildable metric catalog

The marquee multiset metrics (BLEU, ROUGE-N, METEOR, Distinct-2/3) need *token* n-gram counting,
which the current expression primitives do not provide (`.str.chunk` gives character n-grams only,
and there is no multiset/count-vector op over a token list). chrF (character n-grams) is reachable
and built here; the token-n-gram metrics would need a new Rust primitive (a token-n-gram + multiset
op), a larger cross-crate change deferred rather than rushed in a shared tree. The set-based token
and character overlap families cover the tokenizer-free eval surface that is buildable today.
