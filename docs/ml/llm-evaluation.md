# Evaluating LLM output

This page covers measuring generation quality: lexical-overlap scores against a gold
column, and the reference-free monitors a team watches when generation runs at scale.

Every metric here is an expression that aggregates to a corpus score in one scan, so
there is no Python loop over examples and every one of them composes with `group_by`.

## Scoring against a reference

Evaluating generations is comparing a generated column to a gold column.
`bt.exact_match` is the strict character-for-character rate. `bt.normalized_exact_match`
applies SQuAD normalization first (lowercase, drop articles and punctuation), so casing and a
trailing period do not count against a correct answer.

```python
import batcher as bt

evals = bt.from_pydict(
    {"answer": ["The capital is Paris.", "It is Rome"], "gold": ["Paris", "London"]}
)
print(evals.agg(em=bt.normalized_exact_match("answer", "gold")).to_pydict())
```

For free-form answers where neither exact match nor a single word is right, the token metrics
compare the *sets* of words (repeats counted once): `bt.token_set_precision`, `bt.token_set_recall`,
`bt.token_set_f1` (the balanced default), and `bt.token_set_jaccard`. `bt.length_ratio` reports how
verbose the output is relative to the reference, which catches a model that systematically over- or
under-generates. Every one composes with `group_by` to score per model, per prompt template, or per
slice in the same pass.

```python
scored = bt.from_pydict(
    {"model": ["a", "a", "b", "b"],
     "answer": ["the quick brown fox", "yes", "a slow brown fox", "no"],
     "gold": ["a fast brown fox", "yes", "the brown fox", "yes"]}
)
print(scored.group_by("model").agg(f1=bt.token_set_f1("answer", "gold")).sort("model").to_pydict())
```

These are set-based by design, so they are stable and fast rather than a multiset BLEU/ROUGE score;
each metric's docstring states this so it is never confused with one.

The token-set metrics split on whitespace, which fails on a language that does not put spaces
between words. `bt.char_ngram_f1` scores the overlap of *character* n-grams instead, the idea behind
chrF, so it works on Chinese, Japanese, or heavily inflected output with no tokenizer.
`bt.char_ngram_precision`, `bt.char_ngram_recall`, and `bt.char_ngram_jaccard` are the matching
directional and set-similarity views.

```python
cjk = bt.from_pydict({"pred": ["東京都"], "gold": ["東京市"]})
print(cjk.agg(chrf=bt.char_ngram_f1("pred", "gold", n=2)).to_pydict())
# {'chrf': [0.5]}
```

### Counting repeats: the BLEU and ROUGE-N scores

The set metrics above ask which words two texts share. They cannot see repetition, and that
blind spot has a cost: a model stuck emitting `cat cat cat cat` shares the word `cat` with
its reference, so a set precision reads a perfect 1.0.

The clipped metrics count occurrences instead, capping each n-gram at the number of times the
reference actually contains it. `bt.ngram_precision` is BLEU's per-order term,
`bt.ngram_recall` is ROUGE-N, and `bt.ngram_f1` balances the two. Pass `n` to choose the
order: unigrams measure content coverage, and higher orders measure whether the wording
survived.

```python
degenerate = bt.from_pydict({"answer": ["cat cat cat cat"], "gold": ["cat sat down"]})
print(degenerate.agg(clipped=bt.ngram_precision("answer", "gold")).to_pydict())
# {'clipped': [0.25]}
```

`bt.bleu` combines the orders: the geometric mean of the clipped precisions for `1..max_n`,
multiplied by `bt.brevity_penalty`, which is what stops a one-word answer from scoring
perfectly on precision alone. It is unsmoothed, so an example sharing no 4-gram scores zero.
Lower `max_n` for short-answer tasks rather than reading a column of zeros.

```python
summaries = bt.from_pydict(
    {"answer": ["the quick brown fox jumps"], "gold": ["the quick brown fox jumps"]}
)
print(summaries.agg(bleu=bt.bleu("answer", "gold")).to_pydict())
# {'bleu': [1.0]}
```

Two more read the generation against itself or its source. `bt.distinct_ngram_ratio` is the
phrase-level diversity score, which catches a model looping on a sentence long before
`bt.distinct_token_ratio` moves. `bt.ngram_novelty` is the copying check: at `n=4` or higher,
a value near zero means the output is reproducing its retrieved context verbatim rather than
writing from it.

```python
rag = bt.from_pydict(
    {
        "answer": ["the quick brown fox jumps"],
        "context": ["the quick brown fox jumps over the lazy dog"],
    }
)
print(rag.agg(novel=bt.ngram_novelty("answer", "context")).to_pydict())
# {'novel': [0.0]}
```

All of these tokenize with the same SQuAD normalization the token-set metrics use, so the
numbers are comparable across this page. That is not what a reference BLEU implementation
does, so use them to rank runs against each other rather than to publish against a paper.

The two primitives underneath are on the expression accessors, for a score this page does not
already spell. `str.token_ngrams(n)` turns a text column into its list of n-grams, and
`list.multiset_overlap` counts how many of one list's elements another can account for,
capping each at the number of times it appears. Divide the second by a length to build any
clipped-overlap score you need:

```python
grams = bt.from_pydict({"answer": ["cat sat on the mat"], "gold": ["cat sat on a mat"]})
pred = bt.col("answer").str.token_ngrams(2)
gold = bt.col("gold").str.token_ngrams(2)
print(grams.select(shared=pred.list.multiset_overlap(gold), total=pred.list.len()).to_pydict())
# {'shared': [2.0], 'total': [4]}
```

## Scoring generations without a reference

Most generations arrive with no gold answer to compare against, and the questions you still want
answered are about the output itself: is it diverse or repeating, how long is it, and how often is
it empty, a refusal, or cut off. These metrics take one output column and aggregate to a corpus
number, so they run over a million generations in one scan and break down per model or per day with
`group_by`.

`bt.distinct_token_ratio` is the Distinct-1 diversity score, the cheap detector of a model
degenerating into repetition. `bt.mean_output_tokens` tracks verbosity and sizes the token bill.
`bt.empty_generation_rate`, `bt.refusal_rate`, and `bt.truncation_rate` are the three failure rates
worth a dashboard: silent empty outputs, declined answers, and responses that stop mid-sentence.

```python
gens = bt.from_pydict(
    {
        "out": [
            "The capital of France is Paris.",
            "yes yes yes yes yes",
            "I'm sorry, I can't help with that.",
            "The list of steps is as follows",
        ]
    }
)
print(
    gens.agg(
        diversity=bt.distinct_token_ratio("out"),
        refused=bt.refusal_rate("out"),
        truncated=bt.truncation_rate("out"),
    ).to_pydict()
)
# {'diversity': [0.8], 'refused': [0.25], 'truncated': [0.5]}
```

They are lexical heuristics, so read them as monitors that catch a regression between runs, not as
ground-truth judgments of a single generation.

Before a run rather than after it, the token aggregates size the bill and the capacity.
`bt.total_token_estimate` sums the corpus token estimate for a cost number, `bt.token_budget_exceed_rate`
is the fraction of rows that will overflow a given context window, and `bt.token_estimate_quantile`
is the length tail that sizes the window. All take either the prompt or the output column.

```python
reqs = bt.from_pydict({"prompt": ["short one", "a considerably longer prompt string here"]})
print(
    reqs.agg(
        total=bt.total_token_estimate("prompt"),
        over=bt.token_budget_exceed_rate("prompt", budget=5),
    ).to_pydict()
)
# {'total': [12], 'over': [0.5]}
```

A second set of monitors watches for output a model *should not* produce at scale. `bt.all_caps_rate`
and `bt.repeated_punctuation_rate` catch shouting and degenerate punctuation, `bt.non_ascii_rate`
flags encoding or language drift, `bt.url_rate` surfaces hallucinated links or prompt injection, and
`bt.code_block_rate` catches a code block leaking into a prose task. `bt.long_output_rate` and
`bt.short_output_rate` bound the length distribution, and `bt.mean_sentence_count` and
`bt.mean_word_length` track structural and lexical drift.

```python
outputs = bt.from_pydict(
    {"out": ["STOP.", "see https://spam.example", "a normal, useful answer here"]}
)
print(
    outputs.agg(
        shouting=bt.all_caps_rate("out"),
        links=bt.url_rate("out"),
    ).to_pydict()
)
# {'shouting': [0.3333333333333333], 'links': [0.3333333333333333]}
```

## More output monitors

Four further families of single-scan monitors cover the rest of what a generation-at-scale team
watches, and all compose with `group_by`.

For a RAG pipeline, compare the answer column against its retrieved context. `bt.answer_groundedness`
is the share of the answer's tokens the context supports, `bt.context_utilization` the share of the
context the answer drew on, `bt.unsupported_token_rate` the hallucination-proxy complement, and
`bt.fully_grounded_rate` the fraction of answers entirely supported. `bt.citation_rate` tracks how
often the model cited a source at all.

For reading level, `bt.automated_readability_index` is the ARI grade, with `bt.mean_words_per_sentence`,
`bt.mean_chars_per_word`, `bt.long_word_rate`, and `bt.mean_paragraph_count` as the complexity drivers
behind it.

For degeneration, `bt.distinct_char_ngram_ratio` and its complement `bt.char_repetition_rate` catch a
model looping at the character level, `bt.distinct_token_ratio` at the word level,
`bt.repeated_line_rate` catches duplicated lines, and `bt.compression_ratio_proxy` is a cheap
gzip-style repetition score.

For safety, `bt.email_rate`, `bt.phone_rate`, and `bt.pii_rate` flag leaked contact details,
`bt.ssn_like_rate` and `bt.credit_card_like_rate` catch structured identifiers, and
`bt.contains_any_rate` is a configurable blocklist monitor over a list of terms.

For formatting, `bt.heading_rate`, `bt.bullet_list_rate`, `bt.numbered_list_rate`,
`bt.markdown_link_rate`, `bt.table_rate`, and `bt.code_block_present_rate` check whether the model
produced the Markdown elements a task asked for.

For tone, `bt.question_rate` catches a model deflecting an answer task with a question,
`bt.exclamation_rate` and `bt.politeness_rate` track register, `bt.hedge_rate` flags uncertainty,
`bt.first_person_rate` measures first-person voice, and `bt.contains_phrase_rate` is a configurable
phrase monitor.

For language, `bt.cjk_rate`, `bt.cyrillic_rate`, and `bt.arabic_rate` flag unexpected scripts,
`bt.emoji_rate` catches emoji spam, and `bt.latin_only_rate` is the clean-ASCII-output rate.

## See also

- {doc}`llm`: running the generation being scored.
- {doc}`llm-outputs`: turning generations into the typed columns these metrics read.
- {doc}`evaluation`: the model-evaluation metrics for classification and regression.
