# Strings

Text columns are where Python row loops hide, and the {py:class}`.str <batcher.plan.expr_ir.namespaces.strings._StrNamespace>` accessor is how you get rid of them. Every call is one columnar operator in Rust, so cleaning, matching, and measuring a million documents never materializes a Python string.

The fourteen recipes fall into three groups: changing what a value looks like, finding something inside it, and turning it into a number or a key. Preparing text for an LLM or an embedding model? Read cleaning, predicates, and ratios first.

| Group | Recipes | Covers |
|---|---|---|
| {doc}`/cookbook/expressions/strings/shaping/index` | 5 | Case, padding, slicing, cleaning, and chunking |
| {doc}`/cookbook/expressions/strings/matching/index` | 5 | Predicates, search, regex, fuzzy matching, and extraction |
| {doc}`/cookbook/expressions/strings/measuring/index` | 4 | Counts, character-class ratios, hashing, and paths |

```{toctree}
:hidden:

shaping/index
matching/index
measuring/index
```
