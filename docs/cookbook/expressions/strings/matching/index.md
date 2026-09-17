# Matching and extracting

Finding something in a string column, and pulling it out. Predicates and search give you boolean screens for a filter, regex and extraction turn matches into new columns, and similarity finds values that are almost, but not exactly, a known string.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/expressions/strings/matching/strings_predicates` | The boolean screen in front of an expensive stage |
| {doc}`/cookbook/expressions/strings/matching/strings_search` | Substring tests, multi-pattern tests, and match counting |
| {doc}`/cookbook/expressions/strings/matching/strings_regex` | Extract, replace, and count with a pattern |
| {doc}`/cookbook/expressions/strings/matching/strings_similarity` | Fuzzy matching against a reference value |
| {doc}`/cookbook/expressions/strings/matching/strings_extraction` | Pulling entities and leading fragments out of free text |

```{toctree}
:hidden:

strings_predicates
strings_search
strings_regex
strings_similarity
strings_extraction
```
