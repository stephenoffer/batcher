# Measuring and encoding

Turning a string column into a number, a key, or a set of columns. Counts and character-class ratios are the cheap quality signals for filtering a text corpus, hashing gives you stable keys and shards, and path parsing turns an object-storage listing into columns you can group by.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/expressions/strings/measuring/strings_counts` | Words, lines, sentences, and entities |
| {doc}`/cookbook/expressions/strings/measuring/strings_ratios` | Cheap quality signals for a text corpus |
| {doc}`/cookbook/expressions/strings/measuring/strings_hashing` | Keys, checksums, and safe transport |
| {doc}`/cookbook/expressions/strings/measuring/strings_paths` | Parsing file paths held in a column |

```{toctree}
:hidden:

strings_counts
strings_ratios
strings_hashing
strings_paths
```
