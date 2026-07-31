# Measuring and encoding

Turning a string column into a number, a key, or a checksum.

Each page embeds a complete, self-contained script that builds its own in-memory data and asserts on its own output, so a page that stops matching the engine fails the suite instead of drifting.

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
