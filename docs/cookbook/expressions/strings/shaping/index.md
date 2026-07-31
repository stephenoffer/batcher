# Shaping text

Changing what a string value looks like, before anything compares or groups on it.

Each page embeds a complete, self-contained script that builds its own in-memory data and asserts on its own output, so a page that stops matching the engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/expressions/strings/shaping/strings_case` | Normalizing capitalization before you compare or group |
| {doc}`/cookbook/expressions/strings/shaping/strings_padding` | Fixed-width keys, and cleaning stray whitespace |
| {doc}`/cookbook/expressions/strings/shaping/strings_slicing` | Taking a fixed piece of every value |
| {doc}`/cookbook/expressions/strings/shaping/strings_cleaning` | Stripping markup, URLs, emails, and stray punctuation |
| {doc}`/cookbook/expressions/strings/shaping/strings_chunking` | Overlapping chunks for a RAG index |

```{toctree}
:hidden:

strings_case
strings_padding
strings_slicing
strings_cleaning
strings_chunking
```
