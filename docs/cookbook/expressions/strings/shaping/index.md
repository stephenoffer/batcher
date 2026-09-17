# Shaping text

Changing what a string value looks like, before anything compares, joins, or groups on it. Case and padding fix the keys that fail to match across systems, slicing takes a fixed piece of every value, and cleaning and chunking prepare scraped text for an embedding or LLM stage.

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
