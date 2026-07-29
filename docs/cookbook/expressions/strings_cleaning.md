# Cleaning scraped text: strip markup, URLs, emails, and stray punctuation

This is the pre-processing pass in front of an embedding or LLM stage. Each call is one columnar operator, so a chain of ten of them still reads the column once per operator in Rust rather than materializing Python strings.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/strings_cleaning.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_cleaning.py
```
