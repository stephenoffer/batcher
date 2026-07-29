# Moving data in and out of other frameworks, zero-copy where possible

Arrow is the shared contract, so ``from_arrow``/``to_arrow`` are the cheapest boundary there is. The pandas and Polars bridges go through Arrow too, which is why they are much cheaper than a row-by-row conversion.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/io/arrow_interop.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/io/arrow_interop.py
```
