# What is installed, what the engine sees, and what to paste into a bug report

Half of "it works on my machine" is an optional extra present in one environment and absent in the other. These calls answer that in one line, and they are the first thing to include when reporting a problem.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/operations/environment.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/operations/environment.py
```
