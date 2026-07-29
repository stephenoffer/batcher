# Save modes and write manifests: what happens when the target already exists

The default refuses to clobber, which is the safe choice for a job that might be retried. ``overwrite`` replaces, ``append`` adds. Every write returns a manifest describing what it actually produced, which is what you record for lineage or resume.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/io/save_modes.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/io/save_modes.py
```
