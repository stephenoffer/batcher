# Environment report

Half of "it works on my machine" is an optional extra present in one environment and absent in the other. `bt.versions()` and `bt.show_versions()` answer that in one line, so paste them into any bug report. Check `engine_profile` before you trust a timing: a debug build and a release build are not comparable.

The script also lists every option with `option_names()` and the environment variable that overrides each with `env_var_names()`, which is how you configure a container without editing code.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/operations/environment.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/operations/environment.py
```

## See also

- {doc}`configuration`: options, scoped overrides, and profiles.
- {doc}`error_handling`: catching the failure you meant to catch.
- {doc}`/user-guide/operate/tuning/performance`: measuring and tuning a query that is correct but slow.
- {doc}`/user-guide/operate/running/observability`: what the engine records about a run, and where.
