# API

The Batcher API is small and lazy. You build a `Dataset` from a source, transform
it with expression-based operations, and execute it with a terminal operation that
returns Arrow or writes to a sink. Everything reachable from `import batcher as bt`
is documented in this section.

This section has three layers. The **quick reference** is a one-page map of the
surface. The **topic pages** explain each area with runnable examples. The
**complete reference** is generated from the source docstrings and documents every
public name, with a page per function.

```{toctree}
:hidden:
:caption: Overview

reference
```

```{toctree}
:maxdepth: 1
:caption: Topics

dataset
expressions
io
sql
ml
configuration
exceptions
```

```{toctree}
:hidden:
:caption: Complete reference

complete
```
