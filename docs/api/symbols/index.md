# Every symbol

This section lists every public name in `batcher`, rendered from the source docstrings, with one page per object family. It is where you land when you know what you are looking for and want the signature, the arguments, and what it returns.

It is the exhaustive half of the reference. The {doc}`quick reference </api/reference>` is the one-page lookup table for the calls you reach for daily, and the {doc}`area pages </api/index>` explain the same surface with a runnable example per group. Each name below also has its own generated page, so a search result lands on one method rather than on a page holding two hundred.

## The pages

Pages are grouped by the object you call the method on. The following table lists them in the order a pipeline meets them, from building a dataset to running it.

| Page | Holds |
| --- | --- |
| {doc}`construction` | The functions that build a `Dataset` from data already in the process |
| {doc}`readers-and-writers` | Every `bt.read.*` reader and `ds.write.*` writer, by source family |
| {doc}`dataset-transforms` | The `Dataset` methods that return a new plan and run nothing |
| {doc}`dataset-terminal` | The `Dataset` methods that execute a plan, or report on one |
| {doc}`groupby` | The `GroupBy` a `group_by` returns, and its aggregates |
| {doc}`dataset-accessors` | The `ml`, `dq`, `scd`, and `meta` namespaces a dataset hands out |
| {doc}`expression-builders` | `col`, `lit`, `when`, and the column selectors |
| {doc}`expression-methods` | The `Expr` methods that compute one value per row |
| {doc}`expression-aggregates` | The `Expr` methods that collapse rows, and the `AggExpr` they return |
| {doc}`expression-modeling` | Feature transforms, activations, introspection, and the IR |

Three surfaces are listed elsewhere, because each has an area page of its own that already enumerates it: {doc}`/api/accessors/index` for the typed namespaces such as `.str` and `.image`, {doc}`/api/relational/functions` for the free scalar, aggregate, and window functions, and {doc}`/api/models/index` for the model, preprocessor, and metric surfaces.

## Why the surface splits this way

Two hundred methods on one page is a word list, and that is what this section used to be: `expressions` alone rendered 763 signatures. The split follows the seams the engine itself has, so the page you need is the one you would have guessed.

A `Dataset` method either adds a node to a plan or executes one, and nothing does both. An `Expr` method either computes per row, collapses rows, or reads the expression's own tree. A reader returns a lazy dataset and a writer consumes one. Those distinctions decide what a call costs and when it runs, so they are worth carrying in the table of contents rather than burying in a heading halfway down a page.

## See also

- {doc}`/api/reference`: the same surface as a short lookup table rather than a full listing.
- {doc}`/api/relational/index`, {doc}`/api/models/index`, {doc}`/api/operations/index`: the area pages, with the semantics behind each call.
- {doc}`/api/operations/exceptions`: what these calls raise, and which builtin each error also subclasses.
- {doc}`/user-guide/index`: the task-oriented guides this section is the reference for.

```{toctree}
:hidden:

construction
readers-and-writers
dataset-transforms
dataset-terminal
groupby
dataset-accessors
expression-builders
expression-methods
expression-aggregates
expression-modeling
```
