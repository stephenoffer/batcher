# The .list, .struct, .map, and .json namespaces

This page is the reference for the four accessors that read inside a nested column: `.list` on a list column, `.struct` on a struct, `.map` on a map, and `.json` on a string column holding JSON text. Reach each one from an expression, such as {py:obj}`col("tags").list <batcher.plan.expr_ir.core.Expr.list>`.

## The `.list` namespace

Methods on a list expression, reached as `col("xs").list`.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autoclass:: _ListNamespace
   :no-members:
```

### Access

Read the length of a list, take elements out of it by position, or find a value in it.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autosummary::
   :toctree: generated
   :nosignatures:

   _ListNamespace.len
   _ListNamespace.get
   _ListNamespace.first
   _ListNamespace.last
   _ListNamespace.head
   _ListNamespace.slice
   _ListNamespace.gather
   _ListNamespace.contains
   _ListNamespace.position
```

### Reshaping and transforming

Sort, deduplicate, filter, map, or flatten each list, or join its elements into a string.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autosummary::
   :toctree: generated
   :nosignatures:

   _ListNamespace.sort
   _ListNamespace.arg_sort
   _ListNamespace.reverse
   _ListNamespace.unique
   _ListNamespace.drop_nulls
   _ListNamespace.filter
   _ListNamespace.transform
   _ListNamespace.flatten
   _ListNamespace.concat
   _ListNamespace.append
   _ListNamespace.prepend
   _ListNamespace.remove
   _ListNamespace.join
   _ListNamespace.cum_sum
   _ListNamespace.diff
```

### Aggregates

Reduce each list to one value.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autosummary::
   :toctree: generated
   :nosignatures:

   _ListNamespace.sum
   _ListNamespace.product
   _ListNamespace.min
   _ListNamespace.max
   _ListNamespace.mean
   _ListNamespace.median
   _ListNamespace.std
   _ListNamespace.var
   _ListNamespace.n_unique
   _ListNamespace.arg_min
   _ListNamespace.arg_max
   _ListNamespace.entropy
```

### Set operations and overlap

Combine two lists as sets, or measure how much they share.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autosummary::
   :toctree: generated
   :nosignatures:

   _ListNamespace.union
   _ListNamespace.intersect
   _ListNamespace.difference
   _ListNamespace.has_any
   _ListNamespace.has_all
   _ListNamespace.jaccard
   _ListNamespace.multiset_overlap
   _ListNamespace.lcs_length
```

### Vector arithmetic and norms

Treat each list as a numeric vector, such as an embedding, and compute on it element-wise.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autosummary::
   :toctree: generated
   :nosignatures:

   _ListNamespace.dot
   _ListNamespace.add
   _ListNamespace.subtract
   _ListNamespace.multiply
   _ListNamespace.l2_norm
   _ListNamespace.l1_norm
   _ListNamespace.max_abs
   _ListNamespace.sum_squares
   _ListNamespace.normalize
   _ListNamespace.softmax
   _ListNamespace.log_softmax
   _ListNamespace.is_unit_norm
   _ListNamespace.is_zero_vector
```

### Vector distances and signatures

Compare two vector columns, or hash a vector into a signature.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autosummary::
   :toctree: generated
   :nosignatures:

   _ListNamespace.cosine_similarity
   _ListNamespace.cosine_distance
   _ListNamespace.angular_distance
   _ListNamespace.l2_distance
   _ListNamespace.l1_distance
   _ListNamespace.hamming_distance
   _ListNamespace.simhash
```

## The `.struct` namespace

Methods on a struct expression, reached as `col("s").struct`.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autoclass:: _StructNamespace
   :no-members:
```

Read a struct's fields by name, or list the field names.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autosummary::
   :toctree: generated
   :nosignatures:

   _StructNamespace.field
   _StructNamespace.get
   _StructNamespace.keys
```

## The `.map` namespace

Methods on an Arrow `Map` column, reached as `col("m").map`.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autoclass:: _MapNamespace
   :no-members:
```

Look up a key, test for one, or read a map's keys, values, and entries as lists.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autosummary::
   :toctree: generated
   :nosignatures:

   _MapNamespace.get
   _MapNamespace.contains
   _MapNamespace.len
   _MapNamespace.keys
   _MapNamespace.values
   _MapNamespace.entries
```

## The `.json` namespace

Methods on a string column holding JSON documents, reached as `col("doc").json`.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autoclass:: _JsonNamespace
   :no-members:
```

Read typed values at a JSON path, and inspect the document's keys, types, and shape.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autosummary::
   :toctree: generated
   :nosignatures:

   _JsonNamespace.extract_string
   _JsonNamespace.extract_int
   _JsonNamespace.extract_float
   _JsonNamespace.extract_bool
   _JsonNamespace.value
   _JsonNamespace.values
   _JsonNamespace.exists
   _JsonNamespace.contains
   _JsonNamespace.keys
   _JsonNamespace.array_length
   _JsonNamespace.type_of
   _JsonNamespace.structure
   _JsonNamespace.pretty
```

## See also

- {doc}`index`: the other accessor namespaces, and which column kind each one attaches to.
- {doc}`/api/relational/expression-accessors`: the same methods with a runnable example per namespace.
- {doc}`/api/symbols/expression-methods`: the {py:obj}`Expr <batcher.plan.expr_ir.core.Expr>` these namespaces hang off.
- {doc}`/user-guide/transform/columns/map-accessor`: building and reading a map column.
- {doc}`/cookbook/expressions/nested/index`: runnable recipes for lists, structs, maps, and JSON.
