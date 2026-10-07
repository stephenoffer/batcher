"""Arrow → Hugging Face ``datasets`` conversion, with feature translation.

A ``datasets.Dataset`` is an Arrow table plus a ``Features`` description of it, so the
export is mostly a matter of stating the features: ``Features.from_arrow_schema`` covers
every plain column (scalars, lists, structs, nested combinations), and the two kinds HF
models with a richer feature than their storage are translated here:

- **Image.** HF's ``Image()`` stores ``struct<bytes: binary, path: string>``. A column named
  in ``images`` of that shape passes through; a binary column becomes the ``bytes`` field
  and a string column the ``path`` field, built as one `StructArray` per batch.
- **ClassLabel.** HF's ``ClassLabel(names=...)`` stores ``int64`` codes. A string column
  named in ``class_labels`` is encoded against its names with one vectorized
  ``index_in`` per batch; an integer column is taken as codes already and range-checked.
  A value outside the names is refused rather than stored as a null label.

Nothing here knows about plans or the engine: callers hand in Arrow and get HF objects
back. The materialized path wraps the Arrow table in ``datasets.table.InMemoryTable``,
which shares its buffers; the iterable path yields examples through
``IterableDataset.from_generator``, the library's public constructor, which takes Python
rows, so each example is converted as the consumer pulls it.

Not yet verified against a live ``datasets`` install; see tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from batcher._internal.errors import PlanError
from batcher._internal.optional import require

__all__ = [
    "HuggingFaceSpec",
    "hf_dataset",
    "hf_iterable_dataset",
]

#: The storage type of HF's ``Image()`` feature.
_IMAGE_STORAGE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def _import_datasets() -> Any:
    return require(
        "datasets", feature="Dataset.to_huggingface()", provides="datasets", extra="huggingface"
    )


class HuggingFaceSpec:
    """Which columns become ``ClassLabel`` and ``Image`` features, and how to encode them.

    Args:
        schema: The Arrow schema of the batches that will be converted.
        class_labels: Column name → the label names, in code order.
        images: Columns to expose as ``Image()`` features.

    Raises:
        PlanError: If a named column is missing or has a type the feature cannot hold.
    """

    def __init__(
        self,
        schema: pa.Schema,
        *,
        class_labels: Mapping[str, Sequence[str]],
        images: Sequence[str],
    ) -> None:
        for name in [*class_labels, *images]:
            if schema.get_field_index(name) < 0:
                raise PlanError(
                    f"to_huggingface(): no column {name!r}; the columns are {schema.names}"
                )
        overlap = sorted(set(class_labels) & set(images))
        if overlap:
            raise PlanError(f"to_huggingface(): {overlap} named as both a label and an image")
        self._class_labels = {k: [str(n) for n in v] for k, v in class_labels.items()}
        self._images = list(images)
        for name in self._images:
            _check_image_type(name, schema.field(name).type)
        for name in self._class_labels:
            _check_label_type(name, schema.field(name).type)
        self.schema = pa.schema(
            [self._target_field(field) for field in schema],
            metadata=schema.metadata,
        )

    def _target_field(self, field: pa.Field) -> pa.Field:
        if field.name in self._class_labels:
            return pa.field(field.name, pa.int64(), nullable=True)
        if field.name in self._images:
            return pa.field(field.name, _IMAGE_STORAGE, nullable=True)
        return field

    def features(self) -> Any:
        """The ``datasets.Features`` describing the converted batches."""
        datasets = _import_datasets()
        features = datasets.Features.from_arrow_schema(self.schema)
        for name, names in self._class_labels.items():
            features[name] = datasets.ClassLabel(names=names)
        for name in self._images:
            features[name] = datasets.Image()
        return features

    def encode(self, batch: pa.RecordBatch | pa.Table) -> pa.Table:
        """Convert one batch's label and image columns to their HF storage types."""
        table = batch if isinstance(batch, pa.Table) else pa.Table.from_batches([batch])
        for name, names in self._class_labels.items():
            index = table.schema.get_field_index(name)
            codes = _label_codes(name, table.column(name).combine_chunks(), names)
            table = table.set_column(index, self.schema.field(name), codes)
        for name in self._images:
            index = table.schema.get_field_index(name)
            storage = _image_storage(table.column(name).combine_chunks())
            table = table.set_column(index, self.schema.field(name), storage)
        return table


def _check_image_type(name: str, arrow_type: pa.DataType) -> None:
    if _is_binary(arrow_type) or _is_string(arrow_type):
        return
    if pa.types.is_struct(arrow_type):
        fields = {arrow_type.field(i).name for i in range(arrow_type.num_fields)}
        if fields == {"bytes", "path"}:
            return
    raise PlanError(
        f"to_huggingface(): image column {name!r} must be binary (encoded image bytes), "
        f"string (a path), or struct<bytes, path>, not {arrow_type}"
    )


def _check_label_type(name: str, arrow_type: pa.DataType) -> None:
    if pa.types.is_dictionary(arrow_type):
        arrow_type = arrow_type.value_type
    if _is_string(arrow_type) or pa.types.is_integer(arrow_type):
        return
    raise PlanError(
        f"to_huggingface(): class-label column {name!r} must hold strings (label names) "
        f"or integers (codes), not {arrow_type}"
    )


def _is_binary(arrow_type: pa.DataType) -> bool:
    return pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type)


def _is_string(arrow_type: pa.DataType) -> bool:
    return pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type)


def _label_codes(name: str, column: pa.Array, names: list[str]) -> pa.Array:
    """`column` as ``int64`` ClassLabel codes, refusing a value outside `names`."""
    if pa.types.is_dictionary(column.type):
        column = column.dictionary_decode()
    if pa.types.is_integer(column.type):
        codes = column.cast(pa.int64())
        valid = pc.and_(pc.greater_equal(codes, 0), pc.less(codes, len(names)))
        if not pc.all(pc.fill_null(valid, True)).as_py():
            raise PlanError(
                f"to_huggingface(): class-label column {name!r} holds a code outside "
                f"0..{len(names) - 1}"
            )
        return codes
    codes = pc.index_in(column, value_set=pa.array(names, type=column.type)).cast(pa.int64())
    if codes.null_count > column.null_count:
        unmapped = pc.filter(column, pc.and_(pc.is_null(codes), pc.is_valid(column)))
        raise PlanError(
            f"to_huggingface(): class-label column {name!r} holds {unmapped[0].as_py()!r}, "
            f"which is not one of its label names {names}"
        )
    return codes


def _image_storage(column: pa.Array) -> pa.Array:
    """`column` as HF's ``struct<bytes: binary, path: string>`` image storage."""
    mask = column.is_null()
    if pa.types.is_struct(column.type):
        data = column.field("bytes").cast(pa.binary())
        path = column.field("path").cast(pa.string())
    elif _is_binary(column.type):
        data = column.cast(pa.binary())
        path = pa.nulls(len(column), pa.string())
    else:
        data = pa.nulls(len(column), pa.binary())
        path = column.cast(pa.string())
    return pa.StructArray.from_arrays([data, path], fields=list(_IMAGE_STORAGE), mask=mask)


def hf_dataset(table: pa.Table, spec: HuggingFaceSpec) -> Any:
    """A map-style ``datasets.Dataset`` over `table`, sharing its Arrow buffers.

    Args:
        table: The rows to hand over, in `spec`'s source schema.
        spec: The feature translation.

    Returns:
        A ``datasets.Dataset`` whose features are ``spec.features()``.
    """
    datasets = _import_datasets()
    from datasets.table import InMemoryTable

    features = spec.features()
    converted = spec.encode(table).cast(features.arrow_schema)
    return datasets.Dataset(InMemoryTable(converted), info=datasets.DatasetInfo(features=features))


def hf_iterable_dataset(
    batches: Callable[[], Iterable[pa.RecordBatch]], spec: HuggingFaceSpec
) -> Any:
    """A streaming ``datasets.IterableDataset`` that pulls `batches` as it is iterated.

    Args:
        batches: Called once per pass over the dataset; returns that pass's Arrow batches.
        spec: The feature translation.

    Returns:
        A ``datasets.IterableDataset`` whose features are ``spec.features()``.
    """
    datasets = _import_datasets()
    features = spec.features()
    target = features.arrow_schema

    def examples() -> Iterator[dict[str, Any]]:
        for batch in batches():
            yield from spec.encode(batch).cast(target).to_pylist()

    return datasets.IterableDataset.from_generator(examples, features=features)
