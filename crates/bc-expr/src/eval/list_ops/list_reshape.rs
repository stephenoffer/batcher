//! Reshaping `List`-column operations that change nesting depth — currently
//! `flatten` (`List<List<T>>` → `List<T>`). Split out of `eval/list.rs` to keep that
//! file within its line limit; the per-row reductions stay there.

use std::sync::Arc;

use arrow::array::ArrayRef;
use arrow::datatypes::DataType;

use crate::ExprError;

/// `flatten`: concatenate each row's inner lists of a `List<List<T>>` into one
/// `List<T>`, in order. Null inner lists are skipped; a null outer row stays null.
pub(crate) fn eval_flatten(
    list: &arrow::array::GenericListArray<i32>,
) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, AsArray, ListArray, UInt32Array};
    use arrow::buffer::OffsetBuffer;
    use arrow::compute::take;
    use arrow::datatypes::Field;

    let inner = list.values();
    let DataType::List(item_field) = inner.data_type() else {
        return Err(ExprError::ExpectedString {
            func: "list.flatten".into(),
            got: format!(
                "List<{}> (flatten needs a list of lists)",
                inner.data_type()
            ),
        });
    };
    let item_field = Arc::new(Field::new("item", item_field.data_type().clone(), true));
    let inner_list = inner.as_list::<i32>();
    let grandchild = inner_list.values();
    let outer_off = list.value_offsets();
    let inner_off = inner_list.value_offsets();

    // Every grandchild element is gathered at most once — pre-size to that upper bound.
    let mut take_idx: Vec<u32> = Vec::with_capacity(grandchild.len());
    let mut new_offsets: Vec<i32> = Vec::with_capacity(list.len() + 1);
    new_offsets.push(0);
    for i in 0..list.len() {
        if !list.is_null(i) {
            let (s, e) = (outer_off[i] as usize, outer_off[i + 1] as usize);
            for j in s..e {
                if inner_list.is_null(j) {
                    continue;
                }
                let (is_, ie) = (inner_off[j] as usize, inner_off[j + 1] as usize);
                take_idx.extend((is_..ie).map(|k| k as u32));
            }
        }
        new_offsets.push(take_idx.len() as i32);
    }
    let taken = take(grandchild.as_ref(), &UInt32Array::from(take_idx), None)?;
    let nulls = list.nulls().cloned();
    let out = ListArray::try_new(
        item_field,
        OffsetBuffer::new(new_offsets.into()),
        taken,
        nulls,
    )?;
    Ok(Arc::new(out))
}
