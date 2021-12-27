use std::sync::Arc;

use crate::array::*;
use crate::bitmap::*;
use crate::buffer::*;
use crate::datatypes::*;
use crate::error::*;

#[derive(Debug)]
pub struct FixedItemsUtf8Dictionary {
    data_type: DataType,
    keys: MutablePrimitiveArray<i32>,
    values: Utf8Array<i32>,
}

impl FixedItemsUtf8Dictionary {
    pub fn with_capacity(values: Utf8Array<i32>, capacity: usize) -> Self {
        Self {
            data_type: DataType::Dictionary(
                IntegerType::Int32,
                Box::new(values.data_type().clone()),
            ),
            keys: MutablePrimitiveArray::<i32>::with_capacity(capacity),
            values,
        }
    }

    pub fn push_valid(&mut self, key: i32) {
        self.keys.push(Some(key))
    }

    /// pushes a null value
    pub fn push_null(&mut self) {
        self.keys.push(None)
    }
}

impl MutableArray for FixedItemsUtf8Dictionary {
    fn len(&self) -> usize {
        self.keys.len()
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        self.keys.validity()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(DictionaryArray::from_data(
            std::mem::take(&mut self.keys).into(),
            Arc::new(self.values.clone()),
        ))
    }

    fn as_arc(&mut self) -> Arc<dyn Array> {
        Arc::new(DictionaryArray::from_data(
            std::mem::take(&mut self.keys).into(),
            Arc::new(self.values.clone()),
        ))
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[inline]
    fn push_null(&mut self) {
        self.push_null()
    }

    fn shrink_to_fit(&mut self) {
        todo!();
    }
}
