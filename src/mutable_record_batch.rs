use crate::array::{
    Array, ListArray, MutableArray, MutableBinaryArray, MutableBooleanArray,
    MutableFixedSizeBinaryArray, MutablePrimitiveArray, MutableUtf8Array, Offset,
};
use crate::bitmap::MutableBitmap;
use crate::buffer::MutableBuffer;
use crate::datatypes::{DataType, PhysicalType, Schema};
use crate::error::{ArrowError, Result};
use crate::record_batch::RecordBatch;
use std::sync::Arc;

pub struct MutableRecordBatch {
    arrays: Vec<Box<dyn MutableArray>>,
    target_batch_size: usize,
    slots_available: usize,
    schema: Arc<Schema>,
}

impl MutableRecordBatch {
    pub fn new(target_batch_size: usize, schema: Arc<Schema>) -> Result<Self> {
        let arrays = new_arrays(&schema, target_batch_size)?;
        Ok(Self {
            arrays,
            target_batch_size,
            slots_available: target_batch_size,
            schema,
        })
    }

    pub fn output_and_reset(&mut self) -> Result<RecordBatch> {
        let result = self.output();
        let mut new = new_arrays(&self.schema, self.target_batch_size)?;
        self.arrays.append(&mut new);
        result
    }

    pub fn output(&mut self) -> Result<RecordBatch> {
        let result = make_batch(self.schema.clone(), self.arrays.drain(..).collect());
        self.slots_available = self.target_batch_size;
        result
    }

    pub fn append(&mut self, size: usize) {
        if size <= self.slots_available {
            self.slots_available -= size;
        } else {
            self.slots_available = 0;
        }
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.slots_available == 0
    }

    pub fn arrays_mut(&mut self) -> &mut [Box<dyn MutableArray>] {
        &mut self.arrays
    }

    pub fn arrays(&self) -> &[Box<dyn MutableArray>] {
        &self.arrays[..]
    }
}

fn new_arrays(schema: &Arc<Schema>, batch_size: usize) -> Result<Vec<Box<dyn MutableArray>>> {
    let arrays: Vec<Box<dyn MutableArray>> = schema
        .fields()
        .iter()
        .map(|field| {
            let dt = field.data_type.to_logical_type();
            make_mutable(dt, batch_size)
        })
        .collect::<Result<_>>()?;
    Ok(arrays)
}

fn make_batch(schema: Arc<Schema>, mut arrays: Vec<Box<dyn MutableArray>>) -> Result<RecordBatch> {
    let columns = arrays.iter_mut().map(|array| array.as_arc()).collect();
    RecordBatch::try_new(schema, columns)
}

fn make_mutable(data_type: &DataType, capacity: usize) -> Result<Box<dyn MutableArray>> {
    Ok(match data_type.to_physical_type() {
        PhysicalType::Boolean => {
            Box::new(MutableBooleanArray::with_capacity(capacity)) as Box<dyn MutableArray>
        }
        PhysicalType::Primitive(primitive) => {
            with_match_primitive_type!(primitive, |$T| {
                Box::new(MutablePrimitiveArray::<$T>::with_capacity(capacity).to(data_type.clone()))
                    as Box<dyn MutableArray>
            })
        }
        PhysicalType::Binary => {
            Box::new(MutableBinaryArray::<i32>::with_capacity(capacity)) as Box<dyn MutableArray>
        }
        PhysicalType::Utf8 => {
            Box::new(MutableUtf8Array::<i32>::with_capacity(capacity)) as Box<dyn MutableArray>
        }
        _ => match data_type {
            DataType::List(inner) => {
                let values = make_mutable(inner.data_type(), 0)?;
                Box::new(DynMutableListArray::<i32>::new_with_capacity(
                    values, capacity,
                )) as Box<dyn MutableArray>
            }
            DataType::FixedSizeBinary(size) => Box::new(MutableFixedSizeBinaryArray::with_capacity(
                *size as usize,
                capacity,
            )) as Box<dyn MutableArray>,
            _ => {
                return Err(ArrowError::NotYetImplemented(format!(
                    "making mutable of type {} is not implemented yet",
                    data_type
                )));
            }
        },
    })
}

/// Auxiliary struct
#[derive(Debug)]
pub struct DynMutableListArray<O: Offset> {
    data_type: DataType,
    offsets: MutableBuffer<O>,
    values: Box<dyn MutableArray>,
    validity: Option<MutableBitmap>,
}

impl<O: Offset> DynMutableListArray<O> {
    pub fn new_from(values: Box<dyn MutableArray>, data_type: DataType, capacity: usize) -> Self {
        let mut offsets = MutableBuffer::<O>::with_capacity(capacity + 1);
        offsets.push(O::default());
        assert_eq!(values.len(), 0);
        ListArray::<O>::get_child_field(&data_type);
        Self {
            data_type,
            offsets,
            values,
            validity: None,
        }
    }

    /// Creates a new [`MutableListArray`] from a [`MutableArray`] and capacity.
    pub fn new_with_capacity(values: Box<dyn MutableArray>, capacity: usize) -> Self {
        let data_type = ListArray::<O>::default_datatype(values.data_type().clone());
        Self::new_from(values, data_type, capacity)
    }

    /// The values
    pub fn mut_values(&mut self) -> &mut dyn MutableArray {
        self.values.as_mut()
    }

    #[inline]
    pub fn try_push_valid(&mut self) -> Result<()> {
        let size = self.values.len();
        let size = O::from_usize(size).ok_or(ArrowError::Overflow)?;
        assert!(size >= *self.offsets.last().unwrap());

        self.offsets.push(size);
        if let Some(validity) = &mut self.validity {
            validity.push(true)
        }
        Ok(())
    }

    #[inline]
    fn push_null(&mut self) {
        self.offsets.push(self.last_offset());
        match &mut self.validity {
            Some(validity) => validity.push(false),
            None => self.init_validity(),
        }
    }

    #[inline]
    fn last_offset(&self) -> O {
        *self.offsets.last().unwrap()
    }

    fn init_validity(&mut self) {
        let len = self.offsets.len() - 1;

        let mut validity = MutableBitmap::new();
        validity.extend_constant(len, true);
        validity.set(len - 1, false);
        self.validity = Some(validity)
    }
}

impl<O: Offset> MutableArray for DynMutableListArray<O> {
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        self.validity.as_ref()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(ListArray::from_data(
            self.data_type.clone(),
            std::mem::take(&mut self.offsets).into(),
            self.values.as_arc(),
            std::mem::take(&mut self.validity).map(|x| x.into()),
        ))
    }

    fn as_arc(&mut self) -> Arc<dyn Array> {
        Arc::new(ListArray::from_data(
            self.data_type.clone(),
            std::mem::take(&mut self.offsets).into(),
            self.values.as_arc(),
            std::mem::take(&mut self.validity).map(|x| x.into()),
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
