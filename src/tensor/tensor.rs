use bytes::Bytes;
use crate::tensor::meta::TensorMeta;

/// Define the full tensor object to be stored
pub struct Tensor {
    metadata: TensorMeta,
    data: Bytes,
}

impl Tensor {
    pub fn new(metadata: TensorMeta, data: Bytes) -> Result<Self, &'static str> {
        let data_len = data.len();
        let expected = metadata.total_byte_size()?;
        if data_len != expected {
            return Err("Data length does not match expected length (calculated from tensor metadata)");
        }
        Ok(Self { metadata, data })
    }

    /// Returns size in bytes of the tensor object
    pub fn byte_size(&self) -> usize {
        self.metadata.total_byte_size().unwrap()
    }

    pub fn get_metadata(&self) -> &TensorMeta {
        &self.metadata
    }

    /// Returns a reference to the underlying Bytes
    pub fn get_data(&self) -> &Bytes {
        &self.data
    }

    /// Returns a clone of the Bytes (cheap - just increments refcount)
    pub fn get_data_cloned(&self) -> Bytes {
        self.data.clone()
    }

    /// Consumes the tensor and returns the Bytes
    pub fn into_data(self) -> Bytes {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::meta::{DType, StorageLayout};

    #[test]
    fn test_tensor_new_valid() {
        let meta = TensorMeta::new(
            DType::F32,
            vec![4, 4],
            StorageLayout::RowMajor
        ).unwrap();

        let data = Bytes::from(vec![0u8; 64]);
        let tensor = Tensor::new(meta, data);
        assert!(tensor.is_ok());
    }

    #[test]
    fn test_tensor_new_invalid() {
        let meta = TensorMeta::new(
            DType::F32,
            vec![4, 4],
            StorageLayout::RowMajor
        ).unwrap();

        // Convert Vec to Bytes
        let data = Bytes::from(vec![0u8; 63]);
        let tensor = Tensor::new(meta, data);
        assert!(tensor.is_err());
    }

    #[test]
    fn test_tensor_data_clone_is_cheap() {
        let meta = TensorMeta::new(
            DType::F32,
            vec![4, 4],
            StorageLayout::RowMajor
        ).unwrap();

        let data = Bytes::from(vec![0u8; 64]);
        let tensor = Tensor::new(meta, data).unwrap();

        let data1 = tensor.get_data_cloned();
        let data2 = tensor.get_data_cloned();

        assert_eq!(data1.len(), data2.len());
    }

    #[test]
    fn test_tensor_into_data() {
        let meta = TensorMeta::new(
            DType::F32,
            vec![4, 4],
            StorageLayout::RowMajor
        ).unwrap();

        let data = Bytes::from(vec![0u8; 64]);
        let tensor = Tensor::new(meta, data).unwrap();

        let data = tensor.into_data();
        assert_eq!(data.len(), 64);
    }
}