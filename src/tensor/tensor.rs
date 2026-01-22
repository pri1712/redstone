use crate::tensor::meta::TensorMeta;

//define the full tensor object to be stored
struct Tensor {
    metadata: TensorMeta,
    data: Vec<u8>
}

impl Tensor {
    pub fn new(metadata: TensorMeta,data: Vec<u8>) -> Result<Self, &'static str> {
        let data_len = data.len();
        let expected = metadata.total_byte_size()?;
        if data_len != expected {
            return Err("Data length does not match expected length (calculated from tensor metadata)");
        }
        Ok(Self { metadata, data })
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::tensor::meta::{DType, StorageLayout, TensorMeta};
    #[test]
    fn test_tensor_new_valid() {
        let meta = TensorMeta::new(
            DType::F32,
            vec![4,4],
            StorageLayout::RowMajor
        ).unwrap();
        //the meta data is 64 bytes in size, this one should be too.
        let data = vec![0u8; 64];
        let tensor = Tensor::new(meta, data);
        assert!(tensor.is_ok());
    }

    #[test]
    fn test_tensor_new_invalid() {
        let meta = TensorMeta::new(
            DType::F32,
            vec![4,4],
            StorageLayout::RowMajor
        ).unwrap();

        let data = vec![0u8; 63];
        let tensor = Tensor::new(meta, data);
        assert!(tensor.is_err());
    }
}