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
            return Err("Data length does not match expected length (calculated from tensor metadata");
        }
        Ok(Self { metadata, data })
    }
}

mod tests{
    use super::*;

    #[test]
    fn test_tensor_new() {
        
    }
}