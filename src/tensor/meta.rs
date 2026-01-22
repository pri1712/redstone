//declaration for tensor metadata

pub struct TensorMeta {
    dtype: DType, //data type stored in the tensor
    shape: Vec<usize>, //shape of tensor (dimensions)
    layout: StorageLayout //row major or column major in memory for faster access.
}

impl TensorMeta {
    /***
    Constructor for TensorMeta object
     */
    pub fn new(dtype: DType, shape: Vec<usize>,layout: StorageLayout) -> Result<Self,&'static str> {
        //creates a new tensor meta instance.
        let tensor_meta = TensorMeta{dtype, shape, layout};
        tensor_meta.validate()?;
        Ok(tensor_meta)
    }

    /***
    Calculates the number of elements in a given tensor using its shape dimensions
     */
    pub fn num_elements(&self) -> Result<usize,&'static str> {
        //find the number of elements in our tensor using this.
        let mut total_elements = 1usize;
        for dim in &self.shape {
            total_elements = total_elements.checked_mul(*dim).
                ok_or("Shape overflow")?;
        }
        Ok(total_elements)
    }

    /***
    Returns the total size of the object. It is defined by number of elements * size in bytes of each element
     */
    pub fn total_byte_size(&self) -> Result<usize,&'static str> {
        let total_elements = self.num_elements()?;
        total_elements
            .checked_mul(self.dtype.size_bytes())
            .ok_or("tensor byte size overflow")
    }
    /***
    Validates whether a given tensor is structurally correct by:
    1. verifies it exists and has proper dimensionality
    2. verifies that none of the dimensions are 0
    3. verifies that there is no overflow in regard to usize and its total size in bytes.
     */
    pub fn validate(&self) -> Result<(),&'static str> {
        //all dimensions are non zero, non zero shape length, number of elements are equal to
        // product(shape) * size_in_bytes(dtype)

        if self.shape.is_empty() {
           return Err("Shape cannot be empty")
        }
        for dimension in &self.shape {
            if *dimension == 0 {
                return Err("Shape cannot be zero")
            }
        }
        self.total_byte_size()?;
        Ok(())
    }

    //getters for all the params.
    pub fn dtype(&self) -> &DType { &self.dtype }
    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn layout(&self) -> &StorageLayout { &self.layout }

}

pub enum DType{
    F32,
    F64,
    I32,
    I64
}

impl DType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::I32 => 4,
            DType::I64 => 8,
        }
    }
}

pub enum StorageLayout{
    RowMajor,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_valid() {
        let meta = TensorMeta::new(
            DType::F32,
            vec![4, 4],
            StorageLayout::RowMajor
        ).unwrap();

        assert_eq!(meta.total_byte_size().unwrap(), 64);
    }

    #[test]
    fn test_meta_zero_dim_fails() {
        let meta = TensorMeta::new(
            DType::F32,
            vec![4, 0],
            StorageLayout::RowMajor
        );

        assert!(meta.is_err());
    }

    #[test]
    fn test_meta_overflow_fails() {
        let meta = TensorMeta::new(
            DType::F64,
            vec![usize::MAX, 2],
            StorageLayout::RowMajor
        );

        assert!(meta.is_err());
    }
}