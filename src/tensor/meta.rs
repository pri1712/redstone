//declaration for tensor metadata

use std::alloc::Layout;

pub struct TensorMeta {
    pub dtype: DType, //data type stored in the tensor
    pub shape: Vec<usize>, //shape of tensor (dimensions)
    pub layout: StorageLayout //row major or column major in memory for faster access.
}

impl TensorMeta {
    pub fn new(dtype: DType, shape: Vec<usize>,layout: StorageLayout) -> Self {
        //creates a new tensor meta instance.
        let tensor_meta = TensorMeta{dtype, shape, layout};
        match tensor_meta.validate() {
            Ok(_) => {}
            Err(_) => {}
        };
        tensor_meta
    }

    pub fn number_elements(&self) -> Result<usize,&'static str> {
        //find the number of elements in our tensor using this.
        let mut total_elements = 1usize;
        for dim in &self.shape {
            total_elements = total_elements.checked_mul(*dim).
                ok_or("Shape overflow")?;
        }
        Ok(total_elements)
    }

    pub fn total_byte_size(&self) -> Result<usize,&'static str> {
        let total_elements = self.number_elements()?;
        total_elements
            .checked_mul(self.dtype.size_bytes())
            .ok_or("tensor byte size overflow")
    }
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
    ColMajor,
}