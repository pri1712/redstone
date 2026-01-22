//declaration for tensor metadata

use std::alloc::Layout;

pub struct TensorMeta {
    pub dtype: DType, //data type stored in the tensor
    pub shape: Vec<usize>, //shape of tensor
    pub stride: Vec<usize>, //the size of this array and shape array is the same.
    pub layout: StorageLayout //row major or column major in memory for faster access.
}

impl TensorMeta {
    pub fn new(dtype: DType, shape: Vec<usize>, stride: Vec<usize>,layout: StorageLayout) -> Self {
        //creates a new tensor meta instance.
        let tensor_meta = TensorMeta{dtype, shape, stride, layout};
        tensor_meta
    }

}

pub enum DType{
    F32,
    F64,
    I32,
    I64
}

pub enum StorageLayout{
    RowMajor,
    ColMajor,
}