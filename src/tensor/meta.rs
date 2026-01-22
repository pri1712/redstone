//declaration for tensor metadata

struct TensorMeta {
    dtype: String, //data type stored in the tensor
    shape: Vec<usize>, //shape of tensor
    stride: Vec<usize>, //the size of this array and shape array is the same.

}