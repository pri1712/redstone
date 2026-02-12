use std::sync::Arc;
use crate::TensorCache;
use tonic::{transport::Server, Request, Response, Status, Code};
use crate::tensor::meta::{DType,StorageLayout,TensorMeta};

pub mod proto {
    tonic::include_proto!("redstone");
}
pub struct CacheServer {
    cache: Arc<TensorCache>,
}

impl CacheServer {
    pub fn new(cache: Arc<TensorCache>) -> Self {
        Self { cache }
    }
}

//convert from proto data types to rust defined data types.
fn proto_to_dtype(dtype: i32) -> Result<DType, Status> {
    match proto::DType::try_from(dtype) {
        Ok(proto::DType::F32) => Ok(DType::F32),
        Ok(proto::DType::F64) => Ok(DType::F64),
        Ok(proto::DType::I32) => Ok(DType::I32),
        Ok(proto::DType::I64) => Ok(DType::I64),
        Ok(proto::DType::U8) => Ok(DType::U8),
        _ => Err(Status::invalid_argument("Invalid dtype")),
    }
}

fn proto_to_layout(layout: i32) -> Result<StorageLayout, Status> {
    match proto::StorageLayout::try_from(layout) {
        Ok(proto::StorageLayout::RowMajor) => Ok(StorageLayout::RowMajor),
        Ok(proto::StorageLayout::ColumnMajor) => Ok(StorageLayout::ColumnMajor),
        _ => Err(Status::invalid_argument("Invalid storage layout")),
    }
}

fn proto_to_meta(proto_meta: &proto::TensorMeta) -> Result<TensorMeta, Status> {
    let dtype = proto_to_dtype(proto_meta.dtype)?;
    let layout = proto_to_layout(proto_meta.layout)?;
    let shape: Vec<usize> = proto_meta.shape.iter().map(|&s| s as usize).collect();

    TensorMeta::new(dtype, shape, layout)
        .map_err(|e| Status::invalid_argument(format!("Invalid tensor metadata: {:?}", e)))
}

//convert from rust defined data types to proto defined data types.
fn dtype_to_proto(dtype: &DType) -> i32 {
    match dtype {
        DType::F32 => proto::DType::F32 as i32,
        DType::F64 => proto::DType::F64 as i32,
        DType::I32 => proto::DType::I32 as i32,
        DType::I64 => proto::DType::I64 as i32,
        DType::U8 => proto::DType::U8 as i32,
    }
}

fn layout_to_proto(layout: &StorageLayout) -> i32 {
    match layout {
        StorageLayout::RowMajor => proto::StorageLayout::RowMajor as i32,
        StorageLayout::ColumnMajor => proto::StorageLayout::ColumnMajor as i32,
    }
}

fn meta_to_proto(meta: &TensorMeta) -> proto::TensorMeta {
    proto::TensorMeta {
        dtype: dtype_to_proto(meta.dtype()),
        shape: meta.shape().iter().map(|&s| s as u64).collect(),
        layout: layout_to_proto(meta.layout()),
    }
}

//server method definitions
