use std::sync::Arc;
use tonic::{transport::Server, Request, Response, Status, Code};
use crate::proto::{DeleteRequest, DeleteResponse, GetRequest, GetResponse, PutRequest, PutResponse, StatsRequest, StatsResponse};
use crate::proto::red_stone_server::{RedStone, RedStoneServer};
use crate::proto;

use crate::TensorCache;
use crate::tensor::meta::{DType, StorageLayout, TensorMeta};
use crate::CacheError;

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
#[tonic::async_trait]
impl RedStone for CacheServer {
    /// Implementation for the GET method for the gRPC server.
    async fn get(&self, request: Request<GetRequest>) -> Result<Response<GetResponse>, Status> {
        let get_request = request.into_inner();
        if let Some(tensor) = self.cache.get(&get_request.key) {
            let meta = meta_to_proto(tensor.get_metadata());
            Ok(Response::new(GetResponse{
                meta: Some(meta),
                data: tensor.get_data().to_vec(),
            }))
        } else {
            Err(Status::not_found(format!("Key not found in cache: {}", get_request.key)))
        }
    }

    async fn put(&self, request: Request<PutRequest>) -> Result<Response<PutResponse>, Status> {
        let put_request = request.into_inner();
        let proto_meta = put_request.meta.ok_or_else(|| {
            Status::invalid_argument("Missing tensor metadata")
        })?;
        let meta = proto_to_meta(&proto_meta)?;
        match self.cache.put(put_request.key.clone(),meta,put_request.data) {
            Ok(()) => Ok(Response::new(PutResponse{})),
            Err(e) => {
                match e {
                    CacheError::KeyAlreadyExists => {
                        Err(Status::new(
                            Code::AlreadyExists,
                            format!("Key already exists: {}", put_request.key)
                        ))
                    }
                    CacheError::InvalidTensor => {
                        Err(Status::invalid_argument("Invalid tensor data"))
                    }
                    CacheError::InvalidSize => {
                        Err(Status::invalid_argument("Invalid tensor size"))
                    }
                    CacheError::OutOfMemory => {
                        Err(Status::resource_exhausted("Cache is full"))
                    }
                    CacheError::InvalidTensorMetadata => {
                        Err(Status::invalid_argument("Invalid tensor metadata"))
                    }
                }
            }
        }

    }
    async fn delete(&self, request: Request<DeleteRequest>) -> Result<Response<DeleteResponse>, Status> {
       todo!()
    }

    async fn get_stats(&self, request: Request<StatsRequest>) -> Result<Response<StatsResponse>, Status> {
        todo!()
    }
}

pub async fn start_server(addr: String, cache_size: u64) -> Result<(), Box<dyn std::error::Error>> {
    let addr = addr.parse()?;
    if cache_size == 0 {

    }
    let cache = Arc::new(TensorCache::new(cache_size)?);
    let server = CacheServer::new(cache);


    Server::builder()
        .add_service(RedStoneServer::new(server))
        .serve(addr)
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_conversion() {
        let dtype = DType::F32;
        let proto = dtype_to_proto(&dtype);
        assert_eq!(proto, proto::DType::F32 as i32);

        let back = proto_to_dtype(proto).unwrap();
        assert_eq!(back, DType::F32);
    }

    #[test]
    fn test_layout_conversion() {
        let layout = StorageLayout::RowMajor;
        let proto = layout_to_proto(&layout);
        assert_eq!(proto, proto::StorageLayout::RowMajor as i32);

        let back = proto_to_layout(proto).unwrap();
        assert_eq!(back, StorageLayout::RowMajor);
    }
}
