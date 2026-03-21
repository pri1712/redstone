use std::sync::Arc;
use bytes::Bytes;
use tonic::Code;
use tonic::transport::Channel;
use crate::error::cache_error::CacheError;
use crate::error::client_error::ClientError;
use crate::proto;
use crate::proto::{GetRequest,PutRequest,DeleteRequest,StatsRequest};
use crate::proto::red_stone_client::RedStoneClient;
use crate::tensor::meta::{DType, StorageLayout, TensorMeta};
use crate::tensor::tensor::Tensor;

#[derive(Clone)]
pub struct RemoteCacheClient {
    // each client-server connection has a separate channel. total number of channels are equal
    // to the number of servers
    client: RedStoneClient<Channel>
}

impl RemoteCacheClient {
    pub async fn connect(addr: String) -> Result<Self, ClientError> {
        let url = if addr.starts_with("http://") || addr.starts_with("https://") {
            addr
        } else {
            format!("http://{}", addr)
        };
        let client = RedStoneClient::connect(url).await?;
        Ok(Self { client })
    }

    pub async fn get(&mut self, key: String) -> Result<Option<Arc<Tensor>>, ClientError> {
        let request = tonic::Request::new(GetRequest {
            key,
        });

        match self.client.get(request).await {
            Ok(response) => {
                let get_response = response.into_inner();
                let proto_meta = get_response
                    .meta
                    .ok_or_else(|| ClientError::ServerError("Missing metadata".into()))?;
                let meta = proto_to_meta(&proto_meta)?;
                let get_response_bytes = Bytes::from(get_response.data);
                let tensor = Tensor::new(meta, get_response_bytes)
                    .map_err(|_| ClientError::ServerError("Invalid tensor data".into()))?;
                Ok(Some(Arc::new(tensor)))
            }
            Err(status) => {
                match status.code() {
                    Code::NotFound => Ok(None),
                    Code::Internal => Err(ClientError::ServerError("Internal server error".to_string())),
                    Code::Aborted => Err(ClientError::GrpcStatus(status)),
                    _ => Err(ClientError::ServerError("Unknown error".to_string())),
                }
            }
        }
    }

    pub async fn put(&mut self, key: String, meta: TensorMeta, data: Vec<u8>) -> Result<(), ClientError> {
        let proto_meta = proto::TensorMeta { dtype: dtype_to_proto(meta.dtype()),
            shape: meta.shape().iter().map(|&s| s as u64).collect(),
            layout: layout_to_proto(meta.layout()),
        };
        let request = tonic::Request::new(PutRequest {
            //deep copy has cpu overhead
            key,
            meta: Some(proto_meta),
            //deep copy has cpu overhead
            data,
        });

        self.client.put(request).await?;
        Ok(())
    }

    pub async fn delete(&mut self, key: String) -> Result<(), ClientError> {
        let request = tonic::Request::new(DeleteRequest {
            key,
        });
        self.client.delete(request).await?;
        Ok(())
    }

    pub async fn get_stats(&mut self) -> Result<CacheStats, ClientError> {
        let request = tonic::Request::new(StatsRequest {});
        let response = self.client.get_stats(request).await?.into_inner();

        Ok(CacheStats {
            entries: response.entries,
            memory_used: response.memory_used,
            memory_limit: response.memory_limit,
            hits: response.hits,
            misses: response.misses,
            evictions: response.evictions,
            hit_rate: response.hit_rate,
            memory_utilization: response.memory_utilization,
        })
    }

}

pub struct CacheStats {
    pub entries: u64,
    pub memory_used: u64,
    pub memory_limit: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub hit_rate: f64,
    pub memory_utilization: f64,
}

/// utility functions for conversion
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

fn proto_to_dtype(dtype: i32) -> Result<DType, ClientError> {
    proto::DType::try_from(dtype)
        .map_err(|_| ClientError::ServerError("Invalid dtype".into()))
        .and_then(|d| match d {
    proto::DType::F32 => Ok(DType::F32),
    proto::DType::F64 => Ok(DType::F64),
    proto::DType::I32 => Ok(DType::I32),
    proto::DType::I64 => Ok(DType::I64),
    proto::DType::U8 => Ok(DType::U8),
    _ => Err(ClientError::ServerError("Invalid dtype".into())),
    })
}

fn proto_to_layout(layout: i32) -> Result<StorageLayout, ClientError> {
    proto::StorageLayout::try_from(layout)
        .map_err(|_| ClientError::ServerError("Invalid layout".into()))
        .and_then(|l| match l {
            proto::StorageLayout::RowMajor => Ok(StorageLayout::RowMajor),
            proto::StorageLayout::ColumnMajor => Ok(StorageLayout::ColumnMajor),
            _ => Err(ClientError::ServerError("Invalid layout".into())),
        })
}

fn proto_to_meta(proto_meta: &proto::TensorMeta) -> Result<TensorMeta, ClientError> {
    let dtype = proto_to_dtype(proto_meta.dtype)?;
    let layout = proto_to_layout(proto_meta.layout)?;
    let shape: Vec<usize> = proto_meta.shape.iter().map(|&s| s as usize).collect();

    TensorMeta::new(dtype, shape, layout)
        .map_err(|_| ClientError::ServerError("Invalid metadata".into()))
}