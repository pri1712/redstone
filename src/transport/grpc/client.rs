use std::sync::Arc;
use tonic::Code;
use tonic::transport::Channel;
use crate::proto;
use crate::proto::{GetRequest,PutRequest,DeleteRequest,StatsRequest};
use crate::proto::red_stone_client::RedStoneClient;
use crate::tensor::meta::{DType, StorageLayout, TensorMeta};
use crate::tensor::tensor::Tensor;

#[derive(Clone)]
pub struct RemoteCacheClient {
    client: RedStoneClient<Channel>
}

impl RemoteCacheClient {
    pub async fn connect(addr: String) -> Result<Self, Box<dyn std::error::Error>> {
        let url = if addr.starts_with("http://") || addr.starts_with("https://") {
            addr
        } else {
            format!("http://{}", addr)
        };
        let client = RedStoneClient::connect(url).await?;
        Ok(Self { client })
    }

    pub async fn get(&mut self, key: String) -> Result<Option<Arc<Tensor>>, String> {
        let request = tonic::Request::new(GetRequest {
            key,
        });

        match self.client.get(request).await {
            Ok(response) => {
                let get_response = response.into_inner();
                let proto_meta = get_response.meta.ok_or("Invalid metadata in response")?;
                let meta = proto_to_meta(&proto_meta)?;
                let tensor = Tensor::new(meta,get_response.data)
                    .map_err(|e| format!("Invalid tensor data: {:?}", e))?;
                Ok(Some(Arc::new(tensor)))
            }
            Err(status) => {
                match status.code() {
                    Code::NotFound => Ok(None),
                    _ => Err(format!("Get failed: {}", status.message())),
                }
            }
        }
    }

    pub async fn put(&mut self, key: String, meta: TensorMeta, data: Vec<u8>) -> Result<(), String> {
        let proto_meta = proto::TensorMeta { dtype: dtype_to_proto(meta.dtype()),
            shape: meta.shape().iter().map(|&s| s as u64).collect(),
            layout: layout_to_proto(meta.layout()),
        };
        let request = tonic::Request::new(PutRequest {
            //deep copy has cpu overhead
            key: key.clone(),
            meta: Some(proto_meta),
            //deep copy has cpu overhead
            data,
        });

        match self.client.put(request).await {
            Ok(_) => Ok(()),
            Err(status) => {
                match status.code() {
                    Code::AlreadyExists => {
                        Err(format!("Key '{}' already exists", key))
                    }
                    Code::ResourceExhausted => {
                        Err("Cache is full".to_string())
                    }
                    Code::InvalidArgument => {
                        Err(format!("Invalid tensor: {}", status.message()))
                    }
                    _ => {
                        Err(format!("Put failed: {}", status.message()))
                    }
                }
            }
        }
    }

    pub async fn delete(&mut self, key: String) -> Result<(), String> {
        let request = tonic::Request::new(DeleteRequest {
            key,
        });
        match self.client.delete(request).await {
            Ok(_) => Ok(()),
            Err(status) => {
                Err(format!("Delete failed: {}", status.message()))
            }
        }
    }

    pub async fn get_stats(&mut self) -> Result<CacheStats,String> {
        let request = tonic::Request::new(StatsRequest {});

        match self.client.get_stats(request).await {
            Ok(response) => {
                let response = response.into_inner();
                Ok( CacheStats {
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
            Err(status) => { Err(format!("Get statistics failed: {}", status.message())) }
        }
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

fn proto_to_dtype(dtype: i32) -> Result<DType, String> {
    match proto::DType::try_from(dtype) {
        Ok(proto::DType::F32) => Ok(DType::F32),
        Ok(proto::DType::F64) => Ok(DType::F64),
        Ok(proto::DType::I32) => Ok(DType::I32),
        Ok(proto::DType::I64) => Ok(DType::I64),
        Ok(proto::DType::U8) => Ok(DType::U8),
        _ => Err("Invalid dtype".to_string()),
    }
}

fn proto_to_layout(layout: i32) -> Result<StorageLayout, String> {
    match proto::StorageLayout::try_from(layout) {
        Ok(proto::StorageLayout::RowMajor) => Ok(StorageLayout::RowMajor),
        Ok(proto::StorageLayout::ColumnMajor) => Ok(StorageLayout::ColumnMajor),
        _ => Err("Invalid layout".to_string()),
    }
}

fn proto_to_meta(proto_meta: &proto::TensorMeta) -> Result<TensorMeta, String> {
    let dtype = proto_to_dtype(proto_meta.dtype)?;
    let layout = proto_to_layout(proto_meta.layout)?;
    let shape: Vec<usize> = proto_meta.shape.iter().map(|&s| s as usize).collect();

    TensorMeta::new(dtype, shape, layout)
        .map_err(|e| format!("Invalid metadata: {:?}", e))
}
