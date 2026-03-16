use std::sync::Arc;
use std::mem::size_of;
use bytes::Bytes;

pub mod tensor;
pub mod cache;
pub mod transport;
pub mod cluster;
pub mod error;

use crate::cache::lru_cache::Cache;
use crate::tensor::meta::{DType, StorageLayout, TensorMeta};
use crate::tensor::tensor::Tensor;
use crate::cache::lru_cache::CacheStats;
use crate::error::cache_error::CacheError;

pub mod proto {
    tonic::include_proto!("redstone");
}

pub struct TensorCache {
    cache: Cache,
}

impl TensorCache {
    pub fn new(max_cache_size: u64) -> Result<Self, CacheError> {
        Ok(Self {
            cache: Cache::new(max_cache_size)?,
        })
    }

    /// Inserts a tensor into the cache.
    /// It guarantees:
    /// 1. Immutable writes
    /// 2. Tensor validation before insertion, preventing corrupted writes
    /// 3. Atomic inserts
    pub fn put(&self, key: String, meta: TensorMeta, data: Bytes) -> Result<(), CacheError> {
        let tensor = Tensor::new(meta, data)
            .map_err(|_| CacheError::InvalidTensor)?;
        self.cache.put(key, tensor)
    }

    /// Retrieves a tensor by key.
    /// It guarantees:
    /// 1. Atomic reads
    /// 2. Idempotent reads
    pub fn get(&self, key: &str) -> Option<Arc<Tensor>> {
        self.cache.get(key)
    }

    /// Deletes a key value pair and returns the deleted tensor.
    pub fn delete(&self, key: &str) -> Option<Arc<Tensor>> {
        self.cache.delete(key)
    }

    pub fn get_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Put method for f32 data type. It internally implements the core put method.
    pub fn put_f32(&self, key: String, shape: Vec<usize>, data: Vec<f32>) -> Result<(), CacheError> {
        let meta = TensorMeta::new(DType::F32, shape, StorageLayout::RowMajor)
            .map_err(|_| CacheError::InvalidTensorMetadata)?;

        let bytes: Bytes = unsafe {
            let raw_bytes = std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * size_of::<f32>()
            );
            Bytes::copy_from_slice(raw_bytes)
        };

        self.put(key, meta, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_valid_meta() -> TensorMeta {
        TensorMeta::new(
            DType::F32,
            vec![2, 2],
            StorageLayout::RowMajor,
        )
            .unwrap()
    }

    #[test]
    fn put_then_get_works() {
        let cache = TensorCache::new(128).unwrap();

        let meta = make_valid_meta();
        let data = Bytes::from(vec![0u8; 16]);

        cache.put("t1".to_string(), meta, data).unwrap();

        let tensor = cache.get("t1");
        assert!(tensor.is_some());
    }

    #[test]
    fn invalid_tensor_is_rejected() {
        let cache = TensorCache::new(128).unwrap();

        let meta = make_valid_meta();
        let data = Bytes::from(vec![0u8; 15]);

        let result = cache.put("bad".to_string(), meta, data);
        assert_eq!(result, Err(CacheError::InvalidTensor));
    }

    #[test]
    fn duplicate_key_is_rejected() {
        let cache = TensorCache::new(128).unwrap();

        let meta1 = make_valid_meta();
        let data1 = Bytes::from(vec![0u8; 16]);
        cache.put("dup".to_string(), meta1, data1).unwrap();

        let meta2 = make_valid_meta();
        let data2 = Bytes::from(vec![0u8; 16]);
        let result = cache.put("dup".to_string(), meta2, data2);

        assert_eq!(result, Err(CacheError::KeyAlreadyExists));
    }

    #[test]
    fn get_missing_returns_none() {
        let cache = TensorCache::new(128).unwrap();
        assert!(cache.get("missing").is_none());
    }

    #[test]
    fn test_bytes_zero_copy_clone() {
        let cache = TensorCache::new(256).unwrap();

        let meta = make_valid_meta();
        let data = Bytes::from(vec![1u8; 16]);

        cache.put("shared".to_string(), meta, data).unwrap();

        let t1 = cache.get("shared").unwrap();
        let t2 = cache.get("shared").unwrap();

        assert_eq!(t1.byte_size(), t2.byte_size());

        let bytes1 = t1.get_data_cloned();
        let bytes2 = t2.get_data_cloned();

        assert_eq!(bytes1.len(), bytes2.len());
    }

    #[test]
    fn test_put_f32_conversion() {
        let cache = TensorCache::new(256).unwrap();

        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        cache.put_f32("float_tensor".to_string(), vec![2, 2], data).unwrap();

        let tensor = cache.get("float_tensor");
        assert!(tensor.is_some());
        assert_eq!(tensor.unwrap().byte_size(), 16);
    }
}