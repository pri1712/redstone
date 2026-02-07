use std::sync::Arc;
mod tensor;
mod cache;

use crate::cache::lru_cache::{Cache, CacheError};
use crate::tensor::meta::{DType, StorageLayout, TensorMeta};
use crate::tensor::tensor::Tensor;

pub struct TensorCache {
    cache: Cache,
}

impl TensorCache {
    pub fn new(max_cache_size: u64) -> Self {
        Self {
            cache: Cache::new(max_cache_size).unwrap(),
        }
    }

    /// Put method for f32 data type. It internally implements the core put method.
    pub fn put_f32(&self, key: String, shape: Vec<usize>, data: Vec<f32>) -> Result<(), CacheError> {
        let meta = TensorMeta::new(DType::F32, shape, StorageLayout::RowMajor, )
            .map_err(|_| CacheError::InvalidTensorMetadata)?;

        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8,
                data.len() * size_of::<f32>()).to_vec()
        };

        self.put(key, meta, bytes)
    }

    /// Inserts a tensor into the cache.
    /// It guarantees:
    /// 1. Immutable writes
    /// 2. Tensor validation before insertion, preventing corrupted writes
    /// 3. Atomic inserts
    pub fn put(&self, key: String, meta: TensorMeta, data: Vec<u8>, ) -> Result<(), CacheError> {
        let tensor = Tensor::new(meta, data)
            .map_err(|_| CacheError::InvalidTensor)?;
        self.cache.put(key, tensor)
    }

    ///     Retrieves a tensor by key.
    ///     It guarantees:
    ///     1. Atomic reads
    ///     2. Idempotent reads
    pub fn get(&self, key: &str) -> Option<Arc<Tensor>> {
        self.cache.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::meta::{DType, StorageLayout, TensorMeta};

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
        let cache = TensorCache::new(128);

        let meta = make_valid_meta();
        let data = vec![0u8; 16];

        cache.put("t1".to_string(), meta, data).unwrap();

        let tensor = cache.get("t1");
        assert!(tensor.is_some());
    }

    #[test]
    fn invalid_tensor_is_rejected() {
        let cache = TensorCache::new(128);

        let meta = make_valid_meta();
        let data = vec![0u8; 15];

        let result = cache.put("bad".to_string(), meta, data);
        assert_eq!(result, Err(CacheError::InvalidTensor));
    }

    #[test]
    fn duplicate_key_is_rejected() {
        let cache = TensorCache::new(128);

        let meta1 = make_valid_meta();
        let data1 = vec![0u8; 16];
        cache.put("dup".to_string(), meta1, data1).unwrap();

        let meta2 = make_valid_meta();
        let data2 = vec![0u8; 16];
        let result = cache.put("dup".to_string(), meta2, data2);

        assert_eq!(result, Err(CacheError::KeyAlreadyExists));
    }

    #[test]
    fn get_missing_returns_none() {
        let cache = TensorCache::new(128);
        assert!(cache.get("missing").is_none());
    }
}