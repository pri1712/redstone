use std::sync::Arc;
mod tensor;
mod cache;

use crate::cache::cache::{Cache, CacheError};
use crate::tensor::meta::TensorMeta;
use crate::tensor::tensor::Tensor;

pub struct TensorCache {
    cache: Cache,
}

impl TensorCache {
    pub fn new() -> Self {
        Self {
            cache: Cache::new(),
        }
    }

    /***
    Inserts a tensor into the cache.
    It guarantees:
    1. Immutable writes
    2. Tensor validation before insertion, preventing corrupted writes
    3. Atomic inserts
     */
    pub fn put(&self, key: String, meta: TensorMeta, data: Vec<u8>, ) -> Result<(), CacheError> {
        let tensor = Tensor::new(meta, data)
            .map_err(|_| CacheError::InvalidTensor)?;
        self.cache.put(key, tensor)
    }

    /***
    Retrieves a tensor by key.
    It guarantees:
    1. Atomic reads
    2. Idempotent reads
     */
    pub fn get(&self, key: &str) -> Option<Arc<Tensor>> {
        self.cache.get(key)
    }
}