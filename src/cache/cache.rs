use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::tensor::tensor::Tensor;

pub struct Cache {
    inner: RwLock<HashMap<String, Arc<Tensor>>>,
}

#[derive(Debug, PartialEq)]
pub enum CacheError {
    KeyAlreadyExists,
}

impl Cache {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
        }
    }

    pub fn put(&self, key: String, tensor: Tensor) -> Result<(), CacheError> {
        let mut guard = self.inner.write().unwrap();

        if guard.contains_key(&key) {
            return Err(CacheError::KeyAlreadyExists);
        }

        guard.insert(key, Arc::new(tensor));
        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<Arc<Tensor>> {
        let guard = self.inner.read().unwrap();
        guard.get(key).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::meta::{DType, StorageLayout, TensorMeta};
    use crate::tensor::tensor::Tensor;

    fn make_tensor() -> Tensor {
        let meta = TensorMeta::new(
            DType::F32,
            vec![4, 4],
            StorageLayout::RowMajor,
        ).unwrap();

        let data = vec![0u8; 64];
        Tensor::new(meta, data).unwrap()
    }

    #[test]
    fn test_cache_creation() {
        let _cache = Cache::new();
    }

    #[test]
    fn test_cache_put_and_get() {
        let cache = Cache::new();
        let key = "test_key".to_string();

        let tensor = make_tensor();
        cache.put(key.clone(), tensor).unwrap();

        let fetched = cache.get(&key);
        assert!(fetched.is_some());
    }

    #[test]
    fn test_cache_duplicate_put_fails() {
        let cache = Cache::new();
        let key = "dup_key".to_string();

        let tensor1 = make_tensor();
        let tensor2 = make_tensor();

        cache.put(key.clone(), tensor1).unwrap();
        let result = cache.put(key.clone(), tensor2);

        assert_eq!(result, Err(CacheError::KeyAlreadyExists));
    }

    #[test]
    fn test_cache_get_missing_returns_none() {
        let cache = Cache::new();
        assert!(cache.get("missing").is_none());
    }
}