use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::thread;
use crate::tensor::tensor::Tensor;

pub struct Cache {
    inner: RwLock<HashMap<String, Arc<Tensor>>>,
    max_cache_size_bytes: u64,
    current_size_bytes: RwLock<u64>,
}

#[derive(Debug, PartialEq)]
pub enum CacheError {
    KeyAlreadyExists,
    InvalidTensor,
    InvalidSize,
    OutOfMemory,
}

impl Cache {
    pub fn new(max_size: u64) -> Result<Self, CacheError> {
        if max_size == 0 {
            return Err(CacheError::InvalidSize);
        }
        Ok(Self {
            inner: RwLock::new(HashMap::new()),
            max_cache_size_bytes: max_size,
            current_size_bytes: RwLock::new(0),
        })
    }

    pub fn put(&self, key: String, tensor: Tensor) -> Result<(), CacheError> {
        let mut guard = self.inner.write().unwrap();
        let tensor_size = tensor.byte_size();

        if guard.contains_key(&key) {
            return Err(CacheError::KeyAlreadyExists);
        }

        let mut current_size_bytes = self.current_size_bytes.write().unwrap();
        if *current_size_bytes + tensor_size as u64 > self.max_cache_size_bytes {
            return Err(CacheError::OutOfMemory);
        }

        guard.insert(key, Arc::new(tensor));
        *current_size_bytes += tensor_size as u64;
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
    use std::sync::Arc;
    use std::thread;

    fn make_tensor() -> Tensor {
        let meta = TensorMeta::new(
            DType::F32,
            vec![4, 4],
            StorageLayout::RowMajor,
        )
            .unwrap();

        let data = vec![0u8; 64];
        Tensor::new(meta, data).unwrap()
    }

    #[test]
    fn test_cache_creation() {
        let cache = Cache::new(64);
        assert!(cache.is_ok());
    }

    #[test]
    fn test_cache_put_and_get() {
        let cache = Cache::new(64).unwrap();
        let key = "test_key".to_string();

        let tensor = make_tensor();
        cache.put(key.clone(), tensor).unwrap();

        let fetched = cache.get(&key);
        assert!(fetched.is_some());
    }

    #[test]
    fn test_cache_duplicate_put_fails() {
        let cache = Cache::new(64).unwrap();
        let key = "dup_key".to_string();

        let tensor1 = make_tensor();
        let tensor2 = make_tensor();

        cache.put(key.clone(), tensor1).unwrap();
        let result = cache.put(key.clone(), tensor2);

        assert_eq!(result, Err(CacheError::KeyAlreadyExists));
    }

    #[test]
    fn test_cache_get_missing_returns_none() {
        let cache = Cache::new(64).unwrap();
        assert!(cache.get("missing").is_none());
    }

    #[test]
    fn test_concurrent_reads() {
        let cache = Arc::new(Cache::new(64).unwrap());
        let tensor = make_tensor();

        cache
            .put("shared_key".to_string(), tensor)
            .unwrap();

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let cache_clone = Arc::clone(&cache);
                thread::spawn(move || {
                    for _ in 0..100 {
                        assert!(cache_clone.get("shared_key").is_some());
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_invalid_size() {
        assert!(matches!(Cache::new(0), Err(CacheError::InvalidSize)));
    }
    #[test]
    fn test_out_of_memory() {
        let cache = Cache::new(64).unwrap();
        let tensor1 = make_tensor();

        cache.put("key1".to_string(), tensor1).unwrap();

        let tensor2 = make_tensor();

        let result = cache.put("key2".to_string(), tensor2);

        assert_eq!(result, Err(CacheError::OutOfMemory));

        assert!(cache.get("key1").is_some());
    }
}