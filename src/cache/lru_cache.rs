use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, RwLock};

use crate::cache::cache_stats::CacheStats;
use crate::tensor::tensor::Tensor;

pub struct Cache {
    inner: RwLock<CacheInner>,
}

impl Cache {
    pub fn new(max_size: u64) -> Result<Self, CacheError> {
        if max_size == 0 {
            return Err(CacheError::InvalidSize);
        }
        Ok(Self {
            inner: RwLock::new(CacheInner {
                map: HashMap::new(),
                head: None,
                tail: None,
                current_cache_size_bytes: 0,
                max_cache_size_bytes: max_size,
                hits: 0,
                misses: 0,
                evictions: 0,
            })
        })
    }

    pub fn put(&self, key: String, tensor: Tensor) -> Result<(), CacheError> {
        let mut inner = self.inner.write().unwrap();
        inner.put(key, tensor)
    }

    pub fn get(&self, key: &str) -> Option<Arc<Tensor>> {
        let mut inner = self.inner.write().unwrap();
        inner.get(key)
    }

    pub fn stats(&self) -> CacheStats {
        let inner = self.inner.read().unwrap();
        CacheStats {
            entries: inner.map.len(),
            memory_used: inner.current_cache_size_bytes,
            memory_limit: inner.max_cache_size_bytes,
            hits: inner.hits,
            misses: inner.misses,
            evictions: inner.evictions,
        }
    }

}

struct LruNode {
    key: String,
    prev: Option<NonNull<LruNode>>,
    next: Option<NonNull<LruNode>>,
}
impl LruNode {
    fn new(key: String) -> Self {
        Self {
            key,
            prev: None,
            next: None,
        }
    }
}
struct CacheInner {
    /// actual cache
    map: HashMap<String, (Arc<Tensor>, NonNull<LruNode>,u64)>,

    /// for least and most recently used
    head: Option<NonNull<LruNode>>,
    tail: Option<NonNull<LruNode>>,

    /// for eviction
    current_cache_size_bytes: u64,
    max_cache_size_bytes: u64,

    /// metrics
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl CacheInner {
    pub fn put(&mut self, key: String, tensor: Tensor) -> Result<(), CacheError> {
        if self.map.contains_key(&key) {
            return Err(CacheError::KeyAlreadyExists);
        }
        //check if adding the tensor exceeds cache size, if it does ; keep dropping the LRU key
        // till we are able to insert the new key at the head.
        let tensor_size = tensor.byte_size() as u64;
        while tensor_size + self.current_cache_size_bytes > self.max_cache_size_bytes && self.tail.is_some() {
            //keep evicting LRU key.
            self.evict_key();

        }

        if self.current_cache_size_bytes + tensor_size > self.max_cache_size_bytes {
            return Err(CacheError::OutOfMemory);
        }
        let node = Box::new(LruNode::new(key.clone()));

        let node_ptr = unsafe { NonNull::new_unchecked(Box::into_raw(node)) };
        self.attach_node_to_head(node_ptr);
        self.map.insert(key, (Arc::new(tensor), node_ptr, tensor_size));
        self.current_cache_size_bytes += tensor_size;
        Ok(())
    }

    fn get(&mut self, key: &str) -> Option<Arc<Tensor>> {
        let entry = self.map.get(key);
        if entry.is_none() {
            self.misses += 1;
            return None;
        }

        let (tensor, node_ptr, _) = entry.unwrap();
        let tensor = Arc::clone(tensor);
        let node_ptr = *node_ptr;

        self.detach_node(node_ptr);
        self.attach_node_to_head(node_ptr);
        self.hits += 1;

        Some(tensor)
    }

 /// evict least recently used key
    pub fn evict_key(&mut self) {
        if let Some(tail_ptr) = self.tail {
            let tail = unsafe { tail_ptr.as_ref() };
            let key = tail.key.clone();
            if let Some((_tensor,_node_ptr,_node_size)) = self.map.remove(&key) {
                //reduce size of the cache and modify metrics
                self.evictions += 1;
                self.current_cache_size_bytes -= _node_size;
            }
            self.detach_node(tail_ptr);
            //move tail memory to heap so its dropped at end of the method.
            unsafe {Box::from_raw(tail_ptr.as_ptr())};
        }
    }

    fn detach_node(&mut self,node_ptr: NonNull<LruNode>) {
        //3 cases, when node is head,middle node or tail.
        unsafe {
            let node = node_ptr.as_ref();
            match (node.prev,node.next) {
                (None,None) => {
                    //when its the only node.
                    self.tail = None;
                    self.head = None;
                }
                (None,Some(next)) => {
                    //head
                    self.head = Some(next);
                    (*next.as_ptr()).prev = None;
                }
                (Some(prev),None) => {
                    //tail
                    self.tail = Some(prev);
                    (*prev.as_ptr()).next = None;

                }
                (Some(prev),Some(next)) => {
                    //middle node
                    (*prev.as_ptr()).next = Some(next);
                    (*next.as_ptr()).prev = Some(prev);
                }
            }
        }
    }

    fn attach_node_to_head(&mut self,node_ptr: NonNull<LruNode>) {
        unsafe {
            let node = &mut *node_ptr.as_ptr();
            node.prev = None;
            node.next = self.head;

            if let Some(old_head) = self.head {
                (*old_head.as_ptr()).prev = Some(node_ptr);
            }

            self.head = Some(node_ptr);

            if self.tail.is_none() {
                //if its the only node in the linked list.
                self.tail = Some(node_ptr);
            }
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum CacheError {
    KeyAlreadyExists,
    InvalidTensor,
    InvalidSize,
    OutOfMemory,
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
        let cache = Cache::new(256).unwrap();
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

    // #[test]
    // fn test_concurrent_reads() {
    //     let cache = Arc::new(Cache::new(64).unwrap());
    //     let tensor = make_tensor();
    //
    //     cache
    //         .put("shared_key".to_string(), tensor)
    //         .unwrap();
    //
    //     let handles: Vec<_> = (0..10)
    //         .map(|_| {
    //             let cache_clone = Arc::clone(&cache);
    //             thread::spawn(move || {
    //                 for _ in 0..100 {
    //                     assert!(cache_clone.get("shared_key").is_some());
    //                 }
    //             })
    //         })
    //         .collect();
    //
    //     for h in handles {
    //         h.join().unwrap();
    //     }
    // }

    #[test]
    fn test_invalid_size() {
        assert!(matches!(Cache::new(0), Err(CacheError::InvalidSize)));
    }

    #[test]
    fn test_out_of_memory_with_eviction() {

        let cache = Cache::new(100).unwrap(); // 100 bytes total
        let tensor1 = make_tensor(); // 64 bytes

        cache.put("key1".to_string(), tensor1).unwrap();

        let tensor2 = make_tensor();

        let result = cache.put("key2".to_string(), tensor2);
        assert!(result.is_ok());

        assert!(cache.get("key1").is_none());
        assert!(cache.get("key2").is_some());

        let stats = cache.stats();
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn test_lru_eviction_order() {
        let cache = Cache::new(150).unwrap();

        let tensor1 = make_tensor();
        let tensor2 = make_tensor();
        let tensor3 = make_tensor();

        cache.put("key1".to_string(), tensor1).unwrap();
        cache.put("key2".to_string(), tensor2).unwrap();

        cache.get("key1");

        cache.put("key3".to_string(), tensor3).unwrap();

        assert!(cache.get("key1").is_some());
        assert!(cache.get("key2").is_none());
        assert!(cache.get("key3").is_some());
    }

    #[test]
    fn test_stats_tracking() {
        let cache = Cache::new(200).unwrap();

        let tensor = make_tensor();
        cache.put("key1".to_string(), tensor).unwrap();

        cache.get("key1");
        cache.get("missing");

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate(), 0.5);
        assert_eq!(stats.entries, 1);
    }

    // #[test]
    // fn test_delete() {
    //     let cache = Cache::new(200).unwrap();
    //
    //     let tensor = make_tensor();
    //     cache.put("key1".to_string(), tensor).unwrap();
    //
    //     assert!(cache.exists("key1"));
    //
    //     let deleted = cache.delete("key1");
    //     assert!(deleted.is_some());
    //     assert!(!cache.exists("key1"));
    //
    //     let stats = cache.stats();
    //     assert_eq!(stats.entries, 0);
    // }

    // #[test]
    // fn test_clear() {
    //     let cache = Cache::new(200).unwrap();
    //
    //     cache.put("key1".to_string(), make_tensor()).unwrap();
    //     cache.put("key2".to_string(), make_tensor()).unwrap();
    //
    //     assert_eq!(cache.stats().entries, 2);
    //
    //     cache.clear();
    //
    //     let stats = cache.stats();
    //     assert_eq!(stats.entries, 0);
    //     assert_eq!(stats.memory_used, 0);
    // }

    #[test]
    fn test_single_large_tensor_oom() {
        let cache = Cache::new(50).unwrap(); // Only 50 bytes

        let tensor = make_tensor(); // 64 bytes - too big

        let result = cache.put("huge".to_string(), tensor);
        assert_eq!(result, Err(CacheError::OutOfMemory));
    }
}
