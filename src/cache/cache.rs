use std::collections::HashMap;
use crate::tensor::tensor::Tensor;
use std::sync::{Arc, RwLock};
//actual caching logic
//the put()/get() request arrives here via the exposed API in lib.rs , we use a in memory hashmap
// to store data in KV pairs.
// hashing strategy-> ?
// eviction strategy -> ?
//cache is defined as TensorKey->TensorValue mapping
pub struct Cache {
    inner: RwLock<HashMap<String, Arc<Tensor>>>
}

impl Cache {
    pub fn new() -> Self {
        Self{
            inner: RwLock::new(HashMap::new())
        }
    }

    pub fn put(cache: &Self,key: String,value:Arc<Tensor>) -> Result<(), String> {

    }

    pub fn get(cache: &Self,key: &String) -> Option<Arc<Tensor>> {

    }
}
