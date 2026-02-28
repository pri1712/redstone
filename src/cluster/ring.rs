///Contains the implementation for assigning hashed data to server nodes.
use std::collections::BTreeMap;
use std::sync::Arc;
use crate::cluster::hashing::{HashFunction, XxHash};
use crate::cluster::node::Node;

pub struct HashRing {
    //for ordered access of node hashes
    ring: BTreeMap<u64, Arc<Node>>,
    virtual_node_count: u32,
    //for now its fixed to be xxhash, can introduce configurability later.
    hasher: Box<dyn HashFunction>,
}

impl HashRing {
    pub fn new(virtual_node_count: u32) -> Self {
        Self::new_with_hasher(virtual_node_count, Box::new(XxHash))
    }

    pub fn new_with_hasher(virtual_node_count: u32, hasher: Box<dyn HashFunction>) -> Self {
        Self {
            ring: BTreeMap::new(),
            virtual_node_count,
            hasher,
        }
    }
    
    /// adds a node to the existing hash ring. 
    /// This involves computing its hash and inserting it into the hash ring.
    pub fn add_node(&mut self, node: Node) {
        
    }
}