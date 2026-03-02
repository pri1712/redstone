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

    pub fn get_node(&self,key: &str)  -> Option<&Arc<Node>> {
        //hash the key and go in a clockwise order in the ring, till u find a node.
        if self.ring.is_empty() {
            return None;
        }
        let key_hash = self.hasher.hash(key);
        let node = {
            self.ring
                .range(key_hash..)
                .next()
                .or_else(|| self.ring.iter().next())
                .map(|(_, node)| node)
        };
        node
    }

    //utility functions

    /// adds a node to the existing hash ring.
    /// This involves computing its hash and inserting it into the hash ring.
    pub fn add_node(&mut self, node: Arc<Node>) {
        let node_name = node.name.clone();
        let num_virtual_nodes = self.virtual_node_count;
        for i in 0..num_virtual_nodes {
            let virtual_node_key = format!("{}:#:{}", node_name, i);
            let node_hash = self.hasher.hash(&virtual_node_key);
            self.ring.insert(node_hash,Arc::clone(&node));
        }
    }

    // WARNING: removing a node causes data loss in current implementation
    pub fn remove_node(&mut self, node: Arc<Node>) -> bool {
        let num_virtual_nodes = self.virtual_node_count;
        let node_name = node.name.clone();
        let mut removed_any = false;
        for i in 0..num_virtual_nodes {
            let virual_node_key = format!("{}:#:{}", node_name, i);
            let node_hash = self.hasher.hash(&virual_node_key);
            removed_any |= self.ring.remove(&node_hash).is_some();
        }
        removed_any
    }

    pub fn len(&self) -> usize {
        self.ring.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_ring() {
        let ring = HashRing::new(3);
        assert_eq!(ring.virtual_node_count, 3);
    }

    #[test]
    fn test_add_node() {
        let mut ring = HashRing::new(3);
        let node_address = "http://127.0.0.1:5000";
        let node_name = "test_node";
        let node = Node::new(node_address,node_name);
        ring.add_node(Arc::new(node));
        assert_eq!(ring.len(), 3);
    }
}