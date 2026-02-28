///Contains the implementation for assigning hashed data to server nodes.
use std::collections::BTreeMap;
use crate::cluster::hashing::HashFunction;
use crate::cluster::node::Node;

pub struct HashRing {
    //for ordered access of node hashes
    ring: BTreeMap<u64, Node>,
    virtual_node_count: u32,
    //for now its fixed to be xxhash, can introduce configurability later.
    hasher: dyn HashFunction
}