use std::iter::Map;
use std::sync::{Arc};
use parking_lot::RwLock;
use std::collections::HashMap;
use crate::cluster::client_config::ClusterClientConfig;
use crate::cluster::node::Node;
use crate::transport::grpc::client::RemoteCacheClient;
use crate::cluster::ring::HashRing;
/// Cluster-aware client that routes cache operations to the correct
/// server node using consistent hashing.
/// It delegates:
/// - key → node mapping to the HashRing
/// - network communication to RemoteCacheClient
/// It contains no hashing or transport logic itself, it only serves as the main point of entry for
/// clients

pub struct DistributedClient {
    //map servers node name to a single remoteCacheClient instance,
    clients: Arc<RwLock<HashMap<String, RemoteCacheClient>>>,
    ring: Arc<RwLock<HashRing>>,
    client_config: ClusterClientConfig
}

impl DistributedClient {
    fn new_with_config(nodes: Vec<Node>,client_config: ClusterClientConfig) -> Self {
        let mut ring = HashRing::
    }

    fn new_default(nodes: Vec<Node>) -> Self {
       Self::new_with_config(nodes,ClusterClientConfig::default())
    }
}