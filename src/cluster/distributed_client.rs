use std::iter::Map;
use std::sync::{Arc};
use parking_lot::RwLock;
use std::collections::HashMap;

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

}

impl DistributedClient {
    fn new(clients: Arc<RwLock<HashMap<String, RemoteCacheClient>>>, ring: Arc<RwLock<HashRing>>) -> Self {
        Self { clients, ring }
    }
}