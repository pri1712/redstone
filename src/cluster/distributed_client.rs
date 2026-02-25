use std::iter::Map;
use crate::transport::grpc::client::RemoteCacheClient;

/// Cluster-aware client that routes cache operations to the correct
/// server node using consistent hashing.
/// It delegates:
/// - key → node mapping to the HashRing
/// - network communication to RemoteCacheClient
/// It contains no hashing or transport logic itself.

pub struct DistributedClient {
    //map servers nodeID to a single remoteCacheClient instance,
    clients: Map<String,RemoteCacheClient>,
}

impl DistributedClient {

}