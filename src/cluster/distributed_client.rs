use std::sync::{Arc};
use parking_lot::RwLock;
use std::collections::HashMap;
use crate::cluster::client_config::ClusterClientConfig;
use crate::cluster::node::Node;
use crate::transport::grpc::client::RemoteCacheClient;
use crate::cluster::ring::HashRing;

use crate::tensor::tensor::Tensor;
use crate::error::client_error::ClientError;
use crate::error::cache_error::CacheError;
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
    pub fn new_with_config(nodes: Vec<Node>,client_config: ClusterClientConfig) -> Self {
        let mut ring = HashRing::new(client_config.virtual_node_count);
        for node in nodes {
            //insert into the hashring.
            ring.add_node(Arc::from(node));
        }
        Self {
            clients: Arc::new(RwLock::new(HashMap::new())),
            ring: Arc::new(RwLock::new(ring)),
            client_config,
        }
    }

    pub fn new_default(nodes: Vec<Node>) -> Self {
       Self::new_with_config(nodes,ClusterClientConfig::default())
    }

    ///get a tensor from the cache
    /// returns: Ok(someData) if the key is found in the cache.
    /// Ok(None) if the key is not in the cache
    /// An error if there was some error while processing the request.
    pub async fn get(&self, key: &str) ->Result<Option<Arc<Tensor>>, ClientError > {
        for trial in 0..self.client_config.max_retries {
            match self.get_inner(key).await {
                Ok(data) => return Ok(data),
                Err(e) if e.is_retryable() && trial < self.client_config.max_retries - 1 => {
                    tokio::time::sleep(std::time::Duration::from_millis(50 * (trial as u64 + 1))).await;
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        Err(ClientError::MaxRetriesExceeded)
    }

    async fn get_inner(&self,key: &str) ->Result<Option<Arc<Tensor>>, ClientError> {
        /* get which node to send the query to from ring.rs, then send it to the appropriate client */
        /* from self.clients */
        let ring = self.ring.read();
        let selected_node = ring.get_node(key)
                                        .ok_or(ClientError::NoNodesAvailable)?
                                        .clone();
        let mut client = self.get_or_create_client(&selected_node).await?;

        let result = tokio::time::timeout(
            self.client_config.timeout,
            client.get(key.parse().unwrap()),
        )
            .await
            .map_err(|_| ClientError::Timeout)?;

        result
    }

    // helper functions
    pub async fn get_or_create_client(&self, node: &Node) -> Result<RemoteCacheClient, ClientError> {

        if let Some(client) = self.clients.read().get(&node.name) {
            //happy path
            return Ok(client.clone());
        }

        let new_client = RemoteCacheClient::connect(node.address.clone()).await?;

        let mut clients = self.clients.write();

        if let Some(existing) = clients.get(&node.name) {
            return Ok(existing.clone());
        }

        clients.insert(node.name.clone(), new_client.clone());
        Ok(new_client)
    }

}