/// Cluster-aware client that routes cache operations to the correct
/// server node using consistent hashing.
/// It delegates:
/// - key → node mapping to the HashRing
/// - network communication to RemoteCacheClient
/// It contains no hashing or transport logic itself, it only serves as the main point of entry for
/// clients

use std::sync::{Arc};
use parking_lot::RwLock;
use std::collections::HashMap;
use crate::cluster::client_config::ClusterClientConfig;
use crate::cluster::node::Node;
use crate::transport::grpc::client::RemoteCacheClient;
use crate::cluster::ring::HashRing;

use crate::tensor::tensor::Tensor;
use crate::error::client_error::ClientError;
use crate::tensor::meta::TensorMeta;
use crate::cache::cache_stats::CacheStats;

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

    pub async fn get(&self, key: &str) ->Result<Option<Arc<Tensor>>, ClientError > {
        /* get a tensor from the cache */
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

    pub async fn put(&self, key: String, meta: TensorMeta, data: Vec<u8>) -> Result<(), ClientError > {
        /* inserts a key and tensor specified by the user */
        for trial in 0..self.client_config.max_retries {
            match self.put_inner(&*key, meta.clone(), data.clone()).await {
                Ok(..) => return Ok(()),
                Err(e) if e.is_retryable() && trial < self.client_config.max_retries - 1 => {
                    tokio::time::sleep(std::time::Duration::from_millis(50 * (trial as u64 + 1))).await;
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        Err(ClientError::MaxRetriesExceeded)
    }

    pub async fn delete(&self, key: &str) -> Result<(), ClientError > {
        /* deletes a key from the cache specified by key */
        for trial in 0..self.client_config.max_retries {
            match self.delete_inner(key).await {
                Ok(..) => return Ok(()),
                Err(e) if e.is_retryable() && trial < self.client_config.max_retries - 1 => {
                    tokio::time::sleep(std::time::Duration::from_millis(50 * (trial as u64 + 1))).await;
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        Err(ClientError::MaxRetriesExceeded)
    }

    pub async fn get_per_server_stats(&self) -> Result<Vec<CacheStats>, ClientError> {
        let clients: Vec<_> = {
            let guard = self.clients.read();
            guard.values().cloned().collect()
        };

        let mut stats_vec = Vec::with_capacity(clients.len());

        for mut client in clients {
            let result = tokio::time::timeout(
                self.client_config.timeout,
                client.get_stats(),
            )
                .await
                .map_err(|_| ClientError::Timeout)?;

            let stats = result?;
            stats_vec.push(stats.into());
        }
        Ok(stats_vec)
    }

    async fn get_inner(&self,key: &str) ->Result<Option<Arc<Tensor>>, ClientError> {
        /* get which node to send the query to from ring.rs, then send it to the appropriate client */
        /* from self.clients */
        let ring  = self.ring.read();
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

    async fn put_inner(&self, key: &str, meta: TensorMeta, data: Vec<u8>) ->Result<(), ClientError> {
        let ring  = self.ring.read();
        let selected_node = ring.get_node(key)
            .ok_or(ClientError::NoNodesAvailable)?
            .clone();
        let mut client = self.get_or_create_client(&selected_node).await?;

        let result = tokio::time::timeout(
            self.client_config.timeout,
            client.put(key.to_string(), meta, data.clone()),
        )
            .await
            .map_err(|_| ClientError::Timeout)?;
        result
    }

    async fn delete_inner(&self,key: &str) ->Result<(), ClientError> {
        let ring  = self.ring.read();
        let selected_node = ring.get_node(key)
            .ok_or(ClientError::NoNodesAvailable)?
            .clone();
        let mut client = self.get_or_create_client(&selected_node).await?;

        let result = tokio::time::timeout(
            self.client_config.timeout,
            client.delete(key.to_string()),
        )
            .await
            .map_err(|_| ClientError::Timeout)?;
        result
    }

    pub async fn add_node(&self, node: Node) {
        let mut ring = self.ring.write();
        ring.add_node(Arc::from(node));
    }

    // WARNING: removing a node causes data loss in current implementation
    pub async fn remove_node(&self, node: Node) -> Result<(), ClientError> {
        let node_name = node.name.clone();
        {
            let mut ring = self.ring.write();
            ring.remove_node(Arc::from(node));
        }
        let mut clients = self.clients.write();
        clients.remove(&node_name);
        Ok(())
    }

    // helper functions
    async fn get_or_create_client(&self, node: &Node) -> Result<RemoteCacheClient, ClientError> {
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
#[cfg(test)]
//these tests only test basic code functionality. integration tests are in distributed_client_integration.rs
mod tests {
    use super::*;
    use crate::cluster::node::Node;

    #[test]
    fn test_distributed_client_new() {
        /* tests distributed client creation with default config */
        let node_one = Node::new(
            "node1".to_string(),
            "127.0.0.1:50051".to_string(),
        );

        let node_two = Node::new(
            "node2".to_string(),
            "127.0.0.1:50052".to_string(),
        );

        let client = DistributedClient::new_default(vec![node_one, node_two]);

        let ring = client.ring.read();
        let selected = ring.get_node("test_key");

        assert!(selected.is_some());
    }
}

