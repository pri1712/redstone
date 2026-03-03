// cluster/config.rs

use serde::Deserialize;
use std::{fs, time::Duration};
use crate::cluster::distributed_client::DistributedClient;
use super::runtime_config::ClusterClientConfig;
use crate::cluster::node::Node;

#[derive(Debug, Deserialize)]
pub struct ClusterClientFileConfig {
    pub nodes: Vec<NodeFileConfig>,
    pub max_retries: Option<u32>,
    pub timeout_ms: Option<u64>,
    pub virtual_node_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct NodeFileConfig {
    pub name: String,
    pub address: String,
}

impl ClusterClientFileConfig {
    pub fn load(path: &str)
                -> Result<Self, Box<dyn std::error::Error>>
    {
        let content = fs::read_to_string(path)?;
        Ok(toml::from_str(&content)?)
    }

    pub fn into_runtime(self) -> (Vec<Node>, ClusterClientConfig) {
        let nodes = self.nodes
            .into_iter()
            .map(|n| Node::new(n.address, n.name))
            .collect();

        let config = ClusterClientConfig {
            max_retries: self.max_retries.unwrap_or(3),
            timeout: Duration::from_millis(self.timeout_ms.unwrap_or(5000)),
            virtual_node_count: self.virtual_node_count.unwrap_or(50),
        };

        (nodes, config)
    }
}