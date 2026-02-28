pub struct ClusterClientConfig {
    pub max_retries: u32,
    pub timeout: std::time::Duration,
    pub virtual_node_count: u32,
}

impl ClusterClientConfig {
    pub fn new(max_retries: u32, timeout: std::time::Duration,virtual_node_count: u32) -> Self {
        Self {
            max_retries,
            timeout,
            virtual_node_count,
        }
    }

    pub fn new_default() -> Self {
        Self {
            max_retries: 3,
            timeout: std::time::Duration::from_secs(5),
            virtual_node_count: 50,
        }
    }
}
impl Default for ClusterClientConfig {
    fn default() -> Self {
        Self::new_default()
    }
}