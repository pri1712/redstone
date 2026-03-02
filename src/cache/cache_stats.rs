#[derive(Debug, Clone, PartialEq)]
pub struct CacheStats {
    pub entries: u64,
    pub memory_used: u64,
    pub memory_limit: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    pub fn memory_utilization(&self) -> f64 {
        if self.memory_limit == 0 {
            0.0
        } else {
            self.memory_used as f64 / self.memory_limit as f64
        }
    }
}

impl From<crate::transport::grpc::client::CacheStats> for CacheStats {
    fn from(value: crate::transport::grpc::client::CacheStats) -> Self {
        Self {
            entries: value.entries,
            memory_used: value.memory_used,
            memory_limit: value.memory_limit,
            hits: value.hits,
            misses: value.misses,
            evictions: value.evictions,
        }
    }
}