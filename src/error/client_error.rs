use thiserror::Error;
#[derive(Error, Debug)]
pub enum ClientError {
    #[error("No nodes available in cluster")]
    NoNodesAvailable,

    #[error("Request timeout")]
    Timeout,

    #[error("Max retries exceeded")]
    MaxRetriesExceeded,

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Server error: {0}")]
    ServerError(String),
}

impl ClientError {
    pub fn is_retryable(&self) -> bool {
        matches!(self, ClientError::Timeout | ClientError::NetworkError(_))
    }
}