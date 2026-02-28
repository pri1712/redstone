use thiserror::Error;

#[derive(Error, Debug)]
pub enum ClientError {
    #[error("No nodes available in cluster")]
    NoNodesAvailable,

    #[error("Request timeout")]
    Timeout,

    #[error("Max retries exceeded")]
    MaxRetriesExceeded,

    #[error("Transport error")]
    Transport(#[from] tonic::transport::Error),

    #[error("gRPC status error: {0}")]
    GrpcStatus(#[from] tonic::Status),

    #[error("Server error: {0}")]
    ServerError(String),
}

impl ClientError {
    pub fn is_retryable(&self) -> bool {
        match self {
            ClientError::Timeout => true,
            ClientError::Transport(_) => true,
            ClientError::GrpcStatus(status) => {
                matches!(
                    status.code(),
                    tonic::Code::Unavailable
                        | tonic::Code::DeadlineExceeded
                        | tonic::Code::Internal
                )
            }
            _ => false,
        }
    }
}