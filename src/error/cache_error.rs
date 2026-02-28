use thiserror::Error;

#[derive(Error, Debug,PartialEq)]
pub enum CacheError {
    #[error("Key already exists in cache")]
    KeyAlreadyExists,

    #[error("Invalid tensor")]
    InvalidTensor,

    #[error("Invalid tensor size")]
    InvalidSize,

    #[error("Out of memory, increase cache size to insert larger tensors")]
    OutOfMemory,

    #[error("Invalid tensor metadat")]
    InvalidTensorMetadata,
}