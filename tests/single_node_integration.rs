use redstone::transport::grpc::server::start_server;
use redstone::transport::grpc::client::RemoteCacheClient;
use redstone::tensor::meta::{TensorMeta,DType,StorageLayout};

use std::time::Duration;
use tokio::time::sleep;
use rand::{rng, Rng, RngExt};
fn random_port() -> u16 {
    rng().random_range(50060..60000)
}

#[tokio::test]
async fn single_node_put_get_delete_flow() {
    let port = random_port();
    let addr = format!("127.0.0.1:{}", port);

    //clone it so we can take ownership inside the below scope.
    let server_addr = addr.clone();
    tokio::spawn(async move {
        start_server(server_addr, 1024 * 1024)
            .await
            .expect("Server failed");
    });

    sleep(Duration::from_millis(200)).await;

    let mut client = RemoteCacheClient::connect(addr)
        .await
        .expect("Client failed to connect");

    /// Testing put RPC

    let meta = TensorMeta::new(
        DType::F32,
        vec![2, 2],
        StorageLayout::RowMajor,
    )
        .unwrap();

    let data = vec![1.0f32; 4];

    let bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        )
            .to_vec()
    };

    client
        .put("tensor1".to_string(), meta, bytes)
        .await
        .expect("Put failed");

    /// Testing get RPC

    let result = client
        .get("tensor1".to_string())
        .await
        .expect("Get failed");

    assert!(result.is_some());

    let tensor = result.unwrap();

    assert_eq!(tensor.get_metadata().shape(), &vec![2, 2]);
    assert_eq!(tensor.get_data().len(), 16);

/// Testing delete RPC

    client
        .delete("tensor1".to_string())
        .await
        .expect("Delete failed");

    let after_delete = client
        .get("tensor1".to_string())
        .await
        .expect("Get after delete failed");

    assert!(after_delete.is_none());
}

    /// Testing duplicate puts
#[tokio::test]
async fn duplicate_put_fails() {
    let port = random_port();
    let addr = format!("127.0.0.1:{}", port);
    let server_addr = addr.clone();
    tokio::spawn(async move {
        start_server(server_addr, 1024 * 1024)
            .await
            .expect("Server failed");
    });

    sleep(Duration::from_millis(200)).await;

    let mut client = RemoteCacheClient::connect(addr)
        .await
        .expect("Client failed");

    let meta_one = TensorMeta::new(
        DType::F32,
        vec![2, 2],
        StorageLayout::RowMajor,
    ).unwrap();

    let bytes = vec![0u8; 16];

    client
        .put("dup".to_string(), meta_one, bytes.clone())
        .await
        .expect("First put failed");

    let meta_two = TensorMeta::new(
        DType::F32,
        vec![2, 2],
        StorageLayout::RowMajor,
    ).unwrap();
    let second = client
        .put("dup".to_string(), meta_two, bytes)
        .await;

    assert!(second.is_err());
}