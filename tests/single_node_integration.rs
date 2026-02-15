use redstone::transport::grpc::server::start_server;
use redstone::transport::grpc::client::RemoteCacheClient;
use redstone::tensor::meta::{TensorMeta,DType,StorageLayout};

use std::time::Duration;
use tokio::time::sleep;
use rand::{rng, RngExt};

use std::sync::Arc;
fn random_port() -> u16 {
    rng().random_range(50060..60000)
}

#[tokio::test]
async fn single_node_put_get_delete_flow() {

    let addr = server_setup().await;

    sleep(Duration::from_millis(200)).await;

    let mut client = RemoteCacheClient::connect(addr)
        .await
        .expect("Client failed to connect");

    // Testing put RPC

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
            data.len() * size_of::<f32>(),
        )
            .to_vec()
    };

    client
        .put("tensor1".to_string(), meta, bytes)
        .await
        .expect("Put failed");

    // Testing get RPC

    let result = client
        .get("tensor1".to_string())
        .await
        .expect("Get failed");

    assert!(result.is_some());

    let tensor = result.unwrap();

    assert_eq!(tensor.get_metadata().shape(), &vec![2, 2]);
    assert_eq!(tensor.get_data().len(), 16);

// Testing delete RPC

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

    // Testing duplicate puts
#[tokio::test]
async fn duplicate_put_fails() {

    let addr = server_setup().await;
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

#[tokio::test]
async fn test_oom_put() {
    use tokio::time::{sleep, Duration};
    let addr = server_setup().await;
    //allows server to initialize
    sleep(Duration::from_millis(200)).await;

    let mut client = RemoteCacheClient::connect(addr)
        .await
        .expect("Client failed to connect");

    let element_count = 300_000;

    let meta = TensorMeta::new(
        DType::F32,
        vec![element_count],
        StorageLayout::RowMajor,
    )
        .expect("Meta creation failed");

    let data = vec![0f32; element_count];

    let bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * size_of::<f32>(),
        )
            .to_vec()
    };

    let result = client
        .put("huge_tensor".to_string(), meta, bytes)
        .await;
    assert!(
        result.is_err(),
        "Expected OOM error but put succeeded"
    );
}

#[tokio::test]
async fn test_invalid_tensor_data() {

    let addr = server_setup().await;
    sleep(Duration::from_millis(200)).await;

    let mut client = RemoteCacheClient::connect(addr)
        .await
        .expect("Client failed");

    //16 bytes specified in tensor metadata.
    let meta = TensorMeta::new(
        DType::F32,
        vec![2, 2],
        StorageLayout::RowMajor,
    )
        .unwrap();
    //sending only 15 bytes.
    let invalid_bytes = vec![0u8; 15];

    let result = client
        .put("invalid_tensor".to_string(), meta, invalid_bytes)
        .await;

    assert!(result.is_err(),"Expected invalid tensor errors but put succeeded");
}


#[tokio::test]
async fn test_concurrent_clients_race_conditions() {
    //testing concurrent access to a key.
    let addr = server_setup().await;
    sleep(Duration::from_millis(200)).await;

    let addr_arc = Arc::new(addr);

    let writer = {
        let addr = addr_arc.clone();
        tokio::spawn(async move {
            let mut client = RemoteCacheClient::connect((*addr).clone())
                .await
                .expect("Writer failed");

            let bytes = vec![0u8; 40];

            for _ in 0..100 {
                let meta_one = TensorMeta::new(
                    DType::F32,
                    vec![10],
                    StorageLayout::RowMajor,
                ).unwrap();
                let _ = client.put("race_key".to_string(), meta_one, bytes.clone()).await;
            }
        })
    };

    let reader = {
        let addr = addr_arc.clone();
        tokio::spawn(async move {
            let mut client = RemoteCacheClient::connect((*addr).clone())
                .await
                .expect("Reader failed");

            for i in 0..100 {
                let result = client.get("race_key".to_string()).await;
                println!("Iteration {} → {:?}", i, result.is_ok());
            }
        })
    };

    writer.await.unwrap();
    reader.await.unwrap();

    let mut client = RemoteCacheClient::connect((*addr_arc).clone())
        .await
        .expect("Final client failed");

    let _ = client.get("race_key".to_string()).await.unwrap();
}

async fn server_setup() -> String {
    let port = random_port();
    let addr = format!("127.0.0.1:{}", port);
    let server_addr = addr.clone();
    tokio::spawn(async move {
        start_server(server_addr, 1024 * 1024)
            .await
            .expect("Server failed");
    });
    addr
}