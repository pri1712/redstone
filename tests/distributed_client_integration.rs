use redstone::transport::grpc::server::start_server;
use redstone::cluster::distributed_client::DistributedClient;
use redstone::cluster::node::Node;
use redstone::tensor::meta::{TensorMeta, DType, StorageLayout};

use std::time::Duration;
use tokio::time::sleep;
use rand::{rng, RngExt};
use redstone::transport::grpc::client::RemoteCacheClient;

fn random_port() -> u16 {
    rng().random_range(51000..60000)
}

async fn spawn_server() -> String {
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

#[tokio::test]
async fn distributed_client_get_false_node_flow() {
    /* tests scenario when the key that the GET request asks for does not exist*/

    let addr1 = spawn_server().await;
    let addr2 = spawn_server().await;
    let addr3 = spawn_server().await;

    sleep(Duration::from_millis(300)).await;

    let node_one = Node::new(addr1,"server-one");
    let node_two = Node::new(addr2,"server-two");
    let node_three = Node::new(addr3,"server-three");

    let nodes = vec![node_one, node_two, node_three];

    let client = DistributedClient::new_default(nodes);

    let meta = TensorMeta::new(
        DType::F32,
        vec![4],
        StorageLayout::RowMajor,
    ).unwrap();

    let bytes = vec![0u8; 16];

    let result = client
        .get("distributed_key")
        .await
        .expect("Distributed GET failed");
    assert!(result.is_none());
}

#[tokio::test]
async fn distributed_client_put_flow() {
    let addr1 = spawn_server().await;
    let addr2 = spawn_server().await;
    let addr3 = spawn_server().await;

    sleep(Duration::from_millis(300)).await;

    let node_one = Node::new(addr1.clone(), "server-one");
    let node_two = Node::new(addr2.clone(), "server-two");
    let node_three = Node::new(addr3.clone(), "server-three");

    let nodes = vec![node_one, node_two, node_three];

    let client = DistributedClient::new_default(nodes);

    let meta = TensorMeta::new(
        DType::F32,
        vec![4],
        StorageLayout::RowMajor,
    ).unwrap();

    let bytes = vec![0u8; 16];
    let key = "put_test".to_string();

    client.put(key.clone(), meta.clone(), bytes.clone())
        .await
        .expect("Distributed PUT failed");

    let result = client.get(&key).await.expect("Distributed GET failed");
    assert!(result.is_some(), "Key not found via distributed client");
}

#[tokio::test]
async fn distributed_delete_flow() {
    let addr1 = spawn_server().await;
    let addr2 = spawn_server().await;
    let addr3 = spawn_server().await;

    tokio::time::sleep(Duration::from_millis(300)).await;

    let nodes = vec![
        Node::new(addr1.clone(), "node1"),
        Node::new(addr2.clone(), "node2"),
        Node::new(addr3.clone(), "node3"),
    ];

    let client = DistributedClient::new_default(nodes);

    let meta = TensorMeta::new(
        DType::F32,
        vec![4],
        StorageLayout::RowMajor,
    ).unwrap();

    let bytes = vec![0u8; 16];
    let key = "delete_test_key".to_string();

    client.put(key.clone(), meta.clone(), bytes.clone())
        .await
        .expect("PUT failed");

    let exists = client.get(&key).await.unwrap();
    assert!(exists.is_some());

    client.delete(&key).await.expect("Delete failed");

    let after_delete = client.get(&key).await.unwrap();
    assert!(after_delete.is_none());
}

#[tokio::test]
async fn distributed_get_stats_flow() {
    let addr1 = spawn_server().await;
    let addr2 = spawn_server().await;
    let addr3 = spawn_server().await;

    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    let nodes = vec![
        Node::new(addr1.clone(), "node1"),
        Node::new(addr2.clone(), "node2"),
        Node::new(addr3.clone(), "node3"),
    ];

    let client = DistributedClient::new_default(nodes);

    let meta = TensorMeta::new(
        DType::F32,
        vec![4],
        StorageLayout::RowMajor,
    ).unwrap();

    let bytes = vec![0u8; 16];

    for i in 0..10 {
        client.put(format!("stats_key_{}", i), meta.clone(), bytes.clone())
            .await
            .expect("PUT failed");
    }

    let stats = client.get_per_server_stats()
        .await
        .expect("get_per_server_stats failed");

    assert!(!stats.is_empty(), "No stats returned");

    let total_entries: u64 = stats.iter().map(|s| s.entries).sum();
    assert_eq!(
        total_entries,
        10,
        "Total entries across cluster incorrect"
    );
}