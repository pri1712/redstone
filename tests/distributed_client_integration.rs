use redstone::transport::grpc::server::start_server;
use redstone::cluster::distributed_client::DistributedClient;
use redstone::cluster::node::Node;
use redstone::tensor::meta::{TensorMeta, DType, StorageLayout};

use std::time::Duration;
use tokio::time::sleep;
use rand::{rng, RngExt};

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
async fn distributed_client_three_node_flow() {
    let addr1 = spawn_server().await;
    let addr2 = spawn_server().await;
    let addr3 = spawn_server().await;
    println!("addr1: {}", addr1);
    println!("addr2: {}", addr2);
    println!("addr3: {}", addr3);
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
    println!("result is {}",result.is_some());
    assert!(result.is_some());
    assert_eq!(result.unwrap().get_data().len(), 16);
}