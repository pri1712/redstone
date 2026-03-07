use redstone::cluster::distributed_client::DistributedClient;
use redstone::cluster::node::Node;
use redstone::transport::grpc::server::start_server;
use redstone::tensor::meta::{TensorMeta, DType, StorageLayout};

use std::time::{Duration, Instant};
use tokio::time::sleep;

//1GB
const CACHE_SIZE: u64 = 1024 * 1024 * 1024;

async fn spawn_cluster() -> Vec<Node> {
    let ports = [6101, 6102, 6103];

    let mut nodes = Vec::new();

    for port in ports {
        let addr = format!("127.0.0.1:{}", port);
        let server_addr = addr.clone();

        tokio::spawn(async move {
            start_server(server_addr, CACHE_SIZE)
                .await
                .expect("Server failed");
        });

        nodes.push(Node::new(addr, format!("node-{}", port)));
    }

    sleep(Duration::from_millis(500)).await;

    nodes
}

fn make_tensor(size: usize) -> (TensorMeta, Vec<u8>) {
    let meta = TensorMeta::new(
        DType::F32,
        vec![size],
        StorageLayout::RowMajor,
    )
        .unwrap();

    let data = vec![0f32; size];

    let bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        )
            .to_vec()
    };

    (meta, bytes)
}

async fn benchmark_put_latency(client: &DistributedClient) {
    let (meta, bytes) = make_tensor(256);

    let start = Instant::now();

    client
        .put("bench_put".to_string(), meta, bytes)
        .await
        .unwrap();

    println!("PUT latency: {:?}", start.elapsed());
}

async fn benchmark_get_hit_latency(client: &DistributedClient) {
    let (meta, bytes) = make_tensor(256);

    client
        .put("hit_key".to_string(), meta, bytes)
        .await
        .unwrap();

    let start = Instant::now();

    let _ = client.get("hit_key").await.unwrap();

    println!("GET hit latency: {:?}", start.elapsed());
}

async fn benchmark_get_miss_latency(client: &DistributedClient) {
    let start = Instant::now();

    let _ = client.get("missing_key").await.unwrap();

    println!("GET miss latency: {:?}", start.elapsed());
}

async fn benchmark_throughput(client: DistributedClient) {
    let start = Instant::now();

    let total_ops = 50_000;

    for i in 0..total_ops {
        let key = format!("throughput_key_{}", i);

        let _ = client.get(&key).await;
    }

    let elapsed = start.elapsed();

    let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();

    println!(
        "Throughput: {:.0} ops/sec ({} ops in {:?})",
        ops_per_sec, total_ops, elapsed
    );
}

async fn benchmark_distribution(client: &DistributedClient) {
    let total_keys = 10_000;

    for i in 0..total_keys {
        let (meta, bytes) = make_tensor(16);

        let key = format!("dist_key_{}", i);

        client.put(key, meta, bytes).await.unwrap();
    }

    let stats = client.get_per_server_stats().await.unwrap();

    println!("Distribution:");

    for (i, stat) in stats.iter().enumerate() {
        println!("Node {} entries: {}", i, stat.entries);
    }
}

#[tokio::main]
async fn main() {
    println!("Spawning cluster...");

    let nodes = spawn_cluster().await;

    let client = DistributedClient::new_default(nodes);

    println!("\n---- Distributed PUT latency ----");
    benchmark_put_latency(&client).await;

    println!("\n---- Distributed GET hit latency ----");
    benchmark_get_hit_latency(&client).await;

    println!("\n---- Distributed GET miss latency ----");
    benchmark_get_miss_latency(&client).await;

    println!("\n---- Distributed throughput ----");
    benchmark_throughput(client.clone()).await;

    println!("\n---- Distribution balance ----");
    benchmark_distribution(&client).await;

    println!("\nBenchmark completed.");
}