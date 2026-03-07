use redstone::cluster::distributed_client::DistributedClient;
use redstone::cluster::node::Node;
use redstone::transport::grpc::server::start_server;
use redstone::tensor::meta::{TensorMeta, DType, StorageLayout};

use std::time::{Duration, Instant};
use tokio::time::sleep;

// 10GB
const CACHE_SIZE: u64 = 1024 * 1024 * 1024 * 1024;

async fn spawn_cluster(base_port: u16) -> Vec<Node> {
    let ports = [base_port, base_port + 1, base_port + 2];

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
    sleep(Duration::from_millis(800)).await;
    nodes
}

fn make_tensor(elements: usize) -> (TensorMeta, Vec<u8>) {
    let meta = TensorMeta::new(
        DType::F32,
        vec![elements],
        StorageLayout::RowMajor,
    )
        .unwrap();

    let data = vec![0f32; elements];

    let bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * size_of::<f32>(),
        )
            .to_vec()
    };

    (meta, bytes)
}

async fn benchmark_put_latency(client: &DistributedClient, elements: usize) {
    const SAMPLES: usize = 200;
    const WARMUP: usize = 50;

    let (meta, bytes) = make_tensor(elements);

    let mut samples = Vec::with_capacity(SAMPLES);

    for i in 0..WARMUP {
        let key = format!("warmup_put_{}", i);
        let _ = client.put(key, meta.clone(), bytes.clone()).await;
    }

    for i in 0..SAMPLES {
        let key = format!("bench_put_{}", i);

        let start = Instant::now();

        client.put(key, meta.clone(), bytes.clone()).await.unwrap();

        samples.push(start.elapsed());
    }

    print_latency_stats("PUT", samples);
}

async fn benchmark_get_hit_latency(client: &DistributedClient, elements: usize) {
    const SAMPLES: usize = 200;
    const WARMUP: usize = 50;

    let (meta, bytes) = make_tensor(elements);

    for i in 0..SAMPLES {
        let key = format!("hit_key_{}", i);
        client.put(key, meta.clone(), bytes.clone()).await.unwrap();
    }

    let mut samples = Vec::with_capacity(SAMPLES);

    for i in 0..WARMUP {
        let key = format!("hit_key_{}", i);
        let _ = client.get(&key).await;
    }

    for i in 0..SAMPLES {
        let key = format!("hit_key_{}", i);

        let start = Instant::now();

        let _ = client.get(&key).await.unwrap();

        samples.push(start.elapsed());
    }

    print_latency_stats("GET HIT", samples);
}

async fn benchmark_get_miss_latency(client: &DistributedClient) {
    const SAMPLES: usize = 200;

    let mut samples = Vec::with_capacity(SAMPLES);

    for i in 0..SAMPLES {
        let key = format!("missing_key_{}", i);

        let start = Instant::now();

        let _ = client.get(&key).await.unwrap();

        samples.push(start.elapsed());
    }

    print_latency_stats("GET MISS", samples);
}

async fn benchmark_throughput(client: &DistributedClient) {
    let total_ops = 100_000;

    let start = Instant::now();

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

async fn benchmark_distribution(client: &DistributedClient, elements: usize) {
    let total_keys = 10_000;

    for i in 0..total_keys {
        let (meta, bytes) = make_tensor(elements);
        let key = format!("dist_key_{}", i);

        client.put(key, meta, bytes).await.unwrap();
    }

    let stats = client.get_per_server_stats().await.unwrap();

    println!("Distribution:");

    for (i, stat) in stats.iter().enumerate() {
        println!("Node {} entries: {}", i, stat.entries);
    }
}

fn print_latency_stats(name: &str, mut samples: Vec<Duration>) {
    samples.sort();

    let n = samples.len();

    let p50 = samples[n / 2];
    let p95 = samples[(n as f64 * 0.95) as usize];
    let p99 = samples[(n as f64 * 0.99) as usize];

    println!("\n{} latency stats:", name);
    println!("samples: {}", n);
    println!("p50: {:?}", p50);
    println!("p95: {:?}", p95);
    println!("p99: {:?}", p99);
}

#[tokio::main]
async fn main() {
    let sizes = vec![
        ("tiny_3KB", 768),
        ("small_50KB", 12_800),
        ("medium_1MB", 262_144),
    ];

    let mut port_seed = 6100;

    for (label, elements) in sizes {
        println!("\n==============================");
        println!("Tensor size benchmark: {}", label);
        println!("Elements: {}", elements);
        println!("==============================");

        let nodes = spawn_cluster(port_seed).await;
        port_seed += 10;

        let client = DistributedClient::new_default(nodes);

        println!("\n---- PUT latency ----");
        benchmark_put_latency(&client, elements).await;

        println!("\n---- GET hit latency ----");
        benchmark_get_hit_latency(&client, elements).await;

        println!("\n---- GET miss latency ----");
        benchmark_get_miss_latency(&client).await;

        println!("\n---- Throughput ----");
        benchmark_throughput(&client).await;

        println!("\n---- Distribution ----");
        benchmark_distribution(&client, elements).await;

        println!("\nRun completed for {}", label);

        sleep(Duration::from_secs(1)).await;
    }

    println!("\nAll distributed benchmarks completed.");
}