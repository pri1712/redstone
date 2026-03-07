use std::sync::Arc;
use redstone::cluster::distributed_client::DistributedClient;
use redstone::cluster::node::Node;
use redstone::transport::grpc::server::start_server;
use redstone::tensor::meta::{TensorMeta, DType, StorageLayout};

use std::time::{Duration, Instant};
use tokio::time::sleep;

// 10GB
const CACHE_SIZE: u64 = 10 * 1024 * 1024 * 1024;

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

    sleep(Duration::from_secs(2)).await;
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
            data.len() * std::mem::size_of::<f32>(),
        )
            .to_vec()
    };

    (meta, bytes)
}

async fn benchmark_put_latency(client: Arc<DistributedClient>, elements: usize, run_id: &str) {
    const SAMPLES: usize = 200;
    const WARMUP: usize = 50;

    let (meta, bytes) = make_tensor(elements);

    let mut samples = Vec::with_capacity(SAMPLES);

    for i in 0..WARMUP {
        let key = format!("warmup_put_{}_{}", run_id, i);
        let _ = client.put(key, meta.clone(), bytes.clone()).await;
    }

    for i in 0..SAMPLES {
        let key = format!("bench_put_{}_{}", run_id, i);

        let start = Instant::now();

        client.put(key, meta.clone(), bytes.clone()).await.unwrap();

        samples.push(start.elapsed());
    }

    print_latency_stats("PUT", samples);
}

async fn benchmark_get_hit_latency(client: Arc<DistributedClient>, elements: usize, run_id: &str) {
    const SAMPLES: usize = 200;
    const WARMUP: usize = 50;

    let (meta, bytes) = make_tensor(elements);

    for i in 0..SAMPLES + WARMUP {
        let key = format!("hit_key_{}_{}", run_id, i);
        client.put(key, meta.clone(), bytes.clone()).await.unwrap();
    }

    sleep(Duration::from_millis(100)).await;

    let mut samples = Vec::with_capacity(SAMPLES);

    for i in 0..WARMUP {
        let key = format!("hit_key_{}_{}", run_id, i);
        let _ = client.get(&key).await;
    }

    for i in WARMUP..(WARMUP + SAMPLES) {
        let key = format!("hit_key_{}_{}", run_id, i);

        let start = Instant::now();

        let result = client.get(&key).await.unwrap();

        assert!(result.is_some(), "Expected cache hit, got miss for key {}", key);

        samples.push(start.elapsed());
    }

    print_latency_stats("GET HIT", samples);
}

async fn benchmark_get_miss_latency(client: Arc<DistributedClient>, run_id: &str) {
    const SAMPLES: usize = 200;

    let mut samples = Vec::with_capacity(SAMPLES);

    for i in 0..SAMPLES {
        let key = format!("missing_key_{}_{}", run_id, i);

        let start = Instant::now();

        let result = client.get(&key).await.unwrap();

        assert!(result.is_none(), "Expected cache miss, got hit for key {}", key);

        samples.push(start.elapsed());
    }

    print_latency_stats("GET MISS", samples);
}

async fn benchmark_get_miss_throughput(client: Arc<DistributedClient>, run_id: &str) {
    const TOTAL_OPS: usize = 10_000;
    let start = Instant::now();

    for i in 0..TOTAL_OPS {
        let key = format!("throughput_get_miss_key_{}_{}", run_id, i);
        let _ = client.get(&key).await;
    }

    let elapsed = start.elapsed();

    let ops_per_sec = TOTAL_OPS as f64 / elapsed.as_secs_f64();

    println!(
        "Throughput: {:.0} ops/sec ({} ops in {:?})",
        ops_per_sec, TOTAL_OPS, elapsed
    );
}

async fn benchmark_put_throughput(client: Arc<DistributedClient>, elements: usize, run_id: &str) {
    const TOTAL_OPS: usize = 10_000;
    let start = Instant::now();

    let (meta, bytes) = make_tensor(elements);

    for i in 0..TOTAL_OPS {
        let key = format!("throughput_put_key_{}_{}", run_id, i);
        let _ = client.put(key,meta.clone(), bytes.clone()).await;
    }
    let elapsed = start.elapsed();
    let ops_per_sec = TOTAL_OPS as f64 / elapsed.as_secs_f64();
    println!(
        "Throughput: {:.0} ops/sec ({} ops in {:?})",
        ops_per_sec, TOTAL_OPS, elapsed
    );
}

async fn benchmark_distribution(client: Arc<DistributedClient>, elements: usize, run_id: &str) {
    const TOTAL_KEYS: usize = 10_000;

    let (meta, bytes) = make_tensor(elements);

    for i in 0..TOTAL_KEYS {
        let key = format!("dist_key_{}_{}", run_id, i);
        client.put(key, meta.clone(), bytes.clone()).await.unwrap();
    }

    sleep(Duration::from_millis(200)).await;

    let stats = client.get_per_server_stats().await.unwrap();

    println!("\nDistribution across {} nodes:", stats.len());

    let total_entries: u64 = stats.iter().map(|s| s.entries).sum();

    for (i, stat) in stats.iter().enumerate() {
        let percentage = (stat.entries as f64 / total_entries as f64) * 100.0;
        println!(
            "  Node {}: {} entries ({:.1}%)",
            i,
            stat.entries,
            percentage
        );
    }

    let expected_per_node = total_entries as f64 / stats.len() as f64;
    for (i, stat) in stats.iter().enumerate() {
        let deviation = ((stat.entries as f64 - expected_per_node) / expected_per_node * 100.0).abs();
        if deviation > 15.0 {
            println!("Warning: Node {} has {:.1}% deviation from expected distribution", i, deviation);
        }
    }
}

async fn benchmark_sequential_gets_throughput(client: Arc<DistributedClient>, run_id: &str) {
    const NUM_OPS: usize = 5_000;

    for i in 0..NUM_OPS {
        let key = format!("seq_{}_{}", run_id, i);
        let _ = client.get(&key).await;
    }

    let start = Instant::now();

    for i in 0..NUM_OPS {
        let key = format!("seq_{}_{}", run_id, i);
        let _ = client.get(&key).await;
    }

    let elapsed = start.elapsed();
    let ops_per_sec = NUM_OPS as f64 / elapsed.as_secs_f64();

    println!("\nSequential throughput:");
    println!("  Ops: {}", NUM_OPS);
    println!("  Duration: {:?}", elapsed);
    println!("  Throughput: {:.0} ops/sec", ops_per_sec);
}

async fn benchmark_parallel_gets_throughput(client: Arc<DistributedClient>, elements: usize) {
    const TOTAL_OPS: usize = 1_0000;

    let (meta, bytes) = make_tensor(elements);
    for i in 0..TOTAL_OPS {
        let key = format!("throughput_key_{}", i);
        client.put(key, meta.clone(), bytes.clone()).await.unwrap();
    }

    let start = Instant::now();

    let futures: Vec<_> = (0..TOTAL_OPS)
        .map(|i| {
            let client = Arc::clone(&client);
            let key = format!("throughput_key_{}", i);
            async move {
                let _ = client.get(&key).await;
            }
        })
        .collect();

    futures::future::join_all(futures).await;

    let elapsed = start.elapsed();
    let ops_per_sec = TOTAL_OPS as f64 / elapsed.as_secs_f64();

    println!(
        "Throughput [{} elements]: {:.0} ops/sec ({} ops in {:?})",
        elements,
        ops_per_sec,
        TOTAL_OPS,
        elapsed
    );
}

async fn benchmark_batched_gets_throughput(client: Arc<DistributedClient>, run_id: &str) {
    const BATCH_SIZE: usize = 100;
    const NUM_BATCHES: usize = 100;

    let start = Instant::now();

    for batch in 0..NUM_BATCHES {
        let futures: Vec<_> = (0..BATCH_SIZE)
            .map(|i| {
                let key = format!("batch_{}_{}_{}", run_id, batch, i);

                async move {
                    let _ = client.get(&key).await;
                }
            })
            .collect();

        futures::future::join_all(futures).await;
    }

    let elapsed = start.elapsed();
    let total_ops = BATCH_SIZE * NUM_BATCHES;

    let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();

    println!("\nBatched throughput:");
    println!("  Batch size: {}", BATCH_SIZE);
    println!("  Batches: {}", NUM_BATCHES);
    println!("  Total ops: {}", total_ops);
    println!("  Throughput: {:.0} ops/sec", ops_per_sec);
}

async fn benchmark_mixed_workload(client: Arc<DistributedClient>, run_id: &str, elements: usize) {
    const TOTAL_OPS: usize = 5_000;
    const WRITE_RATIO: f64 = 0.2;

    let (meta, bytes) = make_tensor(elements);

    let start = Instant::now();

    let futures: Vec<_> = (0..TOTAL_OPS)
        .map(|i| {
            let key = format!("mixed_{}_{}", run_id, i);

            let meta = meta.clone();
            let bytes = bytes.clone();

            async move {
                let is_write = (i as f64 / TOTAL_OPS as f64) < WRITE_RATIO;

                if is_write {
                    let _ = client.put(key, meta, bytes).await;
                } else {
                    let _ = client.get(&key).await;
                }
            }
        })
        .collect();

    futures::future::join_all(futures).await;

    let elapsed = start.elapsed();
    let ops_per_sec = TOTAL_OPS as f64 / elapsed.as_secs_f64();

    println!("\nMixed workload:");
    println!("  Total ops: {}", TOTAL_OPS);
    println!("  Writes: {}", (TOTAL_OPS as f64 * WRITE_RATIO) as usize);
    println!("  Reads: {}", (TOTAL_OPS as f64 * (1.0 - WRITE_RATIO)) as usize);
    println!("  Throughput: {:.0} ops/sec", ops_per_sec);
}

async fn benchmark_sustained_throughput(client: Arc<DistributedClient>, run_id: &str) {
    const DURATION_SECS: u64 = 10;
    const OPS_PER_INTERVAL: usize = 2000;

    println!("\nSustained throughput ({} seconds)", DURATION_SECS);

    let start = Instant::now();
    let mut interval = 0;

    while start.elapsed() < Duration::from_secs(DURATION_SECS) {

        let interval_start = Instant::now();

        let futures: Vec<_> = (0..OPS_PER_INTERVAL)
            .map(|i| {
                let key = format!("sustained_{}_{}_{}", run_id, interval, i);

                async move {
                    let _ = client.get(&key).await;
                }
            })
            .collect();

        futures::future::join_all(futures).await;

        let elapsed = interval_start.elapsed();
        let throughput = OPS_PER_INTERVAL as f64 / elapsed.as_secs_f64();

        println!("  Interval {} → {:.0} ops/sec", interval, throughput);

        interval += 1;
    }
}

fn print_latency_stats(name: &str, mut samples: Vec<Duration>) {
    samples.sort();

    let n = samples.len();

    if n == 0 {
        println!("{} latency stats: No samples!", name);
        return;
    }

    let min = samples[0];
    let max = samples[n - 1];
    let p50 = samples[n / 2];
    let p95 = samples[(n as f64 * 0.95) as usize];
    let p99 = samples[(n as f64 * 0.99) as usize];

    let sum: Duration = samples.iter().sum();
    let mean = sum / n as u32;

    println!("\n{} latency stats:", name);
    println!("  Samples: {}", n);
    println!("  Min:  {:?}", min);
    println!("  Mean: {:?}", mean);
    println!("  p50:  {:?}", p50);
    println!("  p95:  {:?}", p95);
    println!("  p99:  {:?}", p99);
    println!("  Max:  {:?}", max);
}

#[tokio::main]
async fn main() {
    println!("Starting Distributed Tensor Cache Benchmarks\n");

    let sizes = vec![
        ("tiny_3KB", 768),
        ("small_50KB", 12_800),
        ("medium_1MB", 262_144),
    ];

    let mut port_seed = 6100;

    for (label, elements) in sizes {

        println!("Benchmarking operational latencies for {}",label);

        println!("\nStarting cluster on ports {}-{}...", port_seed, port_seed + 2);
        let nodes = spawn_cluster(port_seed).await;

        let client = Arc::new(DistributedClient::new_default(nodes));

        // let run_id = format!("{}_port{}", label, port_seed);

        // println!("\nPUT benchmarks");
        // benchmark_put_latency(client.clone(), elements, &run_id).await;
        //
        // println!("\nPUT throughput");
        // benchmark_put_throughput(client.clone(), elements, &run_id).await;
        //
        // println!("\nGET benchmarks");
        // benchmark_get_hit_latency(client.clone(), elements, &run_id).await;
        //
        // println!("\nGET miss benchmarks");
        // benchmark_get_miss_latency(client.clone(), &run_id).await;
        //
        // println!("\nGET miss Throughput benchmarks");
        // benchmark_get_miss_throughput(client.clone(), &run_id).await;
        //
        // println!("\nKey distribution benchmarks");
        // benchmark_distribution(client.clone(), elements, &run_id).await;
        //
        // println!("\nCompleted latency benchmarks for {}", label);
        port_seed += 10;

        sleep(Duration::from_secs(1)).await;

        println!("Benchmarking throughput for {}",label);

        println!("Benchmarking concurrent reads");
        benchmark_parallel_gets_throughput(client.clone(), elements).await


    }

}