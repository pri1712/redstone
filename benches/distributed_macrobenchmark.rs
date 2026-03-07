use std::sync::Arc;
use redstone::cluster::distributed_client::DistributedClient;
use redstone::cluster::node::Node;
use redstone::transport::grpc::server::start_server;
use redstone::tensor::meta::{TensorMeta, DType, StorageLayout};

use std::time::{Duration, Instant};
use tokio::time::sleep;
use rand::RngExt;

const CACHE_SIZE: u64 = 20 * 1024 * 1024 * 1024;

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

struct LatencyStats {
    samples: usize,
    min: Duration,
    mean: Duration,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    max: Duration,
}

impl LatencyStats {
    fn from_samples(mut samples: Vec<Duration>) -> Self {
        samples.sort();
        let n = samples.len();

        let min = samples[0];
        let max = samples[n - 1];
        let p50 = samples[n / 2];
        let p95 = samples[(n as f64 * 0.95) as usize];
        let p99 = samples[(n as f64 * 0.99) as usize];
        let sum: Duration = samples.iter().sum();
        let mean = sum / n as u32;

        Self {
            samples: n,
            min,
            mean,
            p50,
            p95,
            p99,
            max,
        }
    }

    fn print(&self, name: &str) {
        println!("{:<20} | {:>8} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
                 name,
                 format!("{:?}", self.min),
                 format!("{:?}", self.mean),
                 format!("{:?}", self.p50),
                 format!("{:?}", self.p95),
                 format!("{:?}", self.p99),
                 format!("{:?}", self.max),
        );
    }
}

struct ThroughputStats {
    total_ops: usize,
    duration: Duration,
    ops_per_sec: f64,
    avg_latency_us: f64,
}

impl ThroughputStats {
    fn new(total_ops: usize, duration: Duration) -> Self {
        let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
        let avg_latency_us = duration.as_micros() as f64 / total_ops as f64;

        Self {
            total_ops,
            duration,
            ops_per_sec,
            avg_latency_us,
        }
    }

    fn print(&self, name: &str) {
        println!("{:<30} | {:>10} | {:>12} | {:>10.0} ops/s | {:>8.2} us",
                 name,
                 self.total_ops,
                 format!("{:?}", self.duration),
                 self.ops_per_sec,
                 self.avg_latency_us,
        );
    }
}

async fn benchmark_put_latency(client: Arc<DistributedClient>, elements: usize, run_id: &str) -> LatencyStats {
    const SAMPLES: usize = 200;
    const WARMUP: usize = 50;

    let (meta, bytes) = make_tensor(elements);

    for i in 0..WARMUP {
        let key = format!("warmup_put_{}_{}", run_id, i);
        let _ = client.put(key, meta.clone(), bytes.clone()).await;
    }

    let mut samples = Vec::with_capacity(SAMPLES);

    for i in 0..SAMPLES {
        let key = format!("bench_put_{}_{}", run_id, i);
        let start = Instant::now();
        client.put(key, meta.clone(), bytes.clone()).await.unwrap();
        samples.push(start.elapsed());
    }

    LatencyStats::from_samples(samples)
}

async fn benchmark_get_hit_latency(client: Arc<DistributedClient>, elements: usize, run_id: &str) -> LatencyStats {
    const SAMPLES: usize = 200;
    const WARMUP: usize = 50;

    let (meta, bytes) = make_tensor(elements);

    for i in 0..SAMPLES + WARMUP {
        let key = format!("hit_key_{}_{}", run_id, i);
        client.put(key, meta.clone(), bytes.clone()).await.unwrap();
    }

    sleep(Duration::from_millis(100)).await;

    for i in 0..WARMUP {
        let key = format!("hit_key_{}_{}", run_id, i);
        let _ = client.get(&key).await;
    }

    let mut samples = Vec::with_capacity(SAMPLES);

    for i in WARMUP..(WARMUP + SAMPLES) {
        let key = format!("hit_key_{}_{}", run_id, i);
        let start = Instant::now();
        let result = client.get(&key).await.unwrap();
        assert!(result.is_some(), "Expected cache hit, got miss for key {}", key);
        samples.push(start.elapsed());
    }

    LatencyStats::from_samples(samples)
}

async fn benchmark_get_miss_latency(client: Arc<DistributedClient>, run_id: &str) -> LatencyStats {
    const SAMPLES: usize = 200;

    let mut samples = Vec::with_capacity(SAMPLES);

    for i in 0..SAMPLES {
        let key = format!("missing_key_{}_{}", run_id, i);
        let start = Instant::now();
        let result = client.get(&key).await.unwrap();
        assert!(result.is_none(), "Expected cache miss, got hit for key {}", key);
        samples.push(start.elapsed());
    }

    LatencyStats::from_samples(samples)
}

async fn benchmark_sequential_gets_throughput(client: Arc<DistributedClient>, run_id: &str) -> ThroughputStats {
    const NUM_OPS: usize = 5_000;

    let start = Instant::now();

    for i in 0..NUM_OPS {
        let key = format!("seq_{}_{}", run_id, i);
        let _ = client.get(&key).await;
    }

    ThroughputStats::new(NUM_OPS, start.elapsed())
}

async fn benchmark_parallel_gets_throughput(client: Arc<DistributedClient>, elements: usize, run_id: &str) -> ThroughputStats {
    const TOTAL_OPS: usize = 10_000;

    let (meta, bytes) = make_tensor(elements);
    for i in 0..TOTAL_OPS {
        let key = format!("concurrent_{}_{}", i, run_id);
        client.put(key, meta.clone(), bytes.clone()).await.unwrap();
    }

    let start = Instant::now();

    let futures: Vec<_> = (0..TOTAL_OPS)
        .map(|i| {
            let client = Arc::clone(&client);
            let key = format!("concurrent_{}_{}", i, run_id);
            async move {
                let _ = client.get(&key).await;
            }
        })
        .collect();

    futures::future::join_all(futures).await;

    ThroughputStats::new(TOTAL_OPS, start.elapsed())
}

async fn benchmark_batched_gets_throughput(client: Arc<DistributedClient>, run_id: &str) -> ThroughputStats {
    const BATCH_SIZE: usize = 100;
    const NUM_BATCHES: usize = 100;

    let start = Instant::now();

    for batch in 0..NUM_BATCHES {
        let futures: Vec<_> = (0..BATCH_SIZE)
            .map(|i| {
                let client = Arc::clone(&client);
                let key = format!("batch_{}_{}_{}", run_id, batch, i);
                async move {
                    let _ = client.get(&key).await;
                }
            })
            .collect();

        futures::future::join_all(futures).await;
    }

    ThroughputStats::new(BATCH_SIZE * NUM_BATCHES, start.elapsed())
}

async fn benchmark_mixed_workload(client: Arc<DistributedClient>, run_id: &str, elements: usize) -> ThroughputStats {
    const TOTAL_OPS: usize = 5_000;
    const WRITE_RATIO: f64 = 0.2;

    let (meta, bytes) = make_tensor(elements);

    for i in 0..TOTAL_OPS {
        let key = format!("mixed_{}_{}", run_id, i);
        let _ = client.put(key, meta.clone(), bytes.clone()).await;
    }

    let mut rng = rand::rng();

    let start = Instant::now();

    let futures: Vec<_> = (0..TOTAL_OPS)
        .map(|_| {
            let client = Arc::clone(&client);
            let is_write = rng.random_bool(WRITE_RATIO);
            let key_idx = rng.random_range(0..TOTAL_OPS);
            let key = format!("mixed_{}_{}", run_id, key_idx);
            let meta = meta.clone();
            let bytes = bytes.clone();

            async move {
                if is_write {
                    let _ = client.put(key, meta, bytes).await;
                } else {
                    let _ = client.get(&key).await;
                }
            }
        })
        .collect();

    futures::future::join_all(futures).await;

    ThroughputStats::new(TOTAL_OPS, start.elapsed())
}

async fn benchmark_sustained_throughput(client: Arc<DistributedClient>, run_id: &str) {
    const DURATION_SECS: u64 = 10;
    const OPS_PER_INTERVAL: usize = 2000;

    println!("\nSustained Throughput ({} seconds):", DURATION_SECS);

    let start = Instant::now();
    let mut interval = 0;
    let mut throughputs = Vec::new();

    while start.elapsed() < Duration::from_secs(DURATION_SECS) {
        let interval_start = Instant::now();

        let futures: Vec<_> = (0..OPS_PER_INTERVAL)
            .map(|i| {
                let client = Arc::clone(&client);
                let key = format!("sustained_{}_{}_{}", run_id, interval, i);
                async move {
                    let _ = client.get(&key).await;
                }
            })
            .collect();

        futures::future::join_all(futures).await;

        let elapsed = interval_start.elapsed();
        let throughput = OPS_PER_INTERVAL as f64 / elapsed.as_secs_f64();
        throughputs.push(throughput);

        interval += 1;
    }

    // Summary statistics instead of printing every interval
    let avg = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
    let min = throughputs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = throughputs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let std_dev = (throughputs.iter()
        .map(|&x| (x - avg).powi(2))
        .sum::<f64>() / throughputs.len() as f64)
        .sqrt();

    println!("  Intervals: {}", throughputs.len());
    println!("  Average:   {:.0} ops/s", avg);
    println!("  Min:       {:.0} ops/s", min);
    println!("  Max:       {:.0} ops/s", max);
    println!("  Std Dev:   {:.0} ops/s", std_dev);
    println!("  CV:        {:.2}%", (std_dev / avg) * 100.0);
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
    let total_entries: u64 = stats.iter().map(|s| s.entries).sum();

    println!("\nDistribution across {} nodes:", stats.len());
    for (i, stat) in stats.iter().enumerate() {
        let percentage = (stat.entries as f64 / total_entries as f64) * 100.0;
        println!("  Node {}: {:>6} entries ({:>5.1}%)", i, stat.entries, percentage);
    }
}

#[tokio::main]
async fn main() {
    println!("Distributed Tensor Cache Benchmarks");

    let sizes = vec![
        ("tiny_3KB", 768),
        ("small_50KB", 12_800),
        ("medium_1MB", 262_144),
    ];

    let mut port_seed = 6100;

    for (label, elements) in sizes {
        print!("Benchmarking {}",label);

        println!("\nStarting cluster on ports {}-{}...", port_seed, port_seed + 2);
        let nodes = spawn_cluster(port_seed).await;
        let client = Arc::new(DistributedClient::new_default(nodes));
        let run_id = format!("{}_port{}", label, port_seed);

        println!("\n--- LATENCY BENCHMARKS ---");
        println!("{:<20} | {:>8} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
                 "Operation", "Min", "Mean", "p50", "p95", "p99", "Max");
        println!("{:-<20}-+-{:-<8}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}",
                 "", "", "", "", "", "", "");

        let put_stats = benchmark_put_latency(client.clone(), elements, &run_id).await;
        put_stats.print("PUT");

        let get_hit_stats = benchmark_get_hit_latency(client.clone(), elements, &run_id).await;
        get_hit_stats.print("GET HIT");

        let get_miss_stats = benchmark_get_miss_latency(client.clone(), &run_id).await;
        get_miss_stats.print("GET MISS");

        println!("\n--- THROUGHPUT BENCHMARKS ---");
        println!("{:<30} | {:>10} | {:>12} | {:>13} | {:>10}",
                 "Operation", "Total Ops", "Duration", "Throughput", "Avg Latency");
        println!("{:-<30}-+-{:-<10}-+-{:-<12}-+-{:-<13}-+-{:-<10}",
                 "", "", "", "", "");

        let seq_stats = benchmark_sequential_gets_throughput(client.clone(), &run_id).await;
        seq_stats.print("Sequential GET");

        let par_stats = benchmark_parallel_gets_throughput(client.clone(), elements, &run_id).await;
        par_stats.print("Parallel GET");

        let batch_stats = benchmark_batched_gets_throughput(client.clone(), &run_id).await;
        batch_stats.print("Batched GET");

        let mixed_stats = benchmark_mixed_workload(client.clone(), &run_id, elements).await;
        mixed_stats.print("Mixed (20% write)");

        benchmark_sustained_throughput(client.clone(), &run_id).await;

        println!("\n--- KEY DISTRIBUTION ---");
        benchmark_distribution(client.clone(), elements, &run_id).await;

        port_seed += 10;
        sleep(Duration::from_secs(1)).await;
    }

    println!("All benchmarks completed");
}