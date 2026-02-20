use redstone::transport::grpc::server::start_server;
use redstone::transport::grpc::client::RemoteCacheClient;
use redstone::tensor::meta::{TensorMeta, DType, StorageLayout};

use tokio::time::{sleep, Duration};
use std::sync::Arc;
use rand::{rng, RngExt};

const THREADS: u64 = 20;
const OPS_PER_THREAD: u64 = 200;

fn random_port() -> u16 {
    rng().random_range(50060..60000)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn stress_test_concurrent_clients() {

   let addr = server_setup().await;

    sleep(Duration::from_millis(300)).await;

    let addr_arc = Arc::new(addr);

    let mut handles = vec![];

    for thread_id in 0..THREADS {
        let addr = addr_arc.clone();

        let handle = tokio::spawn(async move {
            let mut client = RemoteCacheClient::connect((*addr).clone())
                .await
                .expect("Client connect failed");

            for op in 0..OPS_PER_THREAD {
                let key = format!("key_{}_{}", thread_id, op % 10);

                let meta = TensorMeta::new(
                    DType::F32,
                    vec![4],
                    StorageLayout::RowMajor,
                ).unwrap();

                let bytes = vec![0u8; 16];

                match op % 3 {
                    0 => {
                        let _ = client.put(key.clone(), meta, bytes).await;
                    }
                    1 => {
                        let _ = client.get(key.clone()).await;
                    }
                    _ => {
                        let _ = client.delete(key.clone()).await;
                    }
                }
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.await.expect("Thread panicked");
    }

    let mut client = RemoteCacheClient::connect((*addr_arc).clone())
        .await
        .expect("Final client failed");

    let stats = client.get_stats().await.expect("Stats failed");

    println!("Final stats:");
    println!("Entries: {}", stats.entries);
    println!("Hits: {}", stats.hits);
    println!("Misses: {}", stats.misses);
    println!("Evictions: {}", stats.evictions);
    assert!(stats.hits+stats.misses>0);
    assert!(stats.memory_used <= stats.memory_limit);
}
async fn server_setup() -> String {
    let port = random_port();
    let addr = format!("127.0.0.1:{}", port);
    let server_addr = addr.clone();
    tokio::spawn(async move {
        start_server(server_addr, 1024)
            .await
            .expect("Server failed");
    });
    addr
}