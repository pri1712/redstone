use redstone::cluster::config::config_file::ClusterClientFileConfig;
use redstone::cluster::distributed_client::DistributedClient;
use redstone::tensor::meta::{TensorMeta, DType, StorageLayout};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "client_config.toml".to_string());

    println!("Loading cluster config from {}", config_path);

    let file_config = ClusterClientFileConfig::load(&config_path)?;
    let (nodes, runtime_config) = file_config.into_runtime();
    if nodes.is_empty() {
        return Err("No nodes provided.".into());
    }
    println!("Initializing distributed client with {} nodes", nodes.len());

    let client = DistributedClient::new_with_config(nodes, runtime_config);

    println!("\nExample 1: Put a tensor");

    let meta = TensorMeta::new(
        DType::F32,
        vec![10, 20],
        StorageLayout::RowMajor,
    )?;

    let data = vec![1.0f32; 200];
    let bytes = f32_to_bytes(&data);

    client.put("test_tensor".to_string(), meta, bytes).await?;
    println!("Put succeeded");


    println!("\nExample 2: Get tensor");

    match client.get("test_tensor").await? {
        Some(tensor) => {
            println!("Found tensor");
            println!("Shape: {:?}", tensor.get_metadata().shape());
            println!("Dtype: {:?}", tensor.get_metadata().dtype());
            println!("Size: {} bytes", tensor.get_data().len());
        }
        None => println!("Tensor not found"),
    }

    println!("\nExample 3: Get non-existent key");

    let missing = client.get("nonexistent").await?;
    println!("Found? {}", missing.is_some());

    println!("\nExample 4: Put multiple tensors");

    for i in 0..5 {
        let key = format!("tensor_{}", i);

        let meta = TensorMeta::new(
            DType::F32,
            vec![5, 5],
            StorageLayout::RowMajor,
        )?;

        let data = vec![i as f32; 25];
        let bytes = f32_to_bytes(&data);
        client.put(key.clone(), meta, bytes).await?;
        println!("Put {}", key);
    }

    println!("\nExample 5: Per-server stats");

    let stats = client.get_per_server_stats().await?;

    for (i, stat) in stats.iter().enumerate() {
        println!("Node {}:", i);
        println!("  Entries: {}", stat.entries);
        println!("  Memory used: {}", stat.memory_used);
        println!("  Hit rate: {:.2}%", stat.hit_rate() * 100.0);
    }

    println!("\nExample 6: Delete tensor");

    client.delete("tensor_0").await?;
    println!("Deleted tensor_0");

    println!("\nExample 7: Duplicate put");

    let meta1 = TensorMeta::new(
        DType::F32,
        vec![2, 2],
        StorageLayout::RowMajor,
    )?;

    let data = vec![0u8; 16];

    client.put("duplicate".to_string(), meta1, data.clone()).await?;
    println!("First put succeeded");

    let meta2 = TensorMeta::new(
        DType::F32,
        vec![2, 2],
        StorageLayout::RowMajor,
    )?;

    match client.put("duplicate".to_string(), meta2, data).await {
        Ok(_) => println!("Second put succeeded (unexpected)"),
        Err(e) => println!("Second put failed as expected: {:?}", e),
    }

    println!("\nAll distributed examples completed.");

    Ok(())
}

fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * std::mem::size_of::<f32>());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

