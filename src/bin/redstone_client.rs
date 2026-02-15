use redstone::transport::grpc::client::RemoteCacheClient;
use redstone::tensor::meta::{TensorMeta, DType, StorageLayout};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let server_addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:50051".to_string());

    println!("Connecting to Redstone cache at {}...", server_addr);

    let mut client = RemoteCacheClient::connect(server_addr).await?;

    // Example 1: Put a tensor
    println!("Example 1: Put a tensor");
    let meta = TensorMeta::new(
        DType::F32,
        vec![10, 20],
        StorageLayout::RowMajor,
    )?;

    let data = vec![1.0f32; 200];
    let bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        )
            .to_vec()
    };

    match client.put("test_tensor".to_string(), meta, bytes).await {
        Ok(_) => println!("Put succeeded"),
        Err(e) => println!("Put failed: {}", e),
    }

    // Example 2: Get the tensor back
    println!("\nExample 2: Get the tensor");
    match client.get("test_tensor".parse()?).await {
        Ok(Some(tensor)) => {
            println!("Found tensor");
            println!("Shape: {:?}", tensor.get_metadata().shape());
            println!("Dtype: {:?}", tensor.get_metadata().dtype());
            println!("Size: {} bytes", tensor.get_data().len());
        }
        Ok(None) => {
            println!("Tensor not found");
        }
        Err(e) => {
            println!("Get failed: {}", e);
        }
    }

    // Example 3: Try to get non-existent key
    println!("\nExample 3: Get non-existent key");
    match client.get("nonexistent".parse()?).await {
        Ok(Some(_)) => println!("Found (unexpected)"),
        Ok(None) => println!("Key not found (expected)"),
        Err(e) => println!("Error: {}", e),
    }

    // Example 4: Put multiple tensors
    println!("\nExample 4: Put multiple tensors");
    for i in 0..5 {
        let key = format!("tensor_{}", i);
        let meta = TensorMeta::new(
            DType::F32,
            vec![5, 5],
            StorageLayout::RowMajor,
        )?;

        let data = vec![i as f32; 25];
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * 4,
            )
                .to_vec()
        };

        client.put(key.clone(), meta, bytes).await?;
        println!("Put {}", key);
    }

    // Example 5: Get cache stats
    println!("\nExample 5: Cache statistics");
    match client.get_stats().await {
        Ok(stats) => {
            println!("   Entries: {}", stats.entries);
            println!("   Memory used: {} bytes", stats.memory_used);
            println!("   Memory limit: {} bytes", stats.memory_limit);
            println!("   Hit rate: {:.1}%", stats.hit_rate * 100.0);
            println!("   Memory utilization: {:.1}%", stats.memory_utilization * 100.0);
            println!("   Hits: {}", stats.hits);
            println!("   Misses: {}", stats.misses);
            println!("   Evictions: {}", stats.evictions);
        }
        Err(e) => {
            println!("GetStats failed: {}", e);
        }
    }

    // Example 6: Delete a tensor
    println!("\n🗑️  Example 6: Delete a tensor");
    match client.delete("tensor_0".parse()?).await {
        Ok(..) => println!("Deleted tensor_0"),
        Err(e) => println!("Delete failed: {}", e),
    }

    println!("\nExample 7: Try duplicate put");

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
        Err(e) => println!("Second put failed as expected: {}", e),
    }

    // Final stats
    println!("\nFinal cache statistics");
    let stats = client.get_stats().await?;
    println!("   Total entries: {}", stats.entries);
    println!("   Total operations: {}", stats.hits + stats.misses);
    println!("   Hit rate: {:.1}%", stats.hit_rate * 100.0);

    println!("\nAll examples completed!");

    Ok(())
}