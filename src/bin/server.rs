// Binary to run the Redstone cache server

use redstone::transport::grpc::server::start_server;
use std::env;

const DEFAULT_CACHE_SIZE : u64= 1024 * 1024 * 1024;
const DEFAULT_SERVER_PORT : &str = "127.0.0.1:50051";
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let addr = if args.len() > 1 {
        args[1].clone()
    } else {
        DEFAULT_SERVER_PORT.to_string()
    };

    let cache_size = if args.len() > 2 {
        args[2].parse::<u64>()
            .expect("Invalid cache size")
    } else {
        DEFAULT_CACHE_SIZE
    };

    println!("Starting Redstone cache server...");
    println!("Address: {}", addr);
    println!("Cache size: {} bytes ({:.2} GB)",
             cache_size,
             cache_size as f64 / 1024.0 / 1024.0 / 1024.0);

    start_server(addr, cache_size).await?;

    Ok(())
}