fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::configure()
        .bytes("redstone.GetResponseChunk.data")
        .bytes("redstone.PutRequest.data")
        .compile_protos(&["proto/redstone.proto"], &["proto"])?;
    Ok(())
}