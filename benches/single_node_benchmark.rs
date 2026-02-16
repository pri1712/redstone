use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use redstone::TensorCache;
use redstone::tensor::meta::{TensorMeta, DType, StorageLayout};

fn make_tensor_bytes(elements: usize) -> (TensorMeta, Vec<u8>) {
    let meta = TensorMeta::new(
        DType::F32,
        vec![elements],
        StorageLayout::RowMajor,
    ).unwrap();

    let bytes = vec![0u8; elements * size_of::<f32>()];
    (meta, bytes)
}

fn bench_put(c: &mut Criterion) {
    let mut group = c.benchmark_group("put");

    for size in [16usize, 128, 1024, 8192] {
        //dynamic number of elements in each tensor. so we have a benchmark for different sizes.
        group.throughput(Throughput::Bytes(
            (size * size_of::<f32>()) as u64,
        ));

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let cache = TensorCache::new(10 * 1024 * 1024).unwrap();
                    let (meta, bytes) = make_tensor_bytes(size);
                    let key = format!("key_{}", size);
                    let _ = cache.put(key, meta, bytes);
                });
            },
        );
    }

    group.finish();
}

fn bench_get_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_hit");

    for size in [16usize, 128, 1024, 8192] {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                let cache = TensorCache::new(10 * 1024 * 1024).unwrap();
                let (meta, bytes) = make_tensor_bytes(size);
                cache.put("hit_key".to_string(), meta, bytes).unwrap();

                b.iter(|| {
                    let _ = cache.get("hit_key");
                });
            },
        );
    }

    group.finish();
}

fn bench_get_miss(c: &mut Criterion) {
    let cache = TensorCache::new(10 * 1024 * 1024).unwrap();

    c.bench_function("get_miss", |b| {
        b.iter(|| {
            let _ = cache.get("missing_key");
        });
    });
}

fn bench_mixed_workload(c: &mut Criterion) {
    let cache = TensorCache::new(10 * 1024 * 1024).unwrap();

    c.bench_function("mixed_put_get_delete", |b| {
        b.iter(|| {
            for i in 0..50 {
                let key = format!("mixed_{}", i);

                let meta = TensorMeta::new(
                    DType::F32,
                    vec![32],
                    StorageLayout::RowMajor,
                ).unwrap();

                let bytes = vec![0u8; 32 * 4];

                let _ = cache.put(key.clone(), meta, bytes);
                let _ = cache.get(&key);
                let _ = cache.delete(&key);
            }
        });
    });
}

criterion_group!(
    benches,
    bench_put,
    bench_get_hit,
    bench_get_miss,
    bench_mixed_workload
);
criterion_main!(benches);