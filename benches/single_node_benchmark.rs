use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion,
    Throughput, BatchSize, black_box
};
use redstone::TensorCache;
use redstone::tensor::meta::{TensorMeta, DType, StorageLayout};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use std::mem::size_of;

fn make_tensor(elements: usize) -> (TensorMeta, Vec<u8>) {
    let meta = TensorMeta::new(
        DType::F32,
        vec![elements],
        StorageLayout::RowMajor,
    ).unwrap();

    let bytes = vec![0u8; elements * size_of::<f32>()];
    (meta, bytes)
}

fn populate_cache(cache: &TensorCache, count: usize, size_per_entry: usize) {
    for i in 0..count {
        let (meta, bytes) = make_tensor(size_per_entry);
        cache.put(format!("preload_{}", i), meta, bytes).unwrap();
    }
}

fn bench_put_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("01_put_by_size");

    let sizes = vec![
        ("tiny_3KB", 768),
        ("small_50KB", 12_800),
        ("medium_1MB", 262_144),
        ("large_10MB", 2_621_440),
    ];

    for (name, elements) in sizes {
        let size_bytes = elements * size_of::<f32>();
        group.throughput(Throughput::Bytes(size_bytes as u64));

        group.bench_with_input(
            BenchmarkId::new("put", name),
            &elements,
            |b, &elements| {

                b.iter_batched(
                    || {
                        let cache = TensorCache::new(1024 * 1024 * 1024).unwrap();
                        let (meta, bytes) = make_tensor(elements);
                        (cache, meta, bytes)
                    },
                    |(cache, meta, bytes)| {
                        cache.put("key".to_string(), meta, bytes).unwrap();
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_get_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("02_get_hit");

    for (name, elements) in &[("tiny", 768), ("medium", 262_144)] {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("get_hit", name),
            elements,
            |b, &elements| {
                let cache = TensorCache::new(1024 * 1024 * 1024).unwrap();
                let (meta, bytes) = make_tensor(elements);
                cache.put("test_key".to_string(), meta, bytes).unwrap();
                b.iter(|| {
                    black_box(cache.get("test_key"));
                });
            },
        );
    }

    group.finish();
}


fn bench_eviction(c: &mut Criterion) {

    c.bench_function("03_eviction_triggered", |b| {

        b.iter_batched(
            || {
                let cache = TensorCache::new(10 * 1024 * 1024).unwrap();
                populate_cache(&cache, 15, 100_000);
                cache
            },
            |cache| {
                let (meta, bytes) = make_tensor(500_000);
                cache.put("evict_key".to_string(), meta, bytes).unwrap();
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_concurrent_reads(c: &mut Criterion) {
    let mut group = c.benchmark_group("05_concurrent_reads");

    const OPS_PER_THREAD: usize = 1000;

    for &num_threads in &[1, 2, 4, 8] {

        let total_ops = num_threads * OPS_PER_THREAD;
        group.throughput(Throughput::Elements(total_ops as u64));

        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            &num_threads,
            |b, &num_threads| {

                let cache = Arc::new(
                    TensorCache::new(10 * 1024 * 1024).unwrap()
                );
                populate_cache(&cache, 100, 128);

                let keys: Vec<String> = (0..100)
                    .map(|i| format!("preload_{}", i))
                    .collect();

                b.iter(|| {
                    let handles: Vec<_> = (0..num_threads)
                        .map(|_| {
                            let cache_clone = Arc::clone(&cache);
                            let keys = keys.clone();

                            thread::spawn(move || {
                                for i in 0..OPS_PER_THREAD {
                                    let key = &keys[i % 100];
                                    black_box(cache_clone.get(key));
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("08_throughput");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("max_read_ops", |b| {

        let cache = TensorCache::new(1024 * 1024 * 1024).unwrap();
        populate_cache(&cache, 1000, 128);

        let keys: Vec<String> = (0..1000)
            .map(|i| format!("preload_{}", i))
            .collect();

        let mut counter = 0;

        b.iter(|| {
            let key = &keys[counter % 1000];
            black_box(cache.get(key));
            counter += 1;
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_put_sizes,
    bench_get_hit,
    bench_eviction,
    bench_concurrent_reads,
    bench_throughput,
);

criterion_main!(benches);