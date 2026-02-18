use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion,
    Throughput, BatchSize, PlotConfiguration, AxisScale
};
use redstone::TensorCache;
use redstone::tensor::meta::{TensorMeta, DType, StorageLayout};
use std::sync::Arc;
use std::thread;

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
        cache.put(format!("preload_{}", i), meta, bytes).ok();
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
                let cache = TensorCache::new(100 * 1024 * 1024).unwrap();

                b.iter_batched(
                    || make_tensor(elements),
                    |(meta, bytes)| {
                        cache.put(format!("key_{}", rand::random::<u32>()), meta, bytes).ok();
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
                let cache = TensorCache::new(100 * 1024 * 1024).unwrap();
                let (meta, bytes) = make_tensor(elements);
                cache.put("test_key".to_string(), meta, bytes).unwrap();

                b.iter(|| cache.get("test_key"));
            },
        );
    }

    group.finish();
}

fn bench_eviction(c: &mut Criterion) {
    let cache = TensorCache::new(10 * 1024 * 1024).unwrap();

    c.bench_function("03_eviction_triggered", |b| {
        b.iter(|| {
            for i in 0..15 {
                let (meta, bytes) = make_tensor(262_144);
                cache.put(format!("evict_{}", i), meta, bytes).ok();
            }
        });
    });
}

fn bench_concurrent_reads(c: &mut Criterion) {
    let mut group = c.benchmark_group("05_concurrent_reads");

    for num_threads in &[1, 2, 4, 8] {
        group.throughput(Throughput::Elements(*num_threads as u64 * 1000));

        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            num_threads,
            |b, &num_threads| {
                let cache = Arc::new(TensorCache::new(10 * 1024 * 1024).unwrap());
                populate_cache(&cache, 100, 128);

                b.iter(|| {
                    let handles: Vec<_> = (0..num_threads).map(|_| {
                        let cache_clone = Arc::clone(&cache);
                        thread::spawn(move || {
                            for i in 0..1000 {
                                cache_clone.get(&format!("preload_{}", i % 100));
                            }
                        })
                    }).collect();

                    handles.into_iter().for_each(|h| h.join().unwrap());
                });
            },
        );
    }

    group.finish();
}

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("08_throughput");
    group.measurement_time(std::time::Duration::from_secs(10));

    group.bench_function("max_read_ops", |b| {
        let cache = TensorCache::new(10 * 1024 * 1024).unwrap();
        populate_cache(&cache, 1000, 128);

        let mut counter = 0;
        b.iter(|| {
            cache.get(&format!("preload_{}", counter % 1000));
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