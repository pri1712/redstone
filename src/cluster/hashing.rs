use std::hash::Hasher;
use twox_hash::XxHash64;

pub trait HashFunction: Sync + Send {
    fn hash(&self, key: &str) -> u64;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct XxHash;

impl HashFunction for XxHash {
    fn hash(&self, key: &str) -> u64 {
        let mut hasher = XxHash64::with_seed(0);
        hasher.write(key.as_bytes());
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_deterministic_hash() {
        let hasher = XxHash;
        let key = "test_key_123";

        let hash1 = hasher.hash(key);
        let hash2 = hasher.hash(key);

        assert_eq!(hash1, hash2, "Hash must be deterministic");
    }

    #[test]
    fn test_different_keys_different_hashes() {
        let hasher = XxHash;

        let h1 = hasher.hash("key1");
        let h2 = hasher.hash("key2");
        let h3 = hasher.hash("key3");

        assert_ne!(h1, h2);
        assert_ne!(h2, h3);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_distribution_quality() {
        let hasher = XxHash;
        let num_buckets = 100;
        let mut buckets = vec![0; num_buckets];

        for i in 0..10_000 {
            let key = format!("key_{}", i);
            let hash = hasher.hash(&key);
            let bucket = (hash % num_buckets as u64) as usize;
            buckets[bucket] += 1;
        }

        for (i, count) in buckets.iter().enumerate() {
            assert!(
                (50..=150).contains(count),
                "Bucket {} has {} keys (expected ~100)",
                i,
                count
            );
        }
    }
}