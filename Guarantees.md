# System guarantees (as of v0)
1. Ultra low latency data access.
2. Eventual consistency with best-effort replication, thus prioritizing reads over writes.
3. Non durable writes, i.e. writes do not survive node restarts.
4. No read-after-write guarantees across nodes.
5. Cache misses are not automatically backfilled and must be handled by the client.
6. Keys may be evicted at any time under memory pressure.
7. Immutable, write-once tensor keys.
8. Atomic tensor operations - tensors are either fully present or absent on a particular node (no partial reads/writes).
9. No ordering guarantees - operations may complete in any order.
10. Network partitions may cause temporary inconsistencies between nodes.
11. Clients are responsible for retry logic on transient failures.


# Non goals (as of v0)
1. Durability in case of node failures.
2. Automatic backfilling of data.
3. ACID transaction support across multiple keys.
4. Distributed locks or coordination primitives.
5. Query/scan operations (no list-all-keys, no range queries).
6. Authentication, authorization, or encryption (security is out of scope).
7. Compression or encoding of tensor data (store raw bytes as-is).
8. Guaranteed fair scheduling or QoS between clients.