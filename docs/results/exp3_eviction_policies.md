# exp3_eviction_policies

Every arm replays the identical recorded trace, so the only variable is the eviction policy. Counters are read from the gateway's own /metrics rather than inferred from client-side latency: a client cannot observe an eviction.

| arm | requests | ok | errors | ttft_p50_ms | ttft_p95_ms | ttft_p99_ms | total_p50_ms | total_p95_ms | completion_tokens | tokens_per_s | requests_per_s | trace | policy | evictions | wakes | cold_starts | thrash | drain_timeouts | residency_hit_ratio | fleet_mib | budget_mib |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zipf_s10/lru | 20 | 18 | 2 | 942.2 | 3617.6 | 4126.6 | 23824.5 | 28365.6 | 1116 | 9.9 | 0.16 | zipf_s10 | lru | 0 | 0 | 1 | 0 | 0 | 0.783 | 9585 | 7168 |
