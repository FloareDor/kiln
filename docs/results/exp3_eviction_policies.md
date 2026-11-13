# exp3_eviction_policies

Every arm replays the identical recorded trace, so the only variable is the eviction policy. Counters are read from the gateway's own /metrics rather than inferred from client-side latency: a client cannot observe an eviction.

| arm | requests | ok | errors | ttft_p50_ms | ttft_p95_ms | ttft_p99_ms | total_p50_ms | total_p95_ms | completion_tokens | tokens_per_s | requests_per_s | trace | policy | evictions | wakes | cold_starts | thrash | drain_timeouts | residency_hit_ratio | fleet_mib | budget_mib |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zipf_s10/lru | 191 | 32 | 159 | 99.0 | 216.9 | 49594.6 | 2359.2 | 2941.8 | 2021 | 22.9 | 0.36 | zipf_s10 | lru | 0 | 1 | 1 | 15 | 0 | 0.187 | 9585 | 7168 |
| zipf_s10/cost-aware | 12 | 10 | 2 | 1020.1 | 10882.8 | 10882.8 | 22885.7 | 37281.0 | 604 | 8.9 | 0.15 | zipf_s10 | cost-aware | 0 | 0 | 1 | 0 | 0 | 0.714 | 9585 | 7168 |
