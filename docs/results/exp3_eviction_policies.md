# exp3_eviction_policies

Every arm replays the identical recorded trace, so the only variable is the eviction policy. Counters are read from the gateway's own /metrics rather than inferred from client-side latency: a client cannot observe an eviction.

| arm | requests | offered_rps | ok | errors | ttft_p50_ms | ttft_p95_ms | ttft_p99_ms | total_p50_ms | total_p95_ms | completion_tokens | tokens_per_s | requests_per_s | trace | policy | evictions | wakes | cold_starts | thrash | drain_timeouts | residency_hit_ratio | fleet_mib | budget_mib |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zipf_s10/lru | 17 | 0.32 | 17 | 0 | 1075.5 | 26665.6 | 28379.5 | 7231.4 | 27616.1 | 272 | 5.1 | 0.32 | zipf_s10 | lru | 4 | 7 | 0 | 20 | 1 | 0.101 | 8355 | 7168 |
