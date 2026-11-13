# exp3_eviction_policies

Every arm replays the identical recorded trace, so the only variable is the eviction policy. Counters are read from the gateway's own /metrics rather than inferred from client-side latency: a client cannot observe an eviction.

| arm | requests | ok | errors | ttft_p50_ms | ttft_p95_ms | ttft_p99_ms | total_p50_ms | total_p95_ms | completion_tokens | tokens_per_s | requests_per_s | trace | policy | evictions | wakes | cold_starts | thrash | drain_timeouts | residency_hit_ratio | fleet_mib | budget_mib |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zipf_s10/cold | 50 | 7 | 43 | 959.5 | 35666.8 | 35666.8 | 2768.7 | 37531.7 | 425 | 2.5 | 0.04 | zipf_s10 | cold | 4 | 4 | 1 | 0 | 0 | 0.111 | 9666 | 7168 |
| zipf_s10/lru | 125 | 19 | 106 | 84.1 | 7951.7 | 33575.5 | 1953.2 | 13342.7 | 1108 | 6.5 | 0.11 | zipf_s10 | lru | 3 | 4 | 1 | 0 | 0 | 0.359 | 9666 | 7168 |
| zipf_s10/lfu-decay | 167 | 26 | 141 | 75.1 | 2509.6 | 12607.8 | 1882.7 | 7865.9 | 1520 | 8.6 | 0.15 | zipf_s10 | lfu-decay | 3 | 5 | 0 | 0 | 0 | 0.438 | 9666 | 7168 |
| zipf_s10/cost-aware | 292 | 13 | 279 | 75.7 | 898.3 | 1790.8 | 1854.0 | 2680.3 | 737 | 4.1 | 0.07 | zipf_s10 | cost-aware | 0 | 3 | 0 | 66 | 0 | 0.11 | 9666 | 7168 |
| zipf_s10/predictive-prewarm | 197 | 29 | 168 | 75.2 | 4392.1 | 5738.9 | 1874.8 | 10862.7 | 1702 | 9.5 | 0.16 | zipf_s10 | predictive-prewarm | 7 | 9 | 0 | 7 | 0 | 0.344 | 9666 | 7168 |
| bursty/cold | 300 | 8 | 292 | 92.0 | 2587.8 | 2587.8 | 1858.1 | 7058.6 | 438 | 2.4 | 0.04 | bursty | cold | 1 | 4 | 0 | 73 | 0 | 0.082 | 9666 | 7168 |
| bursty/lru | 202 | 24 | 178 | 96.3 | 4179.1 | 36445.9 | 1895.9 | 7312.4 | 1392 | 7.8 | 0.13 | bursty | lru | 6 | 7 | 1 | 0 | 0 | 0.459 | 9666 | 7168 |
| bursty/lfu-decay | 290 | 10 | 280 | 78.6 | 2951.2 | 2951.2 | 1755.9 | 7458.6 | 540 | 3.0 | 0.06 | bursty | lfu-decay | 2 | 5 | 0 | 71 | 0 | 0.06 | 9666 | 7168 |
| bursty/cost-aware | 284 | 8 | 276 | 87.7 | 2534.0 | 2534.0 | 1869.7 | 6994.0 | 438 | 2.4 | 0.04 | bursty | cost-aware | 1 | 4 | 0 | 73 | 0 | 0.049 | 9666 | 7168 |
| bursty/predictive-prewarm | 0 | 0 | 0 | nan | nan | nan | nan | nan | 0 | 0.0 | 0.0 | bursty | predictive-prewarm | 5 | 7 | 0 | 0 | 0 | 0.541 | 9666 | 7168 |
