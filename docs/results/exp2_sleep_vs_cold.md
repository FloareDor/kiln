# exp2_sleep_vs_cold

Measured by the deploy job on the local RTX 3070 (8 GB), vLLM 0.32.0, `--enforce-eager`. Wake L2 includes the `collective_rpc{"method": "reload_weights"}` step that vLLM's own `/wake_up` omits; without it the wake is faster and produces garbage (see docs/PROGRESS.md, M0).

| model | vram_mib | cold_start_s | wake_l1_s | wake_l2_s | l1_speedup_vs_cold | l2_speedup_vs_cold |
|---|---|---|---|---|---|---|
| qwen-0.5b | 2458 | 35.33 | 0.592 | 0.706 | 59.7 | 50.0 |
| qwen3-0.6b | 2621 | 35.94 | 0.823 | 1.047 | 43.7 | 34.3 |
| qwen-1.5b | 4506 | 37.37 | 1.54 | 1.813 | 24.3 | 20.6 |
