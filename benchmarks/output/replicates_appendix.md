
## Replicates and measurement noise

This file's headline run is the Lyon replicate (column 1). A second complete local run (column 2) was made first on the CERN dev box, but that machine is shared (1-min load average 40-60 on 48 cores while it ran) and the run was done at `nice -n 10`, which inflated single-thread timings by about 2x relative to the earlier 2026-08 full run (per-event no-dedup build 74 s vs the 33.6 s that both the 2026-08 run and the Lyon run measured). Column 3 is a *repeat of stage B only* (items 3-4) on the same loaded box. Full local run: `RESULTS_local_loaded_run.md`.

| quantity | Lyon (primary) | local, loaded (run 1) | local, loaded (items 3-4 repeated) |
|---|---:|---:|---:|
| fwd, 1e5 events (ms) | 13.4 | 10.6 | 10.6 |
| grad, 1e5 events (ms) | 72.7 | 47.4 | 47.4 |
| fwd, 1e6 events (ms) | 123 | 148 | 148 |
| grad, 1e6 events (ms) | 568 | 423 | 423 |
| grad/fwd, 1e5 events (Poisson) | 5.43 | 4.48 | 4.48 |
| oscillation component at 1e6 events (ms) | 48.5 | 37.1 | 37.1 |
| spline-product component at 1e6 events (ms) | 71.2 | 43.6 | 43.6 |
| n_syst=5: fwd (ms) | 10.7 | 6.1 | 8.9 |
| n_syst=5: grad (ms) | 60.2 | 39.5 | 59.1 |
| n_syst=20: fwd (ms) | 15.8 | 11.6 | 13.0 |
| n_syst=20: grad (ms) | 58.9 | 47.0 | 48.2 |
| n_syst=50: fwd (ms) | 11.5 | 12.4 | 13.1 |
| n_syst=50: grad (ms) | 86.3 | 127.0 | 60.9 |
| dedup build, 1 syst (ms) | 65 | 142 | 156 |
| no-dedup build, 1 syst (s) | 33.6 | 74.1 | 69.4 |
| dedup eval, 1 syst (ms) | 0.222 | 0.458 | 0.269 |
| no-dedup eval, 1 syst (ms) | 0.255 | 1.982 | 0.436 |
| no-dedup / dedup eval ratio | 1.15 | 4.32 | 1.62 |
| NUTS bounded probe (steps/s) | 3.98 | 2.48 | 2.48 |
| RWM (steps/s, 1e5 events) | 97 | 103 | 103 |
| RWM acceptance (Laplace-covariance proposal) | 0.12 | 0.63 | 0.63 |
| vmapped RWM, 1 chain(s): total steps/s | 87 | 137 | 137 |
| vmapped RWM, 4 chain(s): total steps/s | 96 | 106 | 106 |
| vmapped RWM, 8 chain(s): total steps/s | 146 | 151 | 151 |

Reading it: aggregate costs at large n_events (fwd/grad at 1e5-1e6 events, component split) agree between machines within ~30-40%, and the component ordering at 1e6 events (spline product slightly above oscillation) is stable in both; small-N or fine-grained quantities are not stable (e.g. the no-dedup/dedup eval ratio is 1.15x on Lyon but 4.3x and 1.6x on the loaded box, and the n_syst scaling in item 3 is not monotone within noise in any of the three runs, so the 'ms/systematic' slopes printed in section 3 are not a scaling measurement). The vmapped-RWM chain scan (item 6) shows total throughput that is flat-to-slowly-rising with chain count on CPU (per-chain throughput falling ~1/n_chains) in all runs. The RWM acceptance in item 5 (0.12 vs 0.63) depends on the 150-step L-BFGS MAP and Laplace covariance it is seeded from and is not a stable property of this benchmark.
