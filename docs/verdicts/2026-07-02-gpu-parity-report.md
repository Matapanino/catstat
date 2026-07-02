> **Post-run note (same day):** the single MISMATCH below (`shape_offset_1e9`, transform side)
> was diagnosed as unshifted fit-path reductions cancelling differently on cuDF vs pandas at
> `|mean| >> sd`; fixed in commit `2872a76` (shift-stable fit reductions). The fix is
> T4-validated by the full suite (360 passed) incl. the dedicated
> `test_cpu_gpu_parity_large_offset` covering exactly this case. A parity-table regeneration
> was blocked by a Colab session-assignment quota; the crossover numbers are unaffected
> (the fix is numerically, not performance, relevant).

# catstat CPU/GPU parity + crossover (Colab)
- python: 3.12.13

## Parity (n=200k, 5k categories)
| case | t_allclose | t_maxabs | ft_allclose | ft_maxabs | cpu_ft_s | gpu_ft_s | status |
|------|-----------|----------|-------------|-----------|----------|----------|--------|
| regression_mean | True | 3.3306690738754696e-16 | True | 2.220446049250313e-16 | 0.1138 | 0.1549 | ok |
| regression_var | True | 1.3322676295501878e-15 | True | 8.881784197001252e-16 | 0.1217 | 0.1349 | ok |
| binary_mean | True | 0.0 | True | 0.0 | 0.1196 | 0.1866 | ok |
| multiclass_mean | True | 0.0 | True | 0.0 | 0.2068 | 0.3972 | ok |
| regression_mean_missing | True | 3.3306690738754696e-16 | True | 2.220446049250313e-16 | 0.1208 | 0.1672 | ok |
| numeric_auto | True | 8.673617379884035e-18 | True | 3.8163916471489756e-17 | 0.9056 | 0.6006 | ok |
| numeric_bin | True | 1.0408340855860843e-17 | True | 4.163336342344337e-17 | 0.5224 | 0.557 | ok |
| combination_mean | True | 1.1102230246251565e-15 | True | 4.884981308350689e-15 | 0.3141 | 0.3406 | ok |
| combination_var | True | 3.774758283725532e-15 | True | 1.7763568394002505e-15 | 0.446 | 0.5342 | ok |
| combination_mean_missing | True | 1.1102230246251565e-15 | True | 4.884981308350689e-15 | 0.3178 | 0.3528 | ok |
| interactions_mean | True | 1.1102230246251565e-15 | True | 4.884981308350689e-15 | 0.5099 | 0.564 | ok |
| regression_skew | True | 1.9984014443252818e-15 | True | 2.1094237467877974e-15 | 0.1068 | 0.1639 | ok |
| regression_kurt | True | 7.993605777301127e-15 | True | 6.217248937900877e-15 | 0.1076 | 0.1708 | ok |
| shape_offset_1e9 | False | 2049.3973147431007 | True | 6.661338147750939e-15 | 0.339 | 0.4584 | MISMATCH |
| binary_woe_auto | True | 0.0 | True | 0.0 | 0.1411 | 0.2471 | ok |
| binary_woe_m20 | True | 0.0 | True | 0.0 | 0.1128 | 0.1648 | ok |
| regression_median | True | 0.0 | True | 0.0 | 0.358 | 0.4403 | ok |

## Crossover (fit_transform median seconds; lanes: cpu / gpu host-origin / gpu device-resident)
| n | cardinality | profile | reps | cpu | gpu-host | gpu-dev | speedup host | speedup dev |
|---|---|---|---|---|---|---|---|---|
| 10000 | 250 | mean | 3 | 0.009 | 0.0385 | 0.0099 | 0.23 | 0.91 |
| 10000 | 250 | mvsk | 3 | 0.0197 | 0.131 | 0.0161 | 0.15 | 1.22 |
| 100000 | 2500 | mean | 3 | 0.053 | 0.096 | 0.0207 | 0.55 | 2.56 |
| 100000 | 2500 | mvsk | 3 | 0.1035 | 0.2414 | 0.0279 | 0.43 | 3.71 |
| 1000000 | 25000 | mean | 5 | 0.8447 | 0.7741 | 0.1447 | 1.09 | 5.84 |
| 1000000 | 25000 | mvsk | 5 | 1.3955 | 1.5517 | 0.1658 | 0.9 | 8.42 |
| 5000000 | 125000 | mean | 5 | 4.5367 | 4.0063 | 0.7707 | 1.13 | 5.89 |
| 5000000 | 125000 | mvsk | 5 | 9.3049 | 8.1463 | 1.0193 | 1.14 | 9.13 |
| 10000000 | 250000 | mean | 5 | 10.1257 | 9.2298 | 1.5903 | 1.1 | 6.37 |
| 10000000 | 250000 | mvsk | 5 | 20.6896 | 17.4168 | 1.9204 | 1.19 | 10.77 |
| 1000000 | 25000 | median | 3 | 2.52 | 2.9062 | 0.2502 | 0.87 | 10.07 |
| 5000000 | 125000 | median | 3 | 16.3955 | 15.9759 | 1.3185 | 1.03 | 12.43 |

## Transform-only (fitted encoder; median seconds)
| n | cpu | gpu-dev (cuDF in/out) | speedup dev |
|---|---|---|---|
| 1000000 | 0.1822 | 0.0276 | 6.6 |
| 10000000 | 2.2831 | 0.1731 | 13.19 |
