# catstat CPU/GPU parity + crossover (Colab)
- python: 3.12.13

## Parity (n=200k, 5k categories)
| case | t_allclose | t_maxabs | ft_allclose | ft_maxabs | cpu_ft_s | gpu_ft_s | status |
|------|-----------|----------|-------------|-----------|----------|----------|--------|
| regression_mean | True | 3.3306690738754696e-16 | True | 0.0 | 0.1915 | 0.1998 | ok |
| regression_var | True | 1.3322676295501878e-15 | True | 0.0 | 0.1527 | 0.1719 | ok |
| binary_mean | True | 0.0 | True | 0.0 | 0.164 | 0.2105 | ok |
| multiclass_mean | True | 0.0 | True | 0.0 | 0.309 | 0.513 | ok |
| regression_mean_missing | True | 2.220446049250313e-16 | True | 0.0 | 0.1615 | 0.2015 | ok |
| numeric_auto | True | 1.0408340855860843e-17 | True | 0.0 | 0.8187 | 0.6377 | ok |
| numeric_bin | True | 8.673617379884035e-18 | True | 0.0 | 0.6086 | 0.655 | ok |
| combination_mean | True | 1.1102230246251565e-15 | True | 0.0 | 0.3679 | 0.5603 | ok |
| combination_var | True | 3.774758283725532e-15 | True | 0.0 | 0.3331 | 0.3376 | ok |
| combination_mean_missing | True | 1.1102230246251565e-15 | True | 0.0 | 0.368 | 0.3872 | ok |
| interactions_mean | True | 1.1102230246251565e-15 | True | 0.0 | 0.577 | 0.8054 | ok |

## Crossover (mean encoder; fit_transform median seconds)
| n | cardinality | cpu_ft_s | gpu_ft_s | speedup (cpu/gpu) |
|---|-------------|----------|----------|-------------------|
| 10000 | 250 | 0.0148 | 0.0567 | 0.26 |
| 100000 | 2500 | 0.1045 | 0.173 | 0.6 |
| 1000000 | 25000 | 0.8383 | 0.9039 | 0.93 |
| 5000000 | 125000 | 5.7528 | 4.7329 | 1.22 |
| 10000000 | 250000 | 11.2374 | 10.5415 | 1.07 |
