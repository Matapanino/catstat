# catstat CPU/GPU parity + crossover (Colab)
- python: 3.12.13

## Parity (n=200k, 5k categories)
| case | t_allclose | t_maxabs | ft_allclose | ft_maxabs | cpu_ft_s | gpu_ft_s | status |
|------|-----------|----------|-------------|-----------|----------|----------|--------|
| regression_mean | True | 2.7755575615628914e-16 | True | 3.3306690738754696e-16 | 0.3211 | 0.6156 | ok |
| regression_var | True | 1.3322676295501878e-15 | True | 1.5543122344752192e-15 | 0.4935 | 0.8462 | ok |
| binary_mean | True | 0.0 | True | 0.0 | 0.3632 | 0.6299 | ok |
| multiclass_mean | True | 0.0 | True | 0.0 | 1.3343 | 2.0067 | ok |
| regression_mean_missing | True | 2.220446049250313e-16 | True | 3.3306690738754696e-16 | 0.3656 | 0.6632 | ok |

## Crossover (mean encoder; fit_transform median seconds)
| n | cardinality | cpu_ft_s | gpu_ft_s | speedup (cpu/gpu) |
|---|-------------|----------|----------|-------------------|
| 10000 | 250 | 0.0319 | 0.1156 | 0.28 |
| 100000 | 2500 | 0.1589 | 0.5872 | 0.27 |
| 1000000 | 25000 | 1.8829 | 2.1922 | 0.86 |
