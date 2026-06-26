# catstat CPU/GPU parity + crossover (Colab)
- python: 3.12.13

## Parity (n=200k, 5k categories)
| case | t_allclose | t_maxabs | ft_allclose | ft_maxabs | cpu_ft_s | gpu_ft_s | status |
|------|-----------|----------|-------------|-----------|----------|----------|--------|
| regression_mean | True | 3.3306690738754696e-16 | True | 0.0 | 0.156 | 0.2071 | ok |
| regression_var | True | 1.3322676295501878e-15 | True | 1.5543122344752192e-15 | 0.4408 | 0.5033 | ok |
| binary_mean | True | 0.0 | True | 0.0 | 0.1786 | 0.2275 | ok |
| multiclass_mean | True | 0.0 | True | 0.0 | 0.4423 | 0.6541 | ok |
| regression_mean_missing | True | 2.220446049250313e-16 | True | 0.0 | 0.1691 | 0.2211 | ok |
| numeric_auto | True | 1.1275702593849246e-17 | True | 0.0 | 0.6253 | 0.6774 | ok |
| numeric_bin | True | 9.540979117872439e-18 | True | 0.0 | 0.6356 | 0.6577 | ok |

## Crossover (mean encoder; fit_transform median seconds)
| n | cardinality | cpu_ft_s | gpu_ft_s | speedup (cpu/gpu) |
|---|-------------|----------|----------|-------------------|
| 10000 | 250 | 0.01 | 0.049 | 0.2 |
| 100000 | 2500 | 0.0668 | 0.1083 | 0.62 |
| 1000000 | 25000 | 0.8721 | 1.3093 | 0.67 |
| 5000000 | 125000 | 5.4259 | 4.8961 | 1.11 |
| 10000000 | 250000 | 12.4614 | 11.7917 | 1.06 |
