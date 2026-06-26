# catstat CPU/GPU parity + crossover (Colab)
- python: 3.12.13

## Parity (n=200k, 5k categories)
| case | t_allclose | t_maxabs | ft_allclose | ft_maxabs | cpu_ft_s | gpu_ft_s | status |
|------|-----------|----------|-------------|-----------|----------|----------|--------|
| regression_mean | True | 2.220446049250313e-16 | True | 4.440892098500626e-16 | 0.5455 | 0.7374 | ok |
| regression_var | True | 1.1102230246251565e-15 | True | 1.5543122344752192e-15 | 0.3757 | 0.4511 | ok |
| binary_mean | True | 0.0 | True | 0.0 | 0.3504 | 0.644 | ok |
| multiclass_mean | True | 0.0 | True | 0.0 | 0.8118 | 1.967 | ok |
| regression_mean_missing | True | 3.3306690738754696e-16 | True | 3.3306690738754696e-16 | 0.3509 | 0.9054 | ok |
| numeric_auto | True | 8.673617379884035e-18 | True | 1.5612511283791264e-17 | 1.3155 | 1.6771 | ok |
| numeric_bin | True | 9.540979117872439e-18 | True | 1.734723475976807e-17 | 1.352 | 1.4831 | ok |

## Crossover (mean encoder; fit_transform median seconds)
| n | cardinality | cpu_ft_s | gpu_ft_s | speedup (cpu/gpu) |
|---|-------------|----------|----------|-------------------|
| 10000 | 250 | 0.0314 | 0.1557 | 0.2 |
| 100000 | 2500 | 0.2517 | 0.3988 | 0.63 |
| 1000000 | 25000 | 1.909 | 2.2185 | 0.86 |
