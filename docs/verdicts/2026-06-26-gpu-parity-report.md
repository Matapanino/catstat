# catstat CPU/GPU parity (Colab)
- python: 3.12.13

| case | t_allclose | t_maxabs | ft_allclose | ft_maxabs | gpu_ft_s | status |
|------|-----------|----------|-------------|-----------|----------|--------|
| regression_mean | True | 2.220446049250313e-15 | True | 2.6645352591003757e-15 | 1.36 | ok |
| regression_var | True | 6.8833827526759706e-15 | True | 1.071365218763276e-14 | 0.1387 | ok |
| binary_mean | True | 0.0 | True | 0.0 | 0.115 | ok |
| multiclass_mean | True | 0.0 | True | 0.0 | 0.3455 | ok |
