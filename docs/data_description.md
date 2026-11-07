# Kaggle Competition Data Description

This document provides automated statistical summaries and schema details for the dataset.

## File: `sample_submission.csv`
- **Shape (Rows, Columns):** `(3000, 2)`
- **Estimated Memory Usage:** `0.05 MB` (Sampled from first 5000 rows)

### Column Summary
| Column Name | Data Type | Unique Values | Missing Count | Missing % |
| :--- | :--- | :--- | :--- | :--- |
| `id` | int64 | 3000 | 0 | 0.00% |
| `label` | float64 | 1 | 0 | 0.00% |

### Numerical Statistics Summary
|       |   count |    mean |    std |     min |     25% |     50% |     75% |     max |
|:------|--------:|--------:|-------:|--------:|--------:|--------:|--------:|--------:|
| id    |    3000 | 14500.5 | 866.17 | 13001   | 13750.8 | 14500.5 | 15250.2 | 16000   |
| label |    3000 |     0.5 |   0    |     0.5 |     0.5 |     0.5 |     0.5 |     0.5 |

---

## File: `test.csv`
- **Shape (Rows, Columns):** `(3000, 2)`
- **Estimated Memory Usage:** `0.05 MB` (Sampled from first 5000 rows)

### Column Summary
| Column Name | Data Type | Unique Values | Missing Count | Missing % |
| :--- | :--- | :--- | :--- | :--- |
| `id` | int64 | 3000 | 0 | 0.00% |
| `text` | str | 3000 | 0 | 0.00% |

### Numerical Statistics Summary
|    |   count |    mean |    std |   min |     25% |     50% |     75% |   max |
|:---|--------:|--------:|-------:|------:|--------:|--------:|--------:|------:|
| id |    3000 | 14500.5 | 866.17 | 13001 | 13750.8 | 14500.5 | 15250.2 | 16000 |

---

## File: `train.csv`
- **Shape (Rows, Columns):** `(5000, 3)`
- **Estimated Memory Usage:** `0.11 MB` (Sampled from first 5000 rows)

### Column Summary
| Column Name | Data Type | Unique Values | Missing Count | Missing % |
| :--- | :--- | :--- | :--- | :--- |
| `id` | int64 | 5000 | 0 | 0.00% |
| `text` | str | 5000 | 0 | 0.00% |
| `label` | int64 | 2 | 0 | 0.00% |

### Numerical Statistics Summary
|       |   count |      mean |         std |   min |     25% |    50% |     75% |   max |
|:------|--------:|----------:|------------:|------:|--------:|-------:|--------:|------:|
| id    |    5000 | 2500.5    | 1443.52     |     1 | 1250.75 | 2500.5 | 3750.25 |  5000 |
| label |    5000 |    0.5076 |    0.499992 |     0 |    0    |    1   |    1    |     1 |

---

## File: `train_starter_2000.csv`
- **Shape (Rows, Columns):** `(2000, 3)`
- **Estimated Memory Usage:** `0.05 MB` (Sampled from first 5000 rows)

### Column Summary
| Column Name | Data Type | Unique Values | Missing Count | Missing % |
| :--- | :--- | :--- | :--- | :--- |
| `id` | int64 | 2000 | 0 | 0.00% |
| `text` | str | 2000 | 0 | 0.00% |
| `label` | int64 | 2 | 0 | 0.00% |

### Numerical Statistics Summary
|       |   count |    mean |         std |   min |     25% |    50% |     75% |   max |
|:------|--------:|--------:|------------:|------:|--------:|-------:|--------:|------:|
| id    |    2000 | 5018.36 | 2878.66     |     1 | 2517.25 | 4913   | 7500.75 |  9994 |
| label |    2000 |    0.5  |    0.500125 |     0 |    0    |    0.5 |    1    |     1 |

---

## File: `unlabeled.csv`
- **Shape (Rows, Columns):** `(3000, 2)`
- **Estimated Memory Usage:** `0.05 MB` (Sampled from first 5000 rows)

### Column Summary
| Column Name | Data Type | Unique Values | Missing Count | Missing % |
| :--- | :--- | :--- | :--- | :--- |
| `id` | int64 | 3000 | 0 | 0.00% |
| `text` | str | 3000 | 0 | 0.00% |

### Numerical Statistics Summary
|    |   count |    mean |    std |   min |     25% |     50% |     75% |   max |
|:---|--------:|--------:|-------:|------:|--------:|--------:|--------:|------:|
| id |    3000 | 11500.5 | 866.17 | 10001 | 10750.8 | 11500.5 | 12250.2 | 13000 |

---

