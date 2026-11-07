# Model Performance Report

## Execution

- **Execution timestamp:** 2026-11-07 13:15:23
- **Status:** SUCCESS

## Validation Metrics

| Metric | Value |
|---|---:|
| Validation AUROC | 0.9145 |

## Pipeline Configuration

- **Model:** TF-IDF + Logistic Regression
- **Validation split:** `train_test_split`, test_size=0.2, stratify=label, random_state=42

### TfidfVectorizer

| Parameter | Value |
|---|---|
| `ngram_range` | `(1, 2)` |
| `max_features` | `100000` |
| Actual vocabulary size (final model) | 100000 |

### LogisticRegression

| Parameter | Value |
|---|---|
| `C` | `1.0` |
| `class_weight` | `None` |
| `max_iter` | `100` |
| `random_state` | `42` |
| `solver` | `lbfgs` |
| Iterations used / max_iter | 26 / 100 |

## Dataset Summary

| Dataset | Rows | Label 0 (human) | Label 1 (AI) |
|---|---:|---:|---:|
| Labeled train (full) | 10000 | 5000 | 5000 |
| Training split (fit) | 8000 | 4000 | 4000 |
| Validation split | 2000 | 1000 | 1000 |
| Test (predictions) | 3000 | - | - |

## Outputs

- Final model: retrained on the full labeled training set.
- Submission: `output/submission.csv` (3000 rows).
