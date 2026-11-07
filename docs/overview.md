# AI-Generated Text Classification

## 1. Project Objective

Develop a binary text classification model to predict the probability that a given passage is AI-generated.

**Required approach: TF-IDF + Logistic Regression**

The goal is to build one complete, reproducible machine learning pipeline and generate a Kaggle-ready submission.

## 2. Dataset Description

The following CSV files are provided:

| File | Examples | Columns | Description |
|---|---:|---|---|
| `train.csv` | 10,000 | `id`, `text`, `label` | Labeled training data |
| `unlabeled.csv` | 3,000 | `id`, `text` | Additional unlabeled data |
| `test.csv` | 3,000 | `id`, `text` | Test data for Kaggle predictions |
| `train_starter_2000.csv` | 2,000 | `id`, `text`, `label` | Optional starter subset |
| `sample_submission.csv` | 3,000 | `id`, `label` | Submission format reference |

### Label Definitions

| Label | Meaning |
|---|---|
| `0` | Human-written text |
| `1` | AI-generated text |

The training dataset contains 5,000 human-written and 5,000 AI-generated examples.

### Column Definitions

- `id`: Unique identifier for each text passage.
- `text`: Text passage to be classified.
- `label`: Binary target variable.

## 3. Modeling Requirements

Implement a complete classification pipeline using the following approach:

### Text Preprocessing and Feature Extraction

- Use `TfidfVectorizer` from scikit-learn.
- Use word-level unigrams and bigrams (`ngram_range=(1, 2)`).
- Set `max_features=100000`.
- Handle missing or null text values appropriately.
- Fit the vectorizer on the training data only to avoid data leakage.

### Classification

- Use `LogisticRegression` from scikit-learn.
- Use `predict_proba()` to generate predicted probabilities.
- Set a fixed `random_state` for reproducibility.
- Use default model settings unless a specific adjustment is necessary.

### Validation

- Split the labeled training data into training and validation sets.
- Use a stratified split to preserve class proportions.
- Set `random_state=42`.
- Evaluate model performance using AUROC.

### Final Training and Prediction

- After validation, retrain the TF-IDF vectorizer and Logistic Regression model on the full labeled training dataset.
- Generate predicted probabilities for all examples in `test.csv`.
- Use the probability of class `1` (AI-generated) as the final prediction.

## 4. Evaluation Metric

The competition uses **Area Under the Receiver Operating Characteristic Curve (AUROC)**.

- `0.50`: Random ranking performance.
- `1.00`: Perfect ranking performance.
- Higher AUROC indicates better model performance.

Use `roc_auc_score` from scikit-learn to calculate validation AUROC.

## 5. Final Deliverables

Generate the following two files:

### 1. `output/submission.csv`

The Kaggle-ready prediction file.

| Column | Description |
|---|---|
| `id` | Original test data identifier |
| `label` | Predicted probability of being AI-generated |

Requirements:
- Preserve the original `id` values and their order from `test.csv`.
- The `label` column must contain predicted probabilities between 0 and 1.
- Follow the format of `sample_submission.csv`.

### 2. `src/train.py`

A complete, executable Python script containing the entire machine learning pipeline, including:

- Data loading
- Text preprocessing
- TF-IDF feature extraction
- Logistic Regression model training
- Validation and AUROC evaluation
- Final model training on the full training dataset
- Test set prediction
- `submission.csv` generation

The script should be reproducible and executable without manual intervention.

### 3. `output/model_performance.md`

A Markdown report summarizing the validation performance and pipeline configurations:

Requirements:
- Validation AUROC score (e.g., calculated via stratified train/val split).
- Key pipeline parameters (e.g., TF-IDF `ngram_range`, `max_features`, Logistic Regression settings).
- Dataset summary statistics used for training and validation.
- Execution timestamp and status.

## 6. Implementation Requirements

- Use Python and scikit-learn.
- Use TF-IDF + Logistic Regression as the only modeling approach.
- Avoid unnecessary model comparisons, hyperparameter searches, or alternative algorithms.
- Do not use the unlabeled dataset for training.
- Include clear comments and organize the code into logical sections.
- Ensure all file paths are correctly configured.

**Expected Output:**
1. `train.py`: Complete source code.
2. `submission.csv`: Final Kaggle submission file.
3. `model_performance.md`: Detailed validation metrics and configuration log.

Both files should be saved in the project directory.

