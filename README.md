# AI-Generated Text Classification (Kaggle Competition)

This repository contains the end-to-end machine learning pipeline for the AI-generated text classification competition. The project is designed with a lightweight, reproducible environment in mind, optimized for token-efficient agentic editing and structured experiment reporting.

## 📁 Directory Structure

```
.
├── CLAUDE.md                # Project-specific commands, navigation rules & token-saving guidelines
├── README.md                # Project README file
├── data/                    # [Git Ignored] Raw input datasets
│   ├── train.csv            # Labeled training set (10,000 samples)
│   ├── unlabeled.csv        # Semi-supervised dataset (3,000 samples)
│   ├── test.csv             # Kaggle test set (3,000 samples)
│   └── sample_submission.csv
├── docs/                    # Competition specs and data descriptions
│   ├── overview.md          # Competition objective, requirements, and metric definitions
│   └── data_description.md  # Data dictionary and column definitions
├── output/                  # [Git Ignored] Submission files and performance logs
│   ├── submission.csv       # Final Kaggle-ready prediction probabilities
│   └── model_performance.md # Automated validation AUROC & experiment log
└── src/                     # Source code directory
    ├── eda.py               # Exploratory Data Analysis script
    └── train.py             # End-to-end ML pipeline script

```

## 🛠️ Environment Setup

### Prerequisites

* WSL / Ubuntu Environment

* Python 3.10+

* Virtual environment (`.venv`)

### Installation & Virtual Environment Activation

```
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install pandas numpy scikit-learn

```

## 🚀 Execution & Workflow

### 1. Exploratory Data Analysis (EDA)

Run the EDA script to inspect dataset statistics, label distributions, and missing values:

```
python3 src/eda.py

```

### 2. Running the Baseline Model

Execute the full training, validation, test inference, and report generation pipeline with a single command:

```
python3 src/train.py

```

### 3. Expected Outputs

Upon successful execution, the pipeline will automatically generate/update the following files:

1. `output/submission.csv`: Predictions for `test.csv` in Kaggle submission format (containing `id` and predicted probability `label`).

2. `output/model_performance.md`: Summary report documenting validation AUROC score, hyperparameters (`ngram_range`, `max_features`), and runtime metadata.

## ⚙️ Model Architecture Overview

* **Preprocessing & Feature Extraction**: `TfidfVectorizer` (Word-level unigrams and bigrams, `ngram_range=(1, 2)`, `max_features=100000`).

* **Classifier**: `LogisticRegression` using `predict_proba()` for probability prediction.

* **Validation Protocol**: Stratified train/validation split (`random_state=42`).

* **Evaluation Metric**: Area Under the ROC Curve (AUROC).

## 🤖 AI Assistant & Token Optimization Protocol

This project utilizes a structured `CLAUDE.md` file to guide AI development assistants (e.g., Claude Code CLI using `claude-3-5-haiku-20241022`).

* **No Raw Data Leakage**: AI models must not read raw CSV data files directly into the context window.

* **Spec-Driven Development**: All architectural details and output specifications are sourced directly from `docs/overview.md`.

* **Automated Logging**: Training metrics are written directly to `output/model_performance.md` rather than dumped into terminal output.

## 📦 Version Control & Security

* **Large Datasets**: Data files inside `data/` and prediction logs in `output/` are explicitly excluded via `.gitignore`.

* **Git Push Authentication**: Authenticate using standard GitHub Personal Access Tokens (PAT).