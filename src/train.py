"""
AI-Generated Text Classification: TF-IDF + Logistic Regression pipeline.

Steps:
  1. Load labeled training data (data/train.csv) and test data (data/test.csv).
  2. Stratified train/validation split -> fit TF-IDF on the training split only
     -> fit Logistic Regression -> evaluate validation AUROC.
  3. Retrain TF-IDF + Logistic Regression on the full labeled training set.
  4. Predict P(label = 1) for every test passage -> output/submission.csv.
  5. Write validation metrics, configuration and dataset statistics to
     output/model_performance.md.

Run from any directory:  python3 src/train.py
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"
REPORT_PATH = OUTPUT_DIR / "model_performance.md"

RANDOM_STATE = 42
VAL_SIZE = 0.2  # Fraction of labeled data held out for validation

TFIDF_PARAMS = {"ngram_range": (1, 2), "max_features": 100000}
LR_PARAMS = {"random_state": RANDOM_STATE}  # Otherwise sklearn defaults


# ---------------------------------------------------------------------------
# 1. Data loading and preprocessing
# ---------------------------------------------------------------------------
def load_data():
    """Load train/test CSVs and replace missing text with empty strings."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # Missing text would break the TF-IDF vectorizer; treat it as an empty passage.
    # (The provided files currently contain no nulls, so this is a safeguard.)
    train["text"] = train["text"].fillna("").astype(str)
    test["text"] = test["text"].fillna("").astype(str)
    return train, test


# ---------------------------------------------------------------------------
# 2. Model construction
# ---------------------------------------------------------------------------
def build_model():
    """Return an unfitted (TfidfVectorizer, LogisticRegression) pair."""
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    classifier = LogisticRegression(**LR_PARAMS)
    return vectorizer, classifier


def fit_and_predict_proba(train_texts, train_labels, eval_texts):
    """Fit TF-IDF on train_texts only, fit LR, and return P(label = 1) for eval_texts."""
    vectorizer, classifier = build_model()

    # Fit the vocabulary/IDF on training data only to avoid leakage.
    X_train = vectorizer.fit_transform(train_texts)
    X_eval = vectorizer.transform(eval_texts)

    classifier.fit(X_train, train_labels)

    # Column index of class 1 (AI-generated) in predict_proba output.
    positive_idx = list(classifier.classes_).index(1)
    proba = classifier.predict_proba(X_eval)[:, positive_idx]
    return proba, vectorizer, classifier


# ---------------------------------------------------------------------------
# 3. Reporting
# ---------------------------------------------------------------------------
def write_report(status, stats, error=None):
    """Write the markdown performance report (called on success and on failure)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def val(key, default="N/A"):
        return stats.get(key, default)

    auroc = stats.get("val_auroc")
    auroc_text = f"{auroc:.4f}" if auroc is not None else "N/A"

    lines = [
        "# Model Performance Report",
        "",
        "## Execution",
        "",
        f"- **Execution timestamp:** {val('timestamp')}",
        f"- **Status:** {status}",
    ]
    if error:
        lines.append(f"- **Error:** `{error}`")

    lines += [
        "",
        "## Validation Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Validation AUROC | {auroc_text} |",
        "",
        "## Pipeline Configuration",
        "",
        "- **Model:** TF-IDF + Logistic Regression",
        f"- **Validation split:** `train_test_split`, test_size={VAL_SIZE}, stratify=label, random_state={RANDOM_STATE}",
        "",
        "### TfidfVectorizer",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]
    for k, v in TFIDF_PARAMS.items():
        lines.append(f"| `{k}` | `{v}` |")
    lines.append(f"| Actual vocabulary size (final model) | {val('vocab_size')} |")

    lines += [
        "",
        "### LogisticRegression",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]
    for k, v in val("lr_params", {}).items():
        lines.append(f"| `{k}` | `{v}` |")
    lines.append(f"| Iterations used / max_iter | {val('lr_n_iter')} / {val('lr_max_iter')} |")

    lines += [
        "",
        "## Dataset Summary",
        "",
        "| Dataset | Rows | Label 0 (human) | Label 1 (AI) |",
        "|---|---:|---:|---:|",
        f"| Labeled train (full) | {val('n_full')} | {val('n_full_0')} | {val('n_full_1')} |",
        f"| Training split (fit) | {val('n_fit')} | {val('n_fit_0')} | {val('n_fit_1')} |",
        f"| Validation split | {val('n_val')} | {val('n_val_0')} | {val('n_val_1')} |",
        f"| Test (predictions) | {val('n_test')} | - | - |",
        "",
        "## Outputs",
        "",
        "- Final model: retrained on the full labeled training set.",
        f"- Submission: `{SUBMISSION_PATH.relative_to(PROJECT_DIR).as_posix()}` ({val('n_test')} rows).",
        "",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 4. Main pipeline
# ---------------------------------------------------------------------------
def main():
    stats = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    try:
        # --- Load data -----------------------------------------------------
        train, test = load_data()
        X_text = train["text"]
        y = train["label"].astype(int)

        stats.update(
            n_full=len(train),
            n_full_0=int((y == 0).sum()),
            n_full_1=int((y == 1).sum()),
            n_test=len(test),
        )

        # --- Stratified validation ----------------------------------------
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_text,
            y,
            test_size=VAL_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        stats.update(
            n_fit=len(y_fit),
            n_fit_0=int((y_fit == 0).sum()),
            n_fit_1=int((y_fit == 1).sum()),
            n_val=len(y_val),
            n_val_0=int((y_val == 0).sum()),
            n_val_1=int((y_val == 1).sum()),
        )

        val_proba, _, _ = fit_and_predict_proba(X_fit, y_fit, X_val)
        val_auroc = roc_auc_score(y_val, val_proba)
        stats["val_auroc"] = float(val_auroc)

        # --- Final retraining on the full labeled set ----------------------
        test_proba, vectorizer, classifier = fit_and_predict_proba(X_text, y, test["text"])
        stats.update(
            vocab_size=len(vectorizer.vocabulary_),
            lr_params={k: v for k, v in classifier.get_params().items()
                       if k in ("C", "solver", "max_iter", "class_weight", "random_state")},
            lr_n_iter=int(classifier.n_iter_[0]),
            lr_max_iter=classifier.max_iter,
        )

        # --- Write submission ---------------------------------------------
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        submission = pd.DataFrame({"id": test["id"], "label": test_proba})
        submission.to_csv(SUBMISSION_PATH, index=False)

        write_report("SUCCESS", stats)

        print(f"Validation AUROC: {val_auroc:.4f}")
        print(f"Submission written to {SUBMISSION_PATH} ({len(submission)} rows)")
        print(f"Report written to {REPORT_PATH}")

    except Exception as exc:
        write_report("FAILED", stats, error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
