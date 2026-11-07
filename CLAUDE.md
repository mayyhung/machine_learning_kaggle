# Kaggle Competition Project Guide

## Project Overview
- **Goal**: Machine Learning prediction task. Detailed project goals and evaluation metrics are defined in `docs/overview.md`.
- **Data Source**: Raw CSV files located in `./data/` (ignored by Git).
- **Data Description**: Schema and statistical summaries are available in `docs/data_description.md`.

## Directory Structure
- `data/`: Contains raw input CSV data (`train.csv`, `test.csv`).
- `docs/`: Contains project documentation (`overview.md`, `data_description.md`).
- `output/`: Contains final output predictions (`submission.csv`) and performance reports (`model_performance.md`).
- `src/`: Contains source code (`train.py`).

## Quick Commands
- **Run Training**: `python3 src/train.py`

## Token & Traffic Saving Rules 
- **Never read raw data files**: Do NOT read large CSV files directly into the context window.
- **Rely on summaries**: Use `docs/overview.md` for project goals and `docs/data_description.md` for column schemas.
- **Save evaluation results to file**: Ensure `src/train.py` automatically writes validation metrics (AUROC, parameters, dataset sizes) to `output/model_performance.md` instead of printing raw logs in the chat.
- **Targeted outputs**: Output final predictions to `output/submission.csv`.