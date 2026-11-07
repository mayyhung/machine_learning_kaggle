import os
import pandas as pd
import numpy as np

DATA_DIR = "./data"
DOCS_DIR = "./docs"
OUTPUT_MD = os.path.join(DOCS_DIR, "data_description.md")

def generate_data_description():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} does not exist. Please check your data path.")
        return

    os.makedirs(DOCS_DIR, exist_ok=True)
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    print(f"📂 Found CSV files in {DATA_DIR}: {csv_files}\n")

    markdown_content = "# Kaggle Competition Data Description\n\n"
    markdown_content += "This document provides automated statistical summaries and schema details for the dataset.\n\n"

    for file in csv_files:
        file_path = os.path.join(DATA_DIR, file)
        print(f"🔍 Analyzing file: {file}...")
        
        try:
            df = pd.read_csv(file_path, nrows=5000)
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue

        markdown_content += f"## File: `{file}`\n"
        markdown_content += f"- **Shape (Rows, Columns):** `{df.shape}`\n"
        markdown_content += f"- **Estimated Memory Usage:** `{df.memory_usage().sum() / 1024**2:.2f} MB` (Sampled from first 5000 rows)\n\n"

        markdown_content += "### Column Summary\n"
        markdown_content += "| Column Name | Data Type | Unique Values | Missing Count | Missing % |\n"
        markdown_content += "| :--- | :--- | :--- | :--- | :--- |\n"
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            missing_count = int(df[col].isnull().sum())
            missing_pct = (missing_count / len(df)) * 100
            markdown_content += f"| `{col}` | {dtype} | {unique_count} | {missing_count} | {missing_pct:.2f}% |\n"

        markdown_content += "\n### Numerical Statistics Summary\n"
        desc = df.describe().T
        if not desc.empty:
            markdown_content += desc.to_markdown() + "\n\n"
        else:
            markdown_content += "No numerical columns available.\n\n"
        
        markdown_content += "---\n\n"

    # Write to docs/data_description.md
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"\n✨ Successfully generated data description at: {OUTPUT_MD}")

if __name__ == "__main__":
    generate_data_description()