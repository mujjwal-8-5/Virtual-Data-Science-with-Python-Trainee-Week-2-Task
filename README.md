# Week 2: Exploratory Data Analysis and Visualization

## Overview
A reproducible EDA project using Python, Pandas, Matplotlib, and Seaborn on the public Wine Recognition dataset.

## Dataset
- Original source: UCI Machine Learning Repository — Wine recognition data.
- The CSV is a local export of scikit-learn's packaged `load_wine()` dataset, so the project runs offline.
- Records: 178; exported columns: 15; chemical features: 13.
- Target: `target` (integer class code) and `target_name` (readable class label).

## Folder contents
- `data/wine.csv` — dataset used by the script.
- `src/eda_analysis.py` — complete reproducible analysis and chart-generation script.
- `outputs/figures/` — six annotated PNG charts.
- `outputs/summary_tables/` — descriptive statistics, class means, correlation matrix, missingness, and IQR outlier counts.
- `REPORT_Week2_EDA.docx` — final report.
- `requirements.txt` — dependencies.

## Run the project
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python src/eda_analysis.py
```

## Main findings
- Class distribution: class_0: 59 samples (33.1%); class_1: 71 samples (39.9%); class_2: 48 samples (27.0%).
- Missing cells: 0; duplicate rows: 0.
- Mean alcohol by class: class_0: 13.74, class_1: 12.28, class_2: 13.15.
- Strongest positive feature correlation: total_phenols and flavanoids (r=0.86).
- Strongest negative feature correlation: malic_acid and hue (r=-0.56).
- IQR outliers are review flags only; no rows are automatically removed.

## Interpretation cautions
This is a small observational dataset. Correlation does not imply causation, and findings should not automatically generalize to all wines. This is exploratory analysis, not a trained prediction model.

## References
- UCI Wine recognition data: https://archive.ics.uci.edu/dataset/109/wine
- scikit-learn `load_wine`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
- Pandas: https://pandas.pydata.org/docs/
- Matplotlib: https://matplotlib.org/stable/
- Seaborn: https://seaborn.pydata.org/
