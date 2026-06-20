# ◈ SmartModel AI

**Your Data. Your Model.**

Enterprise AutoML platform that works exclusively with datasets you upload. No external data sources. No Kaggle. No internet fetching.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features

### Workflow
1. **Upload Dataset** — CSV or Excel files only. Your data stays in this environment.
2. **Data Analysis** — Automatic profiling: types, missing values, distributions, correlations.
3. **Data Cleaning** — Impute, encode, scale, and remove outliers.
4. **Feature Selection** — Choose which columns to train on.
5. **AutoML Training** — Up to 10 algorithms ranked by performance.
6. **Evaluate Results** — Leaderboard, confusion matrices, ROC curves, feature importances.
7. **Make Predictions** — Upload new data and download results as CSV.
8. **Export & Report** — Download trained model (.pkl/.joblib) and PDF report.

### ML Models
**Classification:** Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, KNN, SVM, XGBoost*, LightGBM*, CatBoost*

**Regression:** Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, SVR, XGBoost*, LightGBM*, CatBoost*

*Optional — install separately if needed.

### Navigation
- **Dashboard** — Overview, workflow guide, training history
- **Upload Dataset** — Primary entry point; CSV and Excel support
- **Train Model** — Data analysis, cleaning, and AutoML in one place
- **Predictions** — Leaderboard, visualizations, prediction generator, model export
- **Reports** — PDF report generation and training history
- **Settings** — Theme (dark/light), session management, data policy

## Security & Privacy

- Works **only** with user-uploaded datasets
- No Kaggle integration
- No external API calls
- No public dataset suggestions
- No URL import
- No web scraping
- All training uses your data exclusively

## Optional Libraries

```bash
pip install xgboost lightgbm catboost reportlab scipy
```

---

*SmartModel AI — Your Data. Your Model.*
