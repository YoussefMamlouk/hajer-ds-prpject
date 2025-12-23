## AI Usage Disclosure

This project was implemented with assistance from an AI coding assistant.

### What the AI helped with

- Translating the written project specification into a modular Python package layout (`src/`) and a runnable entrypoint (`main.py`).
- Implementing:
  - Data loading and feature engineering utilities for monthly time-series regression.
  - A rolling-origin backtesting loop (expanding window, no shuffling) and evaluation metrics.
  - Model wrappers for baselines, classical models (ARIMA/SARIMA/SARIMAX), and tree-based ML regressors.
- Drafting repository documentation (`README.md`) and reproducibility configuration (`environment.yml`).

### What was manually specified/verified by the student

- The research question and proposal text (provided verbatim in `PROPOSAL.md`).
- The dataset schema clarification needed to avoid mixing sales with production:
  - In `production_sales.xlsx`, column `H04` identifies whether rows correspond to production vs sales.
  - This implementation filters to `H04 == "Physical volume of mining production"`.
- The requirement that XGBoost/LightGBM/CatBoost are **hard dependencies** (the program fails with a clear message if missing).

### Guardrails / correctness measures

- No random shuffling; all splits are time-ordered.
- Direct multi-horizon setup: for horizon \(h\), features at time \(t\) predict \(y_{t+h}\).
- Feature construction uses only information available at or before time \(t\) (no leakage).

### How to reproduce

- Install dependencies using `environment.yml` (conda):
  ```bash
  conda env create -f environment.yml
  conda activate hajer-ds-project
  ```
- Place raw data files under `data/raw/`.
- Run:

```bash
python main.py
```


