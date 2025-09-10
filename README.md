# Advanced Network Anomaly Detection with Ensemble ML and XAI

> **Status:** Initial scaffold — ready for data + experiments.

## 1) What Problem Does This Solve?
Detect malicious or anomalous network behavior (intrusions, C2 traffic, scans) using classical ML and simple deep-learning baselines, with **explainability** baked in so defenders know *why* an alert fired.

## 2) Data Sources
- **Recommended:** CICIDS2017 (Canadian Institute for Cybersecurity). Requires registration to download. Place CSV/Parquet files under `data/raw/`.
- **Alternative (quick start):** NSL-KDD or UNSW-NB15. Convert to CSV and drop under `data/raw/`.
- You can also use your own NetFlow/PCAP-derived features.

> **Note:** This repo includes a small **synthetic generator** so you can validate the pipeline without a big dataset.

## 3) Tech Stack
- Python 3.10+
- NumPy, Pandas, Scikit-learn
- (Optional) PyTorch for autoencoder baseline
- SHAP / LIME for XAI
- Matplotlib for plots
- Jupyter for exploration

## 4) How This Repo Is Organized
```
/project-root
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/          # put source CSV/Parquet here
│   └── processed/    # cached cleaned/encoded data
├── notebooks/        # Jupyter notebooks for EDA and experiments
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── explainability.py
│   └── synth_data.py
├── results/
│   ├── figures/
│   └── output/
└── scripts/
    ├── prepare_dataset.py
    └── train_baselines.py
```

## 5) Quickstart
1. **Create & activate environment**  
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -U pip
   pip install -r requirements.txt
   ```

2. **Add data** under `data/raw/`. CSV with headers recommended. If you don't have data yet, generate synthetic data:
   ```bash
   python -m src.synth_data --n 50000 --attack-rate 0.03 --out data/raw/synth.csv
   ```

3. **Prepare features** (imputing, scaling, encoding, splits) to `data/processed/`:
   ```bash
   python scripts/prepare_dataset.py --in data/raw/synth.csv --target label
   ```

4. **Train baselines** (IsolationForest, OneClassSVM, RandomForest, optional Autoencoder):
   ```bash
   python scripts/train_baselines.py --target label
   ```

5. **Evaluate + Explain** (confusion matrix, ROC, SHAP/LIME):
   - Metrics and plots land in `results/`.
   - SHAP values saved under `results/output/` when feasible.

## 6) Modeling Plan
- **Classical:** IsolationForest (unsupervised), OneClassSVM (unsupervised), RandomForest/XGBoost (supervised if labels exist)
- **Neural:** Simple Autoencoder for reconstruction error (optional)
- **Tuning:** Grid/random search with stratified splits
- **Metrics:** Precision/Recall/F1/ROC-AUC; PR-AUC emphasized for skewed classes

## 7) Explainability (XAI)
- **SHAP** for tree/linear models (feature importance and local explanations)
- **LIME** for sanity checks on individual samples
- **Goal:** Make anomalies explainable enough to action (e.g., “unusual dst_port + high bytes_sent + rare JA3 hash”)

## 8) How to Reproduce
- Pin your Python version in `requirements.txt` or use `environment.yml`.
- `make` targets or simple Python scripts are provided for end-to-end runs.
- Keep **data paths** relative; commit no raw data.

## 9) Limitations
- Public datasets ≠ your enterprise reality. Treat this as a pattern, not a product.
- CICIDS2017/UNSW-NB15 have quirks and class imbalance.
- SHAP can be slow on very large datasets/models; sample intelligently.

## 10) Roadmap / Next Steps
- Add ELK/Splunk export of high-scoring anomalies
- Convert top SHAP features into **Sigma** rule suggestions (PoC in a separate folder)
- Try LightGBM/XGBoost + calibrated thresholds
- Add drift detection for production-ish simulation

## 11) License & Credits
- MIT License. See `LICENSE`.
- Datasets credited to their original authors.

---

**TL;DR**: Real ML, real XAI, real repo hygiene. Not academic fluff.
