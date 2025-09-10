# src/model_training.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.validation import check_is_fitted

def _load_split(proc_dir: str | Path):
    d = Path(proc_dir)
    X_train = pd.read_parquet(d / "X_train.parquet")
    X_test  = pd.read_parquet(d / "X_test.parquet")
    y_train = pd.read_parquet(d / "y_train.parquet")["label"].astype(int)
    y_test  = pd.read_parquet(d / "y_test.parquet")["label"].astype(int)
    return X_train, X_test, y_train, y_test

def train_isolation_forest(proc_dir: str | Path, contamination: float = 0.05):
    X_train, X_test, y_train, y_test = _load_split(proc_dir)
    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    clf.fit(X_train)
    # Scores: anomalies ≈ negative score; convert to label via threshold (0 default in sklearn decision_function)
    scores = -clf.decision_function(X_test)
    preds = (scores > np.percentile(scores, 100*(1-contamination))).astype(int)
    return _evaluate("IsolationForest", y_test, preds, scores, out_dir="results/output"), clf

def train_oneclass_svm(proc_dir: str | Path, nu: float = 0.05, gamma: str | float = "scale"):
    X_train, X_test, y_train, y_test = _load_split(proc_dir)
    clf = OneClassSVM(nu=nu, gamma=gamma)
    clf.fit(X_train)
    scores = -clf.decision_function(X_test)
    preds = (scores > np.percentile(scores, 95)).astype(int)
    return _evaluate("OneClassSVM", y_test, preds, scores, out_dir="results/output"), clf

def train_random_forest(proc_dir: str | Path):
    X_train, X_test, y_train, y_test = _load_split(proc_dir)
    clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced")
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:,1]
    preds = (proba >= 0.5).astype(int)
    return _evaluate("RandomForest", y_test, preds, proba, out_dir="results/output"), clf

def _evaluate(name, y_true, y_pred, scores, out_dir="results/output"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    try:
        roc = roc_auc_score(y_true, scores)
    except Exception:
        roc = None
    metrics = {"model": name, "roc_auc": roc, **report["macro avg"], "f1": report["weighted avg"]["f1-score"]}
    (out / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", dest="proc", default="data/processed")
    ap.add_argument("--model", choices=["iforest", "ocsvm", "rf"], default="iforest")
    args = ap.parse_args()
    if args.model == "iforest":
        m, _ = train_isolation_forest(args.proc)
    elif args.model == "ocsvm":
        m, _ = train_oneclass_svm(args.proc)
    else:
        m, _ = train_random_forest(args.proc)
    print(json.dumps(m, indent=2))
