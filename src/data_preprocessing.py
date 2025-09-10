# src/data_preprocessing.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

NUM_SUFFIXES = ("_bytes", "_pkts", "_len", "_dur", "_count", "_rate")

def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)

def basic_feature_split(df: pd.DataFrame, target: Optional[str]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    if target is not None and target in df.columns:
        y = df[target].astype(int)
        X = df.drop(columns=[target])
    else:
        y = None
        X = df.copy()
    return X, y

def numeric_categorical_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    return df[numeric_cols], df[cat_cols]

def preprocess(
    in_path: str | Path,
    out_dir: str | Path,
    target: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = load_csv(in_path)
    X, y = basic_feature_split(df, target)
    X_num, X_cat = numeric_categorical_split(X)

    # Minimal categorical handling (drop for now to keep pipeline simple).
    if not X_cat.empty:
        # TODO: one-hot encode categories when needed
        X_num = pd.concat([X_num], axis=1)

    # Impute + scale numerics
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_num_imp = imp.fit_transform(X_num)
    X_num_scaled = sc.fit_transform(X_num_imp)
    X_proc = pd.DataFrame(X_num_scaled, columns=X_num.columns)

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_proc, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_proc, np.zeros(len(X_proc)), test_size=test_size, random_state=random_state
        )
        y_train = pd.Series(y_train, name="label"); y_test = pd.Series(y_test, name="label")

    X_train.to_parquet(out / "X_train.parquet")
    X_test.to_parquet(out / "X_test.parquet")
    y_train.to_frame().to_parquet(out / "y_train.parquet")
    y_test.to_frame().to_parquet(out / "y_test.parquet")

    meta = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "target_present": y is not None
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="CSV path with data")
    ap.add_argument("--target", dest="target", default=None, help="Target column name (optional)")
    ap.add_argument("--out", dest="out_dir", default="data/processed", help="Output directory")
    args = ap.parse_args()
    meta = preprocess(args.in_path, args.out_dir, target=args.target)
    print("Wrote processed data. Meta:", meta)
