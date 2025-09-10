# scripts/prepare_dataset.py
import argparse
from src.data_preprocessing import preprocess

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="CSV path in data/raw/")
    ap.add_argument("--target", default="label", help="Target column (if present)")
    ap.add_argument("--out", default="data/processed", help="Output dir for processed parquet files")
    args = ap.parse_args()
    meta = preprocess(args.in_path, args.out, target=args.target)
    print("OK", meta)
