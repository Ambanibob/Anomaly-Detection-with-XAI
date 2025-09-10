# scripts/train_baselines.py
import argparse, json
from src.model_training import (
    train_isolation_forest, train_oneclass_svm, train_random_forest
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--model", choices=["iforest","ocsvm","rf"], default="iforest")
    args = ap.parse_args()

    if args.model == "iforest":
        m, _ = train_isolation_forest(args.proc)
    elif args.model == "ocsvm":
        m, _ = train_oneclass_svm(args.proc)
    else:
        m, _ = train_random_forest(args.proc)

    print(json.dumps(m, indent=2))
