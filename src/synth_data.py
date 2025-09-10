# src/synth_data.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

def make_synth(n=50000, attack_rate=0.03, seed=1337):
    rng = np.random.default_rng(seed)
    n_attack = int(n * attack_rate)
    n_norm = n - n_attack

    # Normal traffic features (toy)
    src_bytes = rng.gamma(2.0, 200.0, size=n_norm)
    dst_bytes = rng.gamma(2.5, 180.0, size=n_norm)
    duration  = rng.exponential(0.8, size=n_norm)
    pkt_rate  = rng.normal(30, 8, size=n_norm).clip(0, None)

    # Attack traffic skewed
    a_src_bytes = rng.gamma(5.0, 400.0, size=n_attack)
    a_dst_bytes = rng.gamma(4.0, 350.0, size=n_attack)
    a_duration  = rng.exponential(2.5, size=n_attack)
    a_pkt_rate  = rng.normal(120, 30, size=n_attack).clip(0, None)

    df_norm = pd.DataFrame({
        'src_bytes': src_bytes, 'dst_bytes': dst_bytes,
        'duration': duration, 'pkt_rate': pkt_rate, 'label': 0
    })
    df_attk = pd.DataFrame({
        'src_bytes': a_src_bytes, 'dst_bytes': a_dst_bytes,
        'duration': a_duration, 'pkt_rate': a_pkt_rate, 'label': 1
    })
    df = pd.concat([df_norm, df_attk], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50000)
    ap.add_argument('--attack-rate', type=float, default=0.03)
    ap.add_argument('--out', type=str, default='data/raw/synth.csv')
    args = ap.parse_args()
    df = make_synth(n=args.n, attack_rate=args.attack_rate)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote synthetic dataset to {args.out} ({len(df)} rows)")
