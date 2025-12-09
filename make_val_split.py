import argparse, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="data/train_with_paths.csv")
    ap.add_argument("--out-train", default="data/train_dev.csv")
    ap.add_argument("--out-val", default="data/val_dev.csv")
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--bins", type=int, default=25)  # log-price bins
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-col", default="price")
    ap.add_argument("--id-col", default="sample_id")
    ap.add_argument("--image-col", default="image_path")
    ap.add_argument("--title-col", default="catalog_cont")
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv)
    if args.target_col not in df.columns: raise SystemExit("Target column missing.")
    # robust log bins
    y = np.log1p(np.clip(df[args.target_col].values.astype(float), 0, None))
    q = min(args.bins, max(4, int(np.sqrt(len(df)))))
    bins = pd.qcut(y, q=q, labels=False, duplicates="drop")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    tr_idx, vl_idx = next(sss.split(df, bins))
    df.iloc[tr_idx].to_csv(args.out_train, index=False)
    df.iloc[vl_idx].to_csv(args.out_val, index=False)
    print(f"Saved {args.out_train} (n={len(tr_idx)}) and {args.out_val} (n={len(vl_idx)})")

if __name__ == "__main__":
    main()