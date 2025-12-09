# eda_clean_pipeline.py
# Amazon ML Challenge — Automated EDA + Cleaning at scale (75k+ rows)
# - No seaborn. Pure pandas + matplotlib. No fancy dependencies.
# - Outputs go to data/eda_outputs/ and data/clean/

import os, re, gc, json, math, time, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------- CONFIG -----------------------
ROOT = Path(".")
DATA_DIR = ROOT / "data"
EDA_DIR = DATA_DIR / "eda_outputs"
CLEAN_DIR = DATA_DIR / "clean"
EDA_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV  = DATA_DIR / "test.csv"

# Outlier handling
CLIP_METHOD = "winsorize"   # "winsorize" or "remove"
MAD_K = 5.0                 # smaller => stricter (e.g., 3.5)
MIN_IPQ = 1.0
MIN_PRICE = 0.01

# Rare value collapsing thresholds (proportional to dataset)
BRAND_MIN_FRAC = 0.0005     # ~0.05% of train
CAT_MIN_FRAC   = 0.0003     # ~0.03% of train
ABS_MIN = 5                 # lower bound for tiny datasets

# Category keyword seeds (will expand from data automatically)
CATEGORY_SEEDS = {
    "phone":  ["phone","iphone","android","smartphone","mobile","handset"],
    "laptop": ["laptop","notebook","macbook","chromebook"],
    "audio":  ["earbud","earbuds","earphone","headphone","speaker","soundbar","tws","ear pod","airpod","airpods"],
    "camera": ["camera","dslr","mirrorless","action","gopro","webcam","lens"],
    "watch":  ["watch","smartwatch","band","fitness"],
    "tv":     ["tv","television","oled","qled","uhd","4k"],
    "accessory":["case","cover","charger","cable","adapter","power bank","screen guard","protector","strap","mount"],
    "appliance":["fridge","refrigerator","washing","machine","ac","conditioner","microwave","oven","dishwasher","purifier"],
}

# Brand corrections (expand as you learn)
BRAND_CORRECTIONS = {
    "mi": "xiaomi",
    "one plus": "oneplus",
    "one-plus": "oneplus",
    "samsungtm": "samsung",
    "apple inc": "apple",
    "hp inc": "hp",
    "lg electronics": "lg",
    "sony corp": "sony",
}

# ----------------------- UTILS -----------------------
def ts(): return time.strftime("%H:%M:%S")

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def basic_text_clean(text):
    if not isinstance(text, str): return ""
    t = text.lower()
    t = t.replace("–", "-").replace("—", "-").replace("×", "x")
    t = re.sub(r"[^\w\s\-\.\,&/+]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def extract_ipq(text):
    if not isinstance(text, str) or not text: return 1.0
    t = text.lower()
    score = 1.0
    if re.search(r"\b(premium|genuine|original|pro|ultra|max)\b", t):
        score += 0.3
    m = re.search(r"(\d+(\.\d+)?)\s*(pack|pcs|pieces|units|ml|g|kg|l)\b", t)
    if m:
        val = float(m.group(1))
        score += min(0.5, math.log1p(val)/5.0)
    return max(1.0, score)

def extract_brand(text_clean):
    if not text_clean: return ""
    m = re.search(r"\bby\s+([a-z0-9][a-z0-9\-\&\s]{1,40})\b", text_clean)
    if m: return m.group(1).strip()
    m = re.search(r"\bbrand\s*[:\-]\s*([a-z0-9][a-z0-9\-\&\s]{1,40})\b", text_clean)
    if m: return m.group(1).strip()
    m = re.match(r"^([a-z0-9\&]{2,20})[\s\-|,]", text_clean)
    if m: return m.group(1).strip()
    return ""

def canonicalize_brand(b):
    b = (b or "").lower().strip()
    b = re.sub(r"[^a-z0-9\-\&\s]", " ", b)
    b = re.sub(r"\b(inc|ltd|pvt|limited|electronics|india|corp|co)\b", "", b)
    b = re.sub(r"\s+", " ", b).strip()
    b = BRAND_CORRECTIONS.get(b, b)
    return b

def pretty_brand(b):
    if not b: return ""
    toks = b.split()
    out=[]
    for t in toks:
        if len(t)<=3 and t.isalpha(): out.append(t.upper())
        elif t.lower() in {"and","&"}: out.append("&")
        else: out.append(t.capitalize())
    return " ".join(out)

def token_counts(series):
    from collections import Counter
    c = Counter()
    for s in series:
        if not isinstance(s, str): continue
        for tok in s.split():
            c[tok] += 1
    return pd.DataFrame({"token": list(c.keys()), "count": list(c.values())}).sort_values("count", ascending=False)

def derive_category_from_tokens(text, vocab_top):
    # seed-based; if no seed matches, fallback to most frequent token hints
    for cat, kws in CATEGORY_SEEDS.items():
        for k in kws:
            if f" {k} " in f" {text} ":
                return cat
    # heuristic: if a top token appears, map a few common ones
    for tok in vocab_top:
        if f" {tok} " in f" {text} ":
            if tok in {"phone","mobile","smartphone"}: return "phone"
            if tok in {"laptop","notebook","macbook"}: return "laptop"
            if tok in {"headphone","earbuds","earphone","speaker","soundbar","tws"}: return "audio"
            if tok in {"camera","dslr","webcam","lens"}: return "camera"
            if tok in {"watch","smartwatch","band"}: return "watch"
            if tok in {"tv","television","oled","qled","uhd","4k"}: return "tv"
            if tok in {"case","cover","charger","cable","adapter","protector","strap"}: return "accessory"
            if tok in {"fridge","refrigerator","washing","microwave","oven","dishwasher","purifier"}: return "appliance"
    return "other"

def robust_log_target(price, ipq):
    ipq = np.clip(ipq, MIN_IPQ, None)
    return np.log1p(price / ipq)

def inverse_log_target(ylog, ipq):
    return np.expm1(ylog) * np.clip(ipq, MIN_IPQ, None)

def mad(arr):
    med = np.median(arr)
    return np.median(np.abs(arr - med)) + 1e-12

# ----------------------- EDA HELPERS -----------------------
def plot_hist(series, title, path, bins=50, xlabel="value"):
    plt.figure(figsize=(8,4))
    data = series.dropna().values
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel("count")
    save_fig(path)

def plot_missingness(df, path):
    miss = df.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(10,4))
    plt.bar(miss.index.astype(str), miss.values)
    plt.title("Missingness ratio by column")
    plt.xticks(rotation=90)
    plt.ylabel("ratio")
    save_fig(path)

def corr_summary(df, target_col, out_csv):
    num = df.select_dtypes(include=[np.number]).copy()
    if target_col not in num.columns:
        num[target_col] = df[target_col].astype(float)
    # Spearman is safer for non-linear; Pearson also reported
    res=[]
    y = num[target_col].values
    for c in num.columns:
        if c==target_col: continue
        x = num[c].values
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.nanstd(x)>0:
            s = pd.Series(x).corr(pd.Series(y), method="spearman")
            p = pd.Series(x).corr(pd.Series(y), method="pearson")
            res.append((c, s, p))
    out = pd.DataFrame(res, columns=["feature","spearman","pearson"]).sort_values("spearman", ascending=False)
    out.to_csv(out_csv, index=False)
    return out

# ----------------------- MAIN -----------------------
def main():
    print(f"[{ts()}] Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)

    if "catalog_content" not in train.columns or "catalog_content" not in test.columns:
        raise RuntimeError("Both train and test must contain 'catalog_content'.")
    if "price" not in train.columns:
        raise RuntimeError("train.csv must contain 'price'.")

    # ---- Basic EDA: shapes & dtypes
    with open(EDA_DIR / "00_info.txt", "w", encoding="utf-8") as f:
        f.write(f"train shape: {train.shape}\n")
        f.write(f"test shape : {test.shape}\n\n")
        f.write("train dtypes:\n")
        f.write(str(train.dtypes)); f.write("\n\n")
        f.write("test dtypes:\n")
        f.write(str(test.dtypes)); f.write("\n")

    # ---- Missingness plots
    print(f"[{ts()}] Missingness...")
    plot_missingness(train, EDA_DIR / "01_missing_train.png")
    plot_missingness(test,  EDA_DIR / "01_missing_test.png")

    # ---- Text cleaning
    print(f"[{ts()}] Cleaning text...")
    train["catalog_clean"] = train["catalog_content"].fillna("").map(basic_text_clean)
    test["catalog_clean"]  = test["catalog_content"].fillna("").map(basic_text_clean)

    # ---- Text length & tokens
    print(f"[{ts()}] Text length / token stats...")
    for df, tag in [(train,"train"), (test,"test")]:
        df["title_len"]   = df["catalog_clean"].map(len)
        df["token_count"] = df["catalog_clean"].map(lambda s: len(s.split()))
        df["num_count"]   = df["catalog_clean"].map(lambda s: len(re.findall(r"\d+", s)))
        df["digit_share"] = (df["num_count"] / df["token_count"].replace(0, np.nan)).fillna(0)

        plot_hist(df["title_len"],   f"title_len ({tag})",   EDA_DIR / f"02_title_len_{tag}.png", bins=60, xlabel="chars")
        plot_hist(df["token_count"], f"token_count ({tag})", EDA_DIR / f"02_token_count_{tag}.png", bins=60, xlabel="tokens")
        plot_hist(df["num_count"],   f"num_count ({tag})",   EDA_DIR / f"02_num_count_{tag}.png", bins=60, xlabel="digits")
        plot_hist(df["digit_share"], f"digit_share ({tag})", EDA_DIR / f"02_digit_share_{tag}.png", bins=60, xlabel="num_count/token_count")

    # ---- Price distributions (train only)
    print(f"[{ts()}] Price distribution...")
    train["price"] = train["price"].astype(float).clip(MIN_PRICE)
    plot_hist(train["price"], "price (train)", EDA_DIR / "03_price_hist.png", bins=80, xlabel="price")

    # ---- IPQ
    print(f"[{ts()}] Computing IPQ...")
    train["ipq"] = train["catalog_content"].map(extract_ipq)
    test["ipq"]  = test["catalog_content"].map(extract_ipq)

    # ---- Brand extraction + canonicalization
    print(f"[{ts()}] Brand extraction...")
    train["brand_raw"] = train["catalog_clean"].map(extract_brand)
    test["brand_raw"]  = test["catalog_clean"].map(extract_brand)

    train["brand_norm"] = train["brand_raw"].map(canonicalize_brand)
    test["brand_norm"]  = test["brand_raw"].map(canonicalize_brand)

    # ---- Auto build a top-token vocabulary from whole corpus (train+test)
    print(f"[{ts()}] Building token vocabulary...")
    full_tokens = token_counts(pd.concat([train["catalog_clean"], test["catalog_clean"]], axis=0))
    full_tokens.to_csv(EDA_DIR / "04_token_frequency.csv", index=False)
    # Take top 300 frequent non-trivial tokens to help category heuristic
    vocab_top = [t for t in full_tokens["token"].head(300).tolist() if len(t)>=3]

    # ---- Category extraction (seeded + vocab-assisted)
    print(f"[{ts()}] Category derivation...")
    train["category_norm"] = train["catalog_clean"].map(lambda s: derive_category_from_tokens(s, vocab_top))
    test["category_norm"]  = test["catalog_clean"].map(lambda s: derive_category_from_tokens(s, vocab_top))

    # ---- Collapse rare brands/categories
    print(f"[{ts()}] Collapsing rare brands/categories...")
    brand_counts = train["brand_norm"].replace("", np.nan).dropna().value_counts()
    cat_counts   = train["category_norm"].value_counts()

    b_min = max(int(BRAND_MIN_FRAC*len(train)), ABS_MIN)
    c_min = max(int(CAT_MIN_FRAC*len(train)), ABS_MIN)

    keep_brands = set(brand_counts[brand_counts>=b_min].index)
    keep_cats   = set(cat_counts[cat_counts>=c_min].index)

    train["brand_norm"] = train["brand_norm"].apply(lambda b: b if b in keep_brands and b!="" else "other")
    test["brand_norm"]  = test["brand_norm"].apply(lambda b: b if b in keep_brands and b!="" else "other")
    train["category_norm"] = train["category_norm"].apply(lambda c: c if c in keep_cats else "other")
    test["category_norm"]  = test["category_norm"].apply(lambda c: c if c in keep_cats else "other")

    # Save brand/category maps
    brand_map = {b: pretty_brand(b) for b in sorted(set(train["brand_norm"].unique()).union(test["brand_norm"].unique()))}
    with open(CLEAN_DIR / "brand_map.json", "w", encoding="utf-8") as f: json.dump(brand_map, f, ensure_ascii=False, indent=2)
    with open(CLEAN_DIR / "category_map.json", "w", encoding="utf-8") as f: json.dump(sorted(list(set(train["category_norm"].unique()).union(test["category_norm"].unique()))), f, ensure_ascii=False, indent=2)

    # ---- Group combos
    train["brand_cat"] = train["brand_norm"] + "|" + train["category_norm"]
    test["brand_cat"]  = test["brand_norm"]  + "|" + test["category_norm"]

    # ---- Correlations (train) vs price and vs log target
    print(f"[{ts()}] Correlation summaries...")
    # Create numeric feature frame for corr
    corr_frame = train[["price","ipq","title_len","token_count","num_count","digit_share"]].copy()
    corr_frame["price_per_ipq"] = corr_frame["price"]/np.clip(train["ipq"], MIN_IPQ, None)
    corr_summary(corr_frame, "price", EDA_DIR / "05_corr_vs_price.csv")
    ylog = robust_log_target(train["price"], train["ipq"])
    corr_frame2 = corr_frame.copy()
    corr_frame2["ylog"] = ylog
    corr_summary(corr_frame2, "ylog", EDA_DIR / "05_corr_vs_ylog.csv")

    # ---- Outlier handling per brand×category on log target
    print(f"[{ts()}] Outlier handling ({CLIP_METHOD}, k={MAD_K})...")
    train["_ylog"] = ylog

    def group_winsorize(g):
        arr = g["_ylog"].values
        med = np.median(arr); m = mad(arr)
        lower = med - MAD_K*1.4826*m
        upper = med + MAD_K*1.4826*m
        ycap = np.clip(arr, lower, upper)
        g = g.copy()
        g["_was_capped"] = (arr<lower) | (arr>upper)
        g["_ylog"] = ycap
        g["price"] = inverse_log_target(g["_ylog"].values, g["ipq"].values)
        return g

    def group_mark_remove(g):
        arr = g["_ylog"].values
        med = np.median(arr); m = mad(arr)
        z = np.abs((arr - med)/(1.4826*m))
        g = g.copy()
        g["_is_outlier"] = z > MAD_K
        return g

    if CLIP_METHOD == "winsorize":
        marked = train.groupby("brand_cat", group_keys=False).apply(group_winsorize, include_groups=False)
        marked[marked["_was_capped"]].to_csv(CLEAN_DIR / "outlier_report.csv", index=False)
        train_clean = marked.drop(columns=["_was_capped"])
    else:
        marked = train.groupby("brand_cat", group_keys=False).apply(group_mark_remove)
        marked[marked["_is_outlier"]].to_csv(CLEAN_DIR / "outlier_report.csv", index=False)
        train_clean = marked[~marked["_is_outlier"]].drop(columns=["_is_outlier"])

    # ---- Final tidy features
    for df in (train_clean, test):
        df["brand_pretty"] = df["brand_norm"].map(pretty_brand)

    # ---- Save cleaned datasets
    keep_cols_train = sorted(list(set(
        ["sample_id","price","catalog_content","catalog_clean","ipq",
         "brand_raw","brand_norm","brand_pretty","category_norm","brand_cat",
         "title_len","token_count","num_count","digit_share"]
    )))
    keep_cols_test = [c for c in keep_cols_train if c!="price"]

    train_clean = train_clean[[c for c in keep_cols_train if c in train_clean.columns]].copy()
    test_clean  = test[[c for c in keep_cols_test  if c in test.columns]].copy()

    train_clean.to_csv(CLEAN_DIR / "train_clean.csv", index=False)
    test_clean.to_csv(CLEAN_DIR / "test_clean.csv", index=False)

    # ---- EDA counts for brands/categories
    train_clean["brand_norm"].value_counts().to_csv(EDA_DIR / "06_brand_counts_train.csv")
    test_clean["brand_norm"].value_counts().to_csv(EDA_DIR / "06_brand_counts_test.csv")
    train_clean["category_norm"].value_counts().to_csv(EDA_DIR / "06_category_counts_train.csv")
    test_clean["category_norm"].value_counts().to_csv(EDA_DIR / "06_category_counts_test.csv")

    # ---- Price by brand/category (box-plot surrogates with summary stats)
    print(f"[{ts()}] Summaries by brand/category...")
    brand_price = (train_clean.groupby("brand_norm")["price"]
                   .agg(["count","mean","median","std","min","max"])
                   .sort_values("count", ascending=False))
    brand_price.to_csv(EDA_DIR / "07_brand_price_summary.csv")
    cat_price = (train_clean.groupby("category_norm")["price"]
                 .agg(["count","mean","median","std","min","max"])
                 .sort_values("count", ascending=False))
    cat_price.to_csv(EDA_DIR / "07_category_price_summary.csv")

    # Simple bar plots of top-30 brand/category counts
    for name, vc, out in [
        ("brand", train_clean["brand_norm"].value_counts().head(30), EDA_DIR / "08_top_brands.png"),
        ("category", train_clean["category_norm"].value_counts().head(30), EDA_DIR / "08_top_categories.png"),
    ]:
        plt.figure(figsize=(10,6))
        plt.bar(vc.index.astype(str), vc.values)
        plt.title(f"Top {name}s by frequency (train)")
        plt.xticks(rotation=90); plt.ylabel("count")
        save_fig(out)

    # Price distribution per top categories (overlay hist per cat one by one)
    top_cats = train_clean["category_norm"].value_counts().head(6).index.tolist()
    for cat in top_cats:
        sub = train_clean.loc[train_clean["category_norm"]==cat, "price"]
        plot_hist(sub, f"price | category={cat}", EDA_DIR / f"09_price_hist_cat_{cat}.png", bins=60, xlabel="price")

    # Done
    print(f"[{ts()}] Saved cleaned data -> {CLEAN_DIR/'train_clean.csv'} ; {CLEAN_DIR/'test_clean.csv'}")
    print(f"[{ts()}] Reports & plots in -> {EDA_DIR}")
    print(f"[{ts()}] Done.")

if __name__ == "__main__":
    # Optional CLI overrides
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_method", type=str, default=CLIP_METHOD, choices=["winsorize","remove"])
    parser.add_argument("--mad_k", type=float, default=MAD_K)
    args = parser.parse_args()
    CLIP_METHOD = args.clip_method
    MAD_K = args.mad_k
    main()