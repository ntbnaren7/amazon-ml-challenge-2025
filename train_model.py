# SMART-PRICING-ML-NN/train_text_models_v3_fix.py
# Handles object columns in structured features (factorizes them safely).
# Works with outputs from extract_features_text_only_v2.py (768D SVD + rich features)

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import RidgeCV
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
EDA_DIR = DATA_DIR / "eda_outputs"
OUT_DIR = ROOT / "outputs_mm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "sample_id"
PRICE_COL = "price"

def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.zeros_like(y_true, float)
    m = denom != 0
    diff[m] = np.abs(y_true[m] - y_pred[m]) / denom[m]
    return diff.mean() * 100

def logify(A, cols):
    B = A.copy()
    if cols:
        B[:, cols] = np.log1p(np.clip(B[:, cols], 0, None))
    return B

def kfold_target_encode(train_df, test_df, col, target, n_splits=5, smoothing=50):
    global_mean = train_df[target].mean()
    oof = np.zeros(len(train_df), dtype=np.float32)
    te_test = np.zeros(len(test_df), dtype=np.float32)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, va_idx in kf.split(train_df):
        tr, va = train_df.iloc[tr_idx], train_df.iloc[va_idx]
        stats = tr.groupby(col)[target].agg(["mean", "count"])
        stats["te"] = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        oof[va_idx] = va[col].map(stats["te"]).fillna(global_mean)
    stats_full = train_df.groupby(col)[target].agg(["mean", "count"])
    stats_full["te"] = (stats_full["count"] * stats_full["mean"] + smoothing * global_mean) / (stats_full["count"] + smoothing)
    te_test = test_df[col].map(stats_full["te"]).fillna(global_mean)
    return oof.astype(np.float32), te_test.astype(np.float32)

# ---------- Load ----------
train_feats = pd.read_csv(CLEAN_DIR / "train_features.csv")
test_feats  = pd.read_csv(CLEAN_DIR / "test_features.csv")
X_text_train = np.load(EDA_DIR / "X_text_svd_train.npy")  # (N, 768)
X_text_test  = np.load(EDA_DIR / "X_text_svd_test.npy")

y = train_feats[PRICE_COL].astype(float).values
train_ids = train_feats[ID_COL].values
test_ids  = test_feats[ID_COL].values

# ---------- Build structured matrices with safe handling of object columns ----------
exclude = {ID_COL, PRICE_COL}
struct_cols = [c for c in train_feats.columns if c not in exclude]

# Split numeric vs object
num_cols = train_feats[struct_cols].select_dtypes(include=[np.number]).columns.tolist()
obj_cols = [c for c in struct_cols if c not in num_cols]

# Start with numeric block
struct_train_df = train_feats[num_cols].copy()
struct_test_df  = test_feats[num_cols].copy()

# Factorize object columns with a SHARED codebook (train+test)
for c in obj_cols:
    all_vals = pd.concat([train_feats[c].astype(str), test_feats[c].astype(str)], ignore_index=True)
    codes, uniques = pd.factorize(all_vals, sort=True)
    struct_train_df[c] = codes[:len(train_feats)].astype(np.int32)
    struct_test_df[c]  = codes[len(train_feats):].astype(np.int32)

# Convert to arrays
X_struct = struct_train_df.astype(np.float32).values
X_struct_test = struct_test_df.astype(np.float32).values

# ---------- Logify skewed features (by name patterns) ----------
all_struct_cols = struct_train_df.columns.tolist()
log_targets = [i for i, c in enumerate(all_struct_cols)
               if any(k in c.lower() for k in [
                   "quantity","weight","volume","pack","count","size",
                   "len","word","bullet","dim","entropy","ratio","units","area"
               ])]
X_struct_log = logify(X_struct, log_targets)
X_struct_test_log = logify(X_struct_test, log_targets)

# ---------- Semantic clusters on text embeddings ----------
n_clusters = 60
km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
cluster_id_train = km.fit_predict(X_text_train)
cluster_id_test  = km.predict(X_text_test)

train_tmp = pd.DataFrame({
    "brand_code": train_feats["brand_code"].astype(np.int32),
    "cluster_id": cluster_id_train.astype(np.int32),
    PRICE_COL: y
})
test_tmp = pd.DataFrame({
    "brand_code": test_feats["brand_code"].astype(np.int32),
    "cluster_id": cluster_id_test.astype(np.int32),
})

brand_te_tr,   brand_te_te   = kfold_target_encode(train_tmp, test_tmp, "brand_code", PRICE_COL)
cluster_te_tr, cluster_te_te = kfold_target_encode(train_tmp, test_tmp, "cluster_id", PRICE_COL)

X_te_train = np.vstack([brand_te_tr, cluster_te_tr]).T.astype(np.float32)
X_te_test  = np.vstack([brand_te_te, cluster_te_te]).T.astype(np.float32)

# ---------- Final design matrices ----------
X_ridge = X_text_train
X_ridge_test = X_text_test

X_lgb = np.hstack([X_text_train, X_struct_log, X_te_train]).astype(np.float32)
X_lgb_test = np.hstack([X_text_test,  X_struct_test_log, X_te_test]).astype(np.float32)

y_log = np.log1p(y)

# ---------- Stratified CV on price bins ----------
bins = pd.qcut(y, q=20, duplicates="drop", labels=False)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgb_params = dict(
    objective="l1",
    metric="mae",
    learning_rate=0.02,
    num_leaves=128,
    max_depth=-1,
    min_child_samples=30,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    reg_alpha=0.3,
    reg_lambda=0.5,
    n_estimators=9000,
    n_jobs=-1,
    verbose=-1
)

oof_lgb_log   = np.zeros_like(y_log)
oof_ridge_log = np.zeros_like(y_log)
pred_lgb_log  = np.zeros(len(test_ids))
pred_ridge_log= np.zeros(len(test_ids))

for fold, (tr, va) in enumerate(skf.split(X_lgb, bins), 1):
    Xtr_lgb, Xva_lgb = X_lgb[tr], X_lgb[va]
    ytr_log, yva_log = y_log[tr], y_log[va]

    print(f"\n=== Fold {fold} ===")
    m_lgb = lgb.LGBMRegressor(**lgb_params)
    m_lgb.fit(Xtr_lgb, ytr_log,
              eval_set=[(Xva_lgb, yva_log)],
              eval_metric="mae",
              callbacks=[lgb.early_stopping(300, verbose=False)])
    oof_lgb_log[va] = m_lgb.predict(Xva_lgb)
    pred_lgb_log   += m_lgb.predict(X_lgb_test) / skf.n_splits
    print(f"Fold {fold} LGBM SMAPE: {smape(np.expm1(yva_log), np.expm1(oof_lgb_log[va])):.2f}%")

    m_ridge = RidgeCV(alphas=np.logspace(-3, 2, 12))
    m_ridge.fit(X_ridge[tr], ytr_log)
    oof_ridge_log[va] = m_ridge.predict(X_ridge[va])
    pred_ridge_log   += m_ridge.predict(X_ridge_test) / skf.n_splits

# ---------- Blend ----------
p1 = np.expm1(oof_lgb_log)
p2 = np.expm1(oof_ridge_log)
best_w, best_s = 0.5, 1e9
for w in np.linspace(0, 1, 51):
    s = smape(y, w*p1 + (1-w)*p2)
    if s < best_s:
        best_s, best_w = s, w

print(f"\nOOF SMAPE: {best_s:.2f}% | LGB weight={best_w:.2f}, Ridge={1-best_w:.2f}")
print(f"OOF MAE: {mean_absolute_error(y, best_w*p1 + (1-best_w)*p2):.3f}")

test_pred = best_w*np.expm1(pred_lgb_log) + (1-best_w)*np.expm1(pred_ridge_log)
test_pred = np.clip(test_pred, 0, np.percentile(test_pred, 99.9))

sub_path = OUT_DIR / "submission_text_only_v3.csv"
pd.DataFrame({ID_COL: test_ids, "price": test_pred}).to_csv(sub_path, index=False)
print(f"Saved submission: {sub_path}")
